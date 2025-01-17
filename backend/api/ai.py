import os
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentType, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_openai import OpenAI
from langchain.chains import LLMChain, ConversationChain
from dotenv import load_dotenv
from google.generativeai import GenerativeModel
import google.generativeai as genai
import openai
from langchain.utilities import SerpAPIWrapper
import streamlit as st






# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# gemni api key
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

def initialize_openai():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("OPENAI_API_KEY environment variable not set. Falling back to Gemini.")
        return None
    try:
        llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key, verbose=True)
        llm.invoke("test")
        return llm
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        print("Falling back to Gemini...")
        return None

def initialize_gemini():
    try:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=GOOGLE_API_KEY)
        model = GenerativeModel('gemini-pro')
        return model
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None


"""
async def  real_time_search(query):
this function will take a query and return the search results in real time using the serpapi
this function will will be able to use the serpapi to get the search results in real time and gemini to generate the search results
"""
async def real_time_search(query):
    """
    Performs a real-time search using SerpAPI and optionally summarizes with Gemini.
    """
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_api_key:
        raise Exception("SERPAPI_API_KEY environment variable not set.")

    search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    results = search.run(query)

    gemini_model = initialize_gemini()
    if gemini_model:
        try:  # Summary generation might fail, so handle it
            summary = gemini_model.generate_content(f"Summarize these search results: {results}").text
            return f"Search Results:\n{results}\n\nSummary:\n{summary}"

        except Exception as e:
             print(f"Gemini summarization error: {e}")
             return f"Search Results:\n{results}\n\nGemini could not summarize the results."

    else: # No Gemini, return raw results
        return f"Search Results:\n{results}"


def create_chain(use_gemini=False):
    prompt_template_name = PromptTemplate(
        input_variables=["AI_Agent"],
        template="I want you to be an {AI_Agent} ans assistant, you'll be helping me out with some tasks."
    )
    
    if use_gemini:
        model = initialize_gemini()
        if model is None:
            raise Exception("Neither OpenAI nor Gemini APIs are working")
        
        # Custom LLMChain-like implementation for Gemini
        class GeminiChain:
            def __init__(self, model, prompt):
                self.model = model
                self.prompt = prompt
                self.memory = ConversationBufferMemory()
            
            def run(self, AI_Agent):
                formatted_prompt = self.prompt.template.format(AI_Agent=AI_Agent)
                response = self.model.generate_content(formatted_prompt)
                self.memory.save_context({"input": formatted_prompt}, {"output": response.text})
                return response.text
        
        return GeminiChain(model, prompt_template_name)
    else:
        llm = OpenAI(temperature=0.7)
        memory = ConversationBufferMemory()
        return LLMChain(llm=llm, prompt=prompt_template_name, memory=memory)

def main():
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
    
    try:
        # Try to initialize with OpenAI first
        llm = initialize_openai()
        if llm is not None:
            # Initialize OpenAI tools and agent
            tools = load_tools(["serpapi", "llm-math"], llm=llm)
            agent = initialize_agent(
                tools,
                llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            chain = create_chain(use_gemini=False)
        else:
            # Fallback to Gemini
            chain = create_chain(use_gemini=True)
        
        # Run the chain
        # name = chain.run("Nigerian")
        # # print("Generated restaurant name:", name)
        # # print("\nMemory buffer:", chain.memory.buffer)
        
        # Additional conversation handling
        if llm is not None:
            convo = ConversationChain(llm=llm)
            # print("\nConversation prompt:", convo.prompt.template)
            # print("\nCapital of Nigeria:", convo.run("What is the capital of Nigeria?"))
            # print("\nConversation memory:", convo.memory)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
