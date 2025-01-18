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
from langchain_community.utilities import SerpAPIWrapper
import streamlit as st
import asyncio
import requests
import uuid
import datetime


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
        return None, False  # Return a flag indicating failure
    try:
        llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key, verbose=True)
        llm.invoke("test")
        return llm, True  # Return a flag indicating success
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        print("Falling back to Gemini...")
        return None, False  # Return a flag indicating failure

def initialize_gemini():
    try:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            print("GEMINI_API_KEY not found in environment variables")
            return None
            
        # Configure genai with safety settings and timeout
        genai.configure(
            api_key=GEMINI_API_KEY,
            transport="rest"  # Use REST instead of gRPC
        )
        
        # Initialize the model with specific configuration
        model = GenerativeModel(
            model_name='gemini-pro',
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "block_none",
                "HARM_CATEGORY_HATE_SPEECH": "block_none",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none"
            }
        )
        
        return model
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None


# """
# async def  real_time_search(query):
# this function will take a query and return the search results in real time using the serpapi
# this function will will be able to use the serpapi to get the search results in real time and gemini to generate the search results
# """
async def real_time_search(query):
    """
    Performs a real-time search using SerpAPI and provides detailed results with context and summary.
    """
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_api_key:
        raise Exception("SERPAPI_API_KEY environment variable not set.")

    try:
        # Perform the search
        search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
        raw_results = search.run(query)
        
        # Convert to dictionary if results are in list format
        if isinstance(raw_results, list):
            results = {"organic_results": raw_results}
        else:
            results = raw_results

        # Initialize Gemini for summarization
        gemini_model = initialize_gemini()
        if not gemini_model:
            return "Error: Could not initialize Gemini model for summarization."

        # Extract and format the search results
        formatted_results = "Search Results for: " + query + "\n\n"
        
        # Process organic results
        if "organic_results" in results:
            for idx, result in enumerate(results["organic_results"][:5], 1):
                formatted_results += f"{idx}. Title: {result.get('title', 'No title')}\n"
                formatted_results += f"   Link: {result.get('link', 'No link')}\n"
                formatted_results += f"   Snippet: {result.get('snippet', 'No snippet available')}\n\n"

        # Generate context and summary using Gemini
        summary_prompt = f"""
        Based on the following search results, provide:
        1. A comprehensive summary
        2. Key points or facts
        3. Additional context if relevant

        Search Query: {query}
        Results: {formatted_results}
        """

        try:
            summary = gemini_model.generate_content(summary_prompt).text
            
            # Combine everything into a structured response
            final_response = f"""
=== Search Query ===
{query}

=== Detailed Results ===
{formatted_results}

=== Analysis and Summary ===
{summary}
"""
            return final_response

        except Exception as e:
            return f"Error generating summary: {str(e)}\n\nRaw Results:\n{formatted_results}"

    except Exception as e:
        return f"Error in search process: {str(e)}"


async def scrape_webpages_with_serpapi(query):
    """
    Performs comprehensive web scraping and analysis using SerpAPI and Gemini.
    """
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_api_key:
        return "Error: SERPAPI_API_KEY environment variable not set."

    try:
        # Perform the search
        search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
        raw_results = search.run(query)

        if not raw_results:
            return f"No search results found for query: {query}"

        # Handle the case where raw_results is a string
        if isinstance(raw_results, str):
            urls = [line.strip() for line in raw_results.split('\n') if line.strip().startswith('http')]
            if not urls:
                return f"No valid URLs found in search results for query: {query}"
            results = [{"title": f"Result {i+1}", "link": url, "snippet": ""} for i, url in enumerate(urls)]
        else:
            results = raw_results if isinstance(raw_results, list) else raw_results.get("organic_results", [])
            if not results:
                return f"No valid results found for query: {query}"

        gemini_model = initialize_gemini()
        if not gemini_model:
            return "Error: Could not initialize Gemini model for analysis."

        comprehensive_results = []
        all_content = []
        max_urls_to_scrape = 5

        # Add query information
        comprehensive_results.append(f"Query: {query}\n")
        
        # Process search results overview
        search_overview = []
        valid_results = False
        
        for idx, result in enumerate(results[:max_urls_to_scrape], 1):
            title = result.get('title', f'Result {idx}') if isinstance(result, dict) else f'Result {idx}'
            url = result.get('link', '') if isinstance(result, dict) else result
            snippet = result.get('snippet', 'No preview available') if isinstance(result, dict) else ''
            
            if url:  # Only add if there's a valid URL
                valid_results = True
                search_overview.append(f"{idx}. {title}")
                search_overview.append(f"   URL: {url}")
                if snippet:
                    search_overview.append(f"   Preview: {snippet}")
                search_overview.append("")  # Empty line for spacing

        if valid_results:
            comprehensive_results.append("=== Search Results Overview ===")
            comprehensive_results.extend(search_overview)
        
        # Process detailed content analysis
        content_analyses = []
        
        for idx, item in enumerate(results[:max_urls_to_scrape], 1):
            url = item.get('link', '') if isinstance(item, dict) else item
            title = item.get('title', f'Result {idx}') if isinstance(item, dict) else f'Result {idx}'
            
            if url:
                try:
                    response = requests.get(url, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }, timeout=15)
                    
                    if response.ok and gemini_model:
                        content = response.text[:3000]
                        
                        analysis_prompt = f"""
                        Analyze this content about {query}:
                        
                        Title: {title}
                        URL: {url}
                        Content: {content}

                        Please provide:
                        1. Main topic and key findings
                        2. Important facts and data points
                        3. Relevance to the query
                        4. Notable quotes or statements
                        5. Key takeaways
                        """
                        
                        analysis = gemini_model.generate_content(analysis_prompt).text
                        if analysis:
                            content_analyses.append(f"Source {idx}: {title}")
                            content_analyses.append(f"URL: {url}")
                            content_analyses.append(f"Analysis:\n{analysis}")
                            content_analyses.append("-" * 50)
                            content_analyses.append("")
                            
                            all_content.append({
                                "title": title,
                                "url": url,
                                "analysis": analysis
                            })
                            
                except Exception as e:
                    print(f"Error processing {url}: {e}")
                    continue

        if content_analyses:
            comprehensive_results.append("\n=== Detailed Content Analysis ===")
            comprehensive_results.extend(content_analyses)
        
        # Generate final synthesis only if we have content to analyze
        if all_content:
            synthesis_prompt = f"""
            Analyze these findings about "{query}":

            {', '.join([f"Source {i+1}: {content['analysis']}" for i, content in enumerate(all_content)])}

            Please provide:
            1. Comprehensive synthesis of all sources
            2. Main themes and patterns
            3. Key conclusions
            4. Different viewpoints or contradictions
            5. Recommendations for further research
            """
            
            try:
                final_synthesis = gemini_model.generate_content(synthesis_prompt).text
                if final_synthesis:
                    comprehensive_results.append("=== Overall Analysis and Synthesis ===")
                    comprehensive_results.append(final_synthesis)
            except Exception as e:
                print(f"Error generating final synthesis: {e}")

        # Only return results if we have actual content
        if len(comprehensive_results) > 1:  # More than just the query
            return "\n".join(comprehensive_results)
        else:
            return f"Unable to generate comprehensive analysis for query: {query}. No valid content could be processed."

    except Exception as e:
        return f"Error in scraping process: {str(e)}"


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
                # self.memory = ConversationBufferMemory(memory_key="chat_history")
            
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
    


# function to create tasks, in this function a user will be able to create a task and assign it to an agent, exmaple, a user can create a reminder, alarm, or a to-do list, and assign it to an agent, the agent can also go create emails, messages, or any other task, this would be done using langchain and google gemini



def main():
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
    
    try:
        # Try to initialize with OpenAI first
        llm, is_openai_working = initialize_openai()
        if is_openai_working:
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


        # Run the search and display the results
            st.title("Real-Time Search App")

            query = st.text_input("Enter your search query:")
            if st.button("Search"):
                if query:  # Check if the user entered a query
                    asyncio.run(run_search_and_display(query)) # Run the search
                else:
                    st.warning("Please enter a search query.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

async def run_search_and_display(query):
    results = await real_time_search(query)
    st.write(results)

async def create_task(task_type, task_details, due_date=None, priority="medium"):
    """
    Creates and manages various types of tasks using Gemini AI.
    """
    try:
        gemini_model = initialize_gemini()
        if not gemini_model:
            return {"status": "error", "message": "Could not initialize Gemini model"}

        # Base task metadata
        task = {
            "task_id": str(uuid.uuid4()),
            "type": task_type,
            "priority": priority,
            "due_date": due_date,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "created"
        }

        # Task-specific processing
        if task_type.lower() == "email":
            email_prompt = f"""
            Generate a professional email based on the following details:
            To: {task_details.get('to')}
            Subject: {task_details.get('subject')}
            Content Brief: {task_details.get('content')}
            Priority: {priority}

            Please provide a professional email with:
            1. A refined subject line
            2. Professional greeting
            3. Main body content (well-structured paragraphs)
            4. Professional closing
            5. Follow-up recommendations
            """
            
            response = gemini_model.generate_content(email_prompt)
            
            task["email_content"] = {
                "to": task_details.get('to'),
                "subject": task_details.get('subject'),
                "generated_content": response.text
            }

        elif task_type.lower() in ["reminder", "alarm"]:
            reminder_prompt = f"""
            Create a structured reminder plan for:
            Task: {task_details.get('content')}
            Due Date: {due_date}
            Priority: {priority}

            Please provide:
            1. Detailed reminder schedule
            2. Key checkpoints
            3. Action items
            4. Progress tracking steps
            """
            
            response = gemini_model.generate_content(reminder_prompt)
            
            task["reminder_details"] = {
                "content": task_details.get('content'),
                "generated_plan": response.text
            }

        elif task_type.lower() == "todo":
            todo_prompt = f"""
            Create a detailed todo list for:
            Task: {task_details.get('content')}
            Due Date: {due_date}
            Priority: {priority}

            Please provide:
            1. Main task breakdown
            2. Subtasks with priorities
            3. Timeline recommendations
            4. Resource requirements
            5. Success metrics
            """
            
            response = gemini_model.generate_content(todo_prompt)
            
            task["todo_details"] = {
                "content": task_details.get('content'),
                "generated_plan": response.text
            }

        return {
            "status": "success",
            "task": task,
            "message": "Task created successfully"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error creating task: {str(e)}"
        }

async def handle_task_creation(task_request):
    """
    Handles task creation requests from users.
    """
    try:
        required_fields = ["task_type", "task_details"]
        if not all(field in task_request for field in required_fields):
            return {"status": "error", "message": "Missing required fields"}

        response = await create_task(
            task_type=task_request["task_type"],
            task_details=task_request["task_details"],
            due_date=task_request.get("due_date"),
            priority=task_request.get("priority", "medium")
        )

        return response

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in task creation: {str(e)}"
        }

if __name__ == "__main__":
    main()
