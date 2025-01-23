import os
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentType, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_openai import OpenAI
# import googleapiclient
import googleapiclient.discovery 
from langchain.chains import LLMChain, ConversationChain
from dotenv import load_dotenv
from google.generativeai import GenerativeModel
import google.generativeai as genai
import openai
from langchain_community.utilities import SerpAPIWrapper
import asyncio
# import googleapiclient
# import googleapiclient.discovery 
import requests
import uuid
from datetime import datetime
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import base64
from email.mime.text import MIMEText
import pytz
from typing import Dict, Any, List
from bs4 import BeautifulSoup
import urllib.parse
import re
# import genai
import google.generativeai as genai
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# gemni api key
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Get the current directory where ai.py is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))  # Go up two levels to project root

# Define the path to the credentials file
CREDENTIALS_PATH = os.path.join(BASE_DIR, 'credentials.json')  # Ensure this points to the correct location
TOKEN_PATH = os.path.join(BASE_DIR, 'token.pickle')

# Add these constants at the top of the file
SCOPES = [
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.compose',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/calendar.events',
    # speech-to-text, text-to-speech, and language translation-

]

def check_credentials() -> Dict[str, Any]:
    """Check if Gmail credentials are properly configured"""
    try:
        service = get_gmail_service()
        if service:
            user_profile = service.users().getProfile(userId='me').execute()
            return {
                "status": "success",
                "email": user_profile['emailAddress']
            }
        return {
            "status": "error",
            "message": "Gmail service not initialized"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

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
    """Initialize Gemini model"""
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-pro')
        return model
    except Exception as e:
        print(f"Error initializing Gemini: {str(e)}")
        return None


# """
# async def  real_time_search(query):
# this function will take a query and return the search results in real time using the serpapi
# this function will will be able to use the serpapi to get the search results in real time and gemini to generate the search results
# """




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
    """
    Main function for backend initialization and testing
    """
    try:
        # Initialize OpenAI
        llm, is_openai_working = initialize_openai()
        if is_openai_working:
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
        
        return {
            "status": "success",
            "message": "Backend initialized successfully",
            "openai_status": is_openai_working,
            "chain": chain
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Backend initialization failed: {str(e)}"
        }

async def create_task(task_type, task_details, due_date=None, priority="medium"):
    """Create a task with the specified details"""
    try:
        gemini_model = initialize_gemini()
        if not gemini_model:
            return {"status": "error", "message": "Could not initialize Gemini model"}

        task = {
            "task_id": str(uuid.uuid4()),
            "type": task_type,
            "priority": priority,
            "due_date": due_date,
            "created_at": datetime.now().isoformat(),
            "status": "created"
        }

        if task_type.lower() == "email":
            email_prompt = f"""
            Generate a professional email based on these details:
            To: {task_details.get('to')}
            Subject: {task_details.get('subject', 'Meeting Discussion')}
            Content Brief: {task_details.get('content')}
            Priority: {priority}

            Requirements:
            1. Professional and courteous tone
            2. Clear and concise language
            3. Start with a professional greeting
            4. Include all relevant details from the content brief
            5. End with a professional closing (but no signature)
            6. Format with proper paragraphs and spacing
            
            Do not include a signature block - it will be added separately.
            """
            
            response = gemini_model.generate_content(email_prompt)
            task["email_content"] = {
                "to": task_details.get('to'),
                "subject": task_details.get('subject', 'Meeting Discussion'),
                "generated_content": response.text.strip()
            }

            # Send the email after creating the task
            email_response = await send_email(
                to=task["email_content"]["to"],
                subject=task["email_content"]["subject"],
                body=task["email_content"]["generated_content"]
            )
            if email_response["status"] == "error":
                return {
                    "status": "error",
                    "message": f"Email could not be sent: {email_response['message']}"
                }

        return {
            "status": "success",
            "task": task
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
            priority=task_request.get("priority", "medium")
        )

        return response

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in task creation: {str(e)}"
        }

async def send_email(to: str, subject: str, body: str) -> Dict[str, Any]:
    try:
        logger.info(f"Sending email to: {to}")
        service = get_gmail_service()
        if not service:
            logger.error("Could not initialize Gmail service")
            return {"status": "error", "message": "Could not initialize Gmail service"}

        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        user_profile = service.users().getProfile(userId='me').execute()
        message['from'] = user_profile['emailAddress']
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        body = {'raw': raw}

        logger.info(f"Email content:\n{message.as_string()}") #Added logging for email content

        sent_message = service.users().messages().send(userId='me', body=body).execute()
        logger.info(f"Email sent successfully. Message ID: {sent_message['id']}")
        return {
            "status": "success",
            "message_id": sent_message['id'],
            "details": {"to": to, "subject": subject, "from": message['from']}
        }
    except googleapiclient.errors.HttpError as e:
        logger.error(f"HTTP Error sending email: {e}")
        error_details = json.loads(e.content.decode()) # Try to parse the error details from the JSON response
        error_message = error_details.get('error', {}).get('message', 'Unspecified HTTP Error')
        return {"status": "error", "message": error_message}
    except Exception as e:
        logger.exception(f"Unexpected error sending email: {e}")  # Log the full stack trace
        return {"status": "error", "message": str(e)}

def get_gmail_service():
    """Initialize Gmail API service"""
    try:
        creds = None
        if os.path.exists(TOKEN_PATH):
            with open(TOKEN_PATH, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_PATH,  # Use the updated path here
                    SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(TOKEN_PATH, 'wb') as token:
                pickle.dump(creds, token)
        
        return build('gmail', 'v1', credentials=creds)
    except Exception as e:
        print(f"Error in get_gmail_service: {str(e)}")
        raise

def clean_text(text: str) -> str:
    """Clean scraped text by removing extra whitespace and special characters"""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with single newline
    return text.strip()

async def scrape_url(url: str) -> str:
    """Scrape content from a URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
            element.decompose()
        
        # Extract main content
        main_content = ""
        
        # Try to find main content container
        content_tags = soup.find_all(['article', 'main', 'div'], class_=re.compile(r'(content|article|post|entry)'))
        if content_tags:
            main_content = content_tags[0].get_text()
        else:
            # Fallback to paragraphs
            paragraphs = soup.find_all('p')
            main_content = ' '.join(p.get_text() for p in paragraphs)
        
        return clean_text(main_content)
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return ""

async def summarize_content(content: str, gemini_model) -> str:
    """Summarize content using Gemini"""
    try:
        summary_prompt = f"""
        Summarize this content concisely and clearly:
        {content[:10000]}  # Limit content length to avoid token limits
        
        Provide:
        1. Key Points
        2. Important Facts
        3. Relevant Dates
        4. Additional Context
        
        Format in markdown for readability.
        """
        
        summary = gemini_model.generate_content(summary_prompt)
        return summary.text.strip()
    except Exception as e:
        print(f"Error summarizing content: {str(e)}")
        return content[:1000] + "..."  # Fallback to truncated content



if __name__ == "__main__":
    main()
