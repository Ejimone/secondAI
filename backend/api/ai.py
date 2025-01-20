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
    'https://www.googleapis.com/auth/gmail.modify'
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
async def get_current_time(location: str) -> Dict[str, Any]:
    """Get current time for a specific location"""
    try:
        # Dictionary of common US timezone mappings
        us_timezones = {
            'eastern': 'US/Eastern',
            'central': 'US/Central',
            'mountain': 'US/Mountain',
            'pacific': 'US/Pacific',
            'alaska': 'US/Alaska',
            'hawaii': 'US/Hawaii'
        }
        
        # Default to showing all major US timezones if specific zone not specified
        if location.lower() == 'us' or location.lower() == 'usa':
            current_times = []
            for zone_name, timezone in us_timezones.items():
                tz = pytz.timezone(timezone)
                current_time = datetime.now(tz)
                current_times.append(f"🕐 {zone_name.title()}: {current_time.strftime('%I:%M %p')}")
            
            return {
                "status": "success",
                "data": "\n".join(current_times),
                "type": "time"
            }
        else:
            # Try to find the specific timezone
            try:
                tz = pytz.timezone(location)
                current_time = datetime.now(tz)
                return {
                    "status": "success",
                    "data": f"🕐 Current time in {location}: {current_time.strftime('%I:%M %p')}",
                    "type": "time"
                }
            except pytz.exceptions.UnknownTimeZoneError:
                return {
                    "status": "error",
                    "message": f"Could not find timezone for {location}"
                }
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def get_weather(location: str) -> Dict[str, Any]:
    """Get current weather for a location"""
    try:
        # Using OpenWeatherMap API
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            return {
                "status": "success",
                "data": "⚠️ Weather information is temporarily unavailable (API key not configured)",
                "type": "weather"
            }
        
        # Make the API request
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Get current time in user's timezone
            current_time = datetime.now().strftime('%I:%M %p')
            
            weather_info = f"""
🌍 Current Weather in {data['name']}, {data.get('sys', {}).get('country', '')}:

🌡️ Temperature: {data['main']['temp']}°C
🌡️ Feels like: {data['main']['feels_like']}°C
💧 Humidity: {data['main']['humidity']}%
🌪️ Wind: {data['wind']['speed']} m/s
☁️ Conditions: {data['weather'][0]['description'].capitalize()}

Updated at: {current_time}
            """
            return {
                "status": "success",
                "data": weather_info.strip(),
                "type": "weather"
            }
        elif response.status_code == 401:
            return {
                "status": "success",
                "data": "⚠️ Weather API key is invalid. Please check your configuration.",
                "type": "weather"
            }
        elif response.status_code == 404:
            return {
                "status": "success",
                "data": f"📍 Could not find weather data for '{location}'. Please check the location name.",
                "type": "weather"
            }
        else:
            return {
                "status": "success",
                "data": f"⚠️ Weather service returned status code: {response.status_code}",
                "type": "weather"
            }
            
    except requests.Timeout:
        return {
            "status": "success",
            "data": "⚠️ Weather service is taking too long to respond. Please try again.",
            "type": "weather"
        }
    except requests.RequestException as e:
        return {
            "status": "success",
            "data": f"⚠️ Error fetching weather data: {str(e)}",
            "type": "weather"
        }
    except Exception as e:
        return {
            "status": "success",
            "data": f"⚠️ Unexpected error: {str(e)}",
            "type": "weather"
        }

async def real_time_search(user_prompt: str) -> Dict[str, Any]:
    """Handle real-time information requests"""
    try:
        gemini_model = initialize_gemini()
        if not gemini_model:
            return {"status": "error", "message": "Could not initialize Gemini model"}
        
        # Analyze the type of real-time information needed
        analysis_prompt = f"""
        You are a request classifier. Analyze this request and return a JSON response.
        Request: "{user_prompt}"
        
        Rules:
        1. For weather requests, extract the city/location name
        2. For time requests, extract the location/timezone
        3. Always return a valid JSON object
        4. Default to "unknown" if location cannot be determined
        
        Return EXACTLY in this format:
        {{
            "type": "weather",
            "location": "london",
            "query_type": "current_weather"
        }}
        
        Example valid responses:
        For "what's the weather in london":
        {{"type": "weather", "location": "london", "query_type": "current_weather"}}
        
        For "time in new york":
        {{"type": "time", "location": "US/Eastern", "query_type": "current_time"}}
        """
        
        response = gemini_model.generate_content(analysis_prompt)
        response_text = response.text.strip()
        
        # Debug logging
        print(f"Gemini Response: {response_text}")
        
        try:
            request_info = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Raw response: {response_text}")
            return {
                "status": "success",
                "request_type": "REAL_TIME_INFO",
                "search_result": "⚠️ I had trouble understanding that request. Please try rephrasing it."
            }
        
        # Ensure required fields exist
        if not all(key in request_info for key in ["type", "location"]):
            return {
                "status": "success",
                "request_type": "REAL_TIME_INFO",
                "search_result": "⚠️ Could not determine the type of information needed. Please try rephrasing your request."
            }
        
        # Route to appropriate handler based on request type
        if request_info["type"] == "time":
            result = await get_current_time(request_info["location"])
        elif request_info["type"] == "weather":
            result = await get_weather(request_info["location"])
        else:
            return {
                "status": "success",
                "request_type": "REAL_TIME_INFO",
                "search_result": "I can currently only provide time and weather information. Other real-time data types are coming soon!"
            }
        
        if result["status"] == "success":
            return {
                "status": "success",
                "request_type": "REAL_TIME_INFO",
                "data_type": result["type"],
                "search_result": result["data"]
            }
        else:
            return {
                "status": "success",
                "request_type": "REAL_TIME_INFO",
                "search_result": f"⚠️ {result['message']}"
            }
            
    except Exception as e:
        print(f"Error in real_time_search: {str(e)}")  # Debug logging
        return {
            "status": "success",
            "request_type": "REAL_TIME_INFO",
            "search_result": f"⚠️ Error processing request: {str(e)}"
        }


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

async def web_search(query: str) -> Dict[str, Any]:
    """Perform web search, scrape URLs, and summarize content"""
    try:
        # Direct API call to SerpAPI
        params = {
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "engine": "google",
            "q": query,
            "num": 5,
            "gl": "us"
        }
        
        search_response = requests.get("https://serpapi.com/search", params=params)
        raw_results = search_response.json()
        
        print(f"Search query: {query}")  # Debug log
        
        if "error" in raw_results:
            return {
                "status": "success",
                "data": f"Search API error: {raw_results['error']}",
                "type": "search"
            }
        
        search_results = raw_results.get("organic_results", [])
        
        if not search_results:
            return {
                "status": "success",
                "data": f"No search results found for query: {query}",
                "type": "search"
            }
        
        gemini_model = initialize_gemini()
        if not gemini_model:
            return {
                "status": "error",
                "message": "Could not initialize Gemini model"
            }
        
        # Process each result individually for detailed summaries
        detailed_summaries = []
        for result in search_results[:3]:
            try:
                url = result.get('link', '')
                title = result.get('title', 'No title')
                snippet = result.get('snippet', '')
                
                # Scrape content from URL
                content = await scrape_url(url) if url else ""
                
                # Generate detailed summary for each source
                source_prompt = f"""
                Provide a comprehensive analysis of this content about {query}:

                Title: {title}
                URL: {url}
                Content: {content if content else snippet}

                Requirements:
                1. Write at least 250 words
                2. Include specific details, facts, and figures
                3. Mention dates and relevant context
                4. Analyze the significance of the information
                5. Include any controversies or different perspectives
                6. Explain how this information relates to {query}

                Format the response with these sections:
                1. Main Points (detailed explanation)
                2. Key Facts and Figures
                3. Context and Background
                4. Analysis and Implications
                5. Related Developments
                """
                
                source_summary = gemini_model.generate_content(source_prompt)
                detailed_summaries.append({
                    "title": title,
                    "url": url,
                    "summary": source_summary.text.strip()
                })
                
            except Exception as e:
                print(f"Error processing result {url}: {str(e)}")
                continue
        
        # Generate comprehensive overview
        overview_prompt = f"""
        Create a comprehensive overview of {query} based on all these sources:

        {[summary['summary'] for summary in detailed_summaries]}

        Requirements:
        1. Write at least 300 words
        2. Synthesize information from all sources
        3. Highlight key themes and patterns
        4. Include contrasting viewpoints if any
        5. Provide temporal context (recent vs historical information)
        6. Explain broader implications

        Format with these sections:
        1. Executive Summary
        2. Detailed Analysis
        3. Key Developments
        4. Implications and Future Outlook
        """
        
        overview_response = gemini_model.generate_content(overview_prompt)
        overview = overview_response.text.strip()
        
        # Format the final output
        formatted_result = f"""
# Comprehensive Analysis: {query}

{overview}

## Detailed Source Summaries

"""
        # Add individual source summaries
        for idx, summary in enumerate(detailed_summaries, 1):
            formatted_result += f"""
### Source {idx}: {summary['title']}
[Link to original article]({summary['url']})

{summary['summary']}

---
"""
        
        return {
            "status": "success",
            "data": formatted_result,
            "type": "search"
        }
        
    except Exception as e:
        print(f"Search error: {str(e)}")  # Debug log
        return {
            "status": "success",
            "data": f"""
# Search Error

⚠️ An error occurred while searching for "{query}":
- Error details: {str(e)}

Please try again with a different search query.
            """,
            "type": "search"
        }

async def analyze_prompt_and_route_task(user_prompt: str):
    """Analyzes user prompt and routes to appropriate function"""
    try:
        gemini_model = initialize_gemini()
        if not gemini_model:
            return {"status": "error", "message": "Could not initialize Gemini model"}

        # First, determine if this is a task creation request
        classification_prompt = f"""
        Classify this request. Return EXACTLY in this format without any additional text:
        {{
            "type": "TASK_CREATION",
            "category": "email",
            "query_type": "email_task"
        }}
        
        Request: "{user_prompt}"
        """
        
        response = gemini_model.generate_content(classification_prompt)
        classification = json.loads(response.text.strip())
        
        if classification["type"] == "TASK_CREATION" and classification["category"] == "email":
            # Extract email details using a specific prompt
            email_prompt = f"""
            Extract email details from this request and return a JSON object.
            Request: "{user_prompt}"
            
            Return EXACTLY in this format without any additional text:
            {{
                "task_type": "email",
                "priority": "medium",
                "task_details": {{
                    "to": "recipient email",
                    "subject": "Meeting Discussion",
                    "content": "email body content",
                    "sender_name": "OpenCodeHq.Agent",
            "recipient_name": task_details.get('recipient_name', 'Banxs'),
            "sender_name": task_details.get('sender_name', 'Erico')
                }}
            }}
            """
            
            email_response = gemini_model.generate_content(email_prompt)
            email_info = json.loads(email_response.text.strip())
            
            # Create email content with proper formatting
            email_content = f"""
Dear {email_info['task_details']['recipient_name']},

{email_info['task_details']['content']}

Best regards,
{email_info['task_details']['sender_name']}
            """
            
            # Update the task details with formatted content
            email_info['task_details']['content'] = email_content.strip()
            
            # Create the task
            result = {
                "status": "success",
                "task": {
                    "task_id": str(uuid.uuid4()),
                    "type": "email",
                    "priority": email_info.get("priority", "medium"),
                    "email_content": {
                        "to": email_info['task_details']['to'],
                        "subject": email_info['task_details']['subject'],
                        "generated_content": email_content
                    }
                }
            }
            
            return {
                "status": "success",
                "original_prompt": user_prompt,  # Include original prompt
                "request_type": "TASK_CREATION",
                "interpreted_task": email_info,
                "task_result": result
            }
            
        elif classification["type"] in ["REAL_TIME_INFO", "WEB_SEARCH"]:
            return await web_search(user_prompt)
            
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        return {
            "status": "error",
            "message": "Failed to parse task details. Please try rephrasing your request."
        }
    except Exception as e:
        print(f"Error in analyze_prompt_and_route_task: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing prompt: {str(e)}"
        }



if __name__ == "__main__":
    main()
