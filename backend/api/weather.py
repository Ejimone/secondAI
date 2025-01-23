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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_weather(location: str) -> Dict[str, Any]:
    """Get current weather for a location"""
    try:
        # Using OpenWeatherMap API
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            logger.error("Weather API key not configured.")
            return {
                "status": "error",
                "data": "⚠️ Weather information is temporarily unavailable (API key not configured)",
                "type": "weather"
            }
        
        # Make the API request
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)  # Added timeout for better error handling
        
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
        else:
            logger.error(f"Weather API error: {response.status_code} - {response.text}")
            return {
                "status": "error",
                "data": f"⚠️ Weather service returned status code: {response.status_code}",
                "type": "weather"
            }
            
    except requests.Timeout:
        logger.error("Weather service timeout.")
        return {
            "status": "error",
            "data": "⚠️ Weather service is taking too long to respond. Please try again.",
            "type": "weather"
        }
    except requests.RequestException as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        return {
            "status": "error",
            "data": f"⚠️ Error fetching weather data: {str(e)}",
            "type": "weather"
        }
    except Exception as e:
        logger.exception("Unexpected error in get_weather.")
        return {
            "status": "error",
            "data": f"⚠️ Unexpected error: {str(e)}",
            "type": "weather"
        }





