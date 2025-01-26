import json
import logging
from typing import Dict, Any

# Import all necessary modules from your existing project
import google.generativeai as genai
from ai import send_email, handle_task_creation, create_task, get_gmail_service, scrape_url, summarize_content
from docsprocessing import RAGProcessor
from realtimeSearch import get_current_time, real_time_search
from weather import get_weather
from sendEmail import AIService
from todo import TodoManager
from webScrapeAndProcess import web_search, scrape_url, summarize_content

class TaskManager:
    def __init__(self, model):
        """
        Initialize TaskManager with a generative AI model
        
        Args:
            model: Generative AI model for task classification
        """
        self.model = model
        self.ai_service = AIService()
        self.todo_manager = TodoManager()
        self.rag_processor = RAGProcessor()
        
    def route_task(self, prompt_text: str) -> Dict[str, Any]:
        """
        Analyze and route task based on prompt text
        
        Args:
            prompt_text (str): User's voice-transcribed instruction
        
        Returns:
            Dict containing task execution result
        """
        try:
            # Comprehensive task classification prompt
            classification_prompt = f"""
            Classify the following request precisely into one of these categories:
            1. Email (send an email)
            2. Task Creation (create a todo/task)
            3. Web Search
            4. Document Processing
            5. Weather Information
            6. Time Lookup
            7. URL Scraping
            8. Content Summarization
            9. General Conversation

            Request: "{prompt_text}"

            Respond with a JSON structure:
            {{
                "type": "CATEGORY_NAME",
                "details": {{
                    "specific_parameters": "as needed"
                }}
            }}
            """
            
            # Generate classification using Gemini
            response = self.model.generate_content(classification_prompt)
            classification = json.loads(response.text.replace("```json", "").replace("```", "").strip())
            
            # Routing based on classification
            match classification['type'].upper():
                case "EMAIL":
                    return self.ai_service.send_email(
                        to=classification['details'].get('to', ''),
                        subject=classification['details'].get('subject', 'AI-Generated Email'),
                        body=classification['details'].get('body', prompt_text)
                    )
                
                case "TASK_CREATION":
                    return self.todo_manager.process_natural_language_request(
                        classification['details'].get('task', prompt_text)
                    )
                
                case "WEB_SEARCH":
                    return web_search(classification['details'].get('query', prompt_text))
                
                case "DOCUMENT_PROCESSING":
                    return self.rag_processor.process_documents(
                        classification['details'].get('documents', [])
                    )
                
                case "WEATHER_INFORMATION":
                    return get_weather(
                        classification['details'].get('location', prompt_text)
                    )
                
                case "TIME_LOOKUP":
                    return get_current_time(
                        classification['details'].get('location', 'US')
                    )
                
                case "URL_SCRAPING":
                    return scrape_url(
                        classification['details'].get('url', '')
                    )
                
                case "CONTENT_SUMMARIZATION":
                    return summarize_content(
                        classification['details'].get('content', prompt_text)
                    )
                
                case "GENERAL_CONVERSATION":
                    # Fallback to default conversational model
                    return {"type": "conversation", "response": self.model.generate_content(prompt_text).text}
                
                case _:
                    return {
                        "status": "error", 
                        "message": f"Unrecognized task type: {classification['type']}"
                    }
        
        except Exception as e:
            logging.error(f"Task routing error: {e}")
            return {
                "status": "error", 
                "message": f"Could not process request: {str(e)}"
            }

# Example usage in main AI script
def process_audio_prompt(model, prompt_text):
    """
    Process audio prompt using TaskManager
    
    Args:
        model: Generative AI model
        prompt_text (str): Transcribed audio prompt
    
    Returns:
        Processed task result
    """
    task_manager = TaskManager(model)
    return task_manager.route_task(prompt_text)

if __name__ == "__main__":
    process_audio_prompt()