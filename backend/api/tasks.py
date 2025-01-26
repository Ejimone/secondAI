import json
import logging
import uuid
from typing import Dict, Any
import asyncio
from ai import initialize_gemini
from docsprocessing import RAGProcessor
from realtimeSearch import real_time_search
from weather import get_weather
from todo import TodoManager 
from sendEmail import AIService, test_service  # Changed import
from webScrapeAndProcess import web_search
from Audio import speak

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskRouter:
    def __init__(self):
        self.ai_service = AIService()
        self.todo_manager = TodoManager()
        self.rag_processor = RAGProcessor()
        
    async def analyze_prompt_and_route_task(self, user_prompt: str) -> Dict[str, Any]:
        """Analyzes user prompt and routes to appropriate function"""
        try:
            gemini_model = initialize_gemini()
            if not gemini_model:
                return {"status": "error", "message": "Could not initialize Gemini model"}

            classification_prompt = """
            Analyze this request and classify it. Return JSON structure:
            For real-time info (weather, time, current events):
            {
                "type": "REALTIME",
                "details": {
                    "query": "processed query",
                    "category": "weather|time|news"
                }
            }
            For web search/research:
            {
                "type": "WEBSEARCH",
                "details": {
                    "query": "search query"
                }
            }
            For emails:
            {
                "type": "EMAIL",
                "details": {
                    "to": "email@address.com",
                    "subject": "Generated subject",
                    "body": "Generated email body"
                }
            }
            For todos:
            {
                "type": "TODO",
                "details": {
                    "query": "todo task details"
                }
            }
            For general conversation:
            {
                "type": "CONVERSATION",
                "details": {
                    "query": "conversation query"
                }
            }
            """
            
            logger.info(f"Processing prompt: {user_prompt}")
            
            response = gemini_model.generate_content(
                classification_prompt + f'\nRequest: "{user_prompt}"'
            )
            
            # Clean response text
            response_text = response.text.replace("```json", "").replace("```", "").strip()
            classification = json.loads(response_text)
            
            logger.info(f"Classification: {classification}")
            
            # Route based on classification
            match classification["type"].upper():
                case "REALTIME":
                    return await real_time_search(user_prompt)
                
                    
                case "WEBSEARCH":
                    return await web_search(classification["details"]["query"])
                    
                case "EMAIL":
                    # Call test_service with the AIService instance
                    return await test_service(self.ai_service)
                    
                case "TODO":
                    return await self.todo_manager.process_natural_language_request(
                        classification["details"]["query"]
                    )
                
                case "CONVERSATION":
                    return {"status": "success", "response": gemini_model.generate_content(user_prompt).text}
                    
                case _:
                    logger.error(f"Unknown request type: {classification['type']}")
                    return {
                        "status": "error",
                        "message": f"Unknown request type: {classification['type']}"
                    }

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return {"status": "error", "message": "Failed to parse response"}
        except KeyError as e:
            logger.error(f"Missing required field: {str(e)}")
            return {"status": "error", "message": f"Missing required field: {str(e)}"}
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            return {
                "status": "error", 
                "message": f"Error processing request: {str(e)}"
            }

# Initialize router
task_router = TaskRouter()

# Main entry point
async def route_task(user_prompt: str) -> Dict[str, Any]:
    """Main entry point for task routing"""
    return await task_router.analyze_prompt_and_route_task(user_prompt)

async def main():
    while True:
        try:
            user_prompt = input("What would you like to do? (or 'exit' to quit): ")
            if user_prompt.lower() == 'exit':
                break
                
            response = await route_task(user_prompt)
            print("\nResponse:", json.dumps(response, indent=2))
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())