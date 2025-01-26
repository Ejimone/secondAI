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
from sendEmail import AIService

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

            # Determine request type
            classification_prompt = """
            Classify this request. Return EXACTLY in this format:
            {
                "type": "TASK_TYPE",
                "category": "CATEGORY",
                "action": "ACTION"
            }
            Where:
            - TASK_TYPE: EMAIL | TODO | WEATHER | SEARCH | REALTIME
            - CATEGORY: task|reminder|email|weather|search
            - ACTION: create|query|search|get
            """
            
            response = gemini_model.generate_content(
                classification_prompt + f'\nRequest: "{user_prompt}"'
            )
            
            classification = json.loads(response.text.strip())
            
            # Route based on classification
            match classification["type"]:
                case "EMAIL":
                    return await self.ai_service.send_email(
                        to=classification["to"],
                        subject=classification["subject"],
                        body=classification["body"]
                    )
                    
                case "TODO":
                    return await self.todo_manager.process_natural_language_request(
                        user_prompt
                    )
                    
                case "WEATHER":
                    location = user_prompt.split("weather in")[-1].strip()
                    return await get_weather(location)
                    
                case "REALTIME":
                    return await real_time_search(user_prompt)
                    
                case "SEARCH":
                    return await self.rag_processor.ask_question(user_prompt)
                    
                case _:
                    return {
                        "status": "error",
                        "message": "Unknown request type"
                    }

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
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
	# Example usage
	response = await route_task("Send an email to kejimone30@gmail.com telling him to check out the new OpenCode model")
	print(response)


if __name__ == "__main__":
    asyncio.run(main())