import os
import json
import base64
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import asyncio
from dataclasses import dataclass, asdict
from googleapiclient.discovery import build
from sendEmail import AIService
from weather import get_weather
from realtimeSearch import real_time_search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@dataclass
class TodoItem:
    id: str
    title: str
    description: str
    due_date: Optional[datetime]
    priority: str
    status: str
    category: str
    location: Optional[str]
    weather_check: bool
    notifications: List[str]
    created_at: datetime
    last_modified: datetime
    calendar_event_id: Optional[str] = None

    def to_dict(self):
        """Convert TodoItem to a dictionary with serializable values"""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

class TodoManager:
    def __init__(self):
        self.storage_path = Path("data/todos.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.todos: Dict[str, TodoItem] = self._load_todos()
        
        # Initialize services
        self.ai_service = AIService()
        self._initialize_calendar_service()

    def _initialize_calendar_service(self):
        """Initialize Google Calendar service separately from Gmail"""
        try:
            # Reuse credentials from AIService
            creds = self.ai_service._get_gmail_credentials()
            self.calendar_service = build('calendar', 'v3', credentials=creds)
            logger.info("Calendar service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Calendar service: {e}")
            self.calendar_service = None

    async def create_todo(self, title: str, description: str, due_date: Optional[str] = None,
                         priority: str = "medium", category: str = "task",
                         location: Optional[str] = None, weather_check: bool = False,
                         notifications: List[str] = None) -> Dict[str, Any]:
        try:
            todo_id = str(uuid.uuid4())
            
            # Parse due_date if provided
            due_date_obj = None
            if due_date:
                try:
                    due_date_obj = datetime.fromisoformat(due_date)
                except ValueError as e:
                    logger.error(f"Invalid date format: {e}")
                    return {"status": "error", "message": f"Invalid date format: {due_date}"}
            
            # Get weather if location provided
            weather_info = None
            if location and weather_check:
                try:
                    weather_info = await get_weather(location)
                    if weather_info.get("status") == "error":
                        logger.warning(f"Weather info not available for location: {location}")
                        weather_info = {"status": "warning", "message": "Weather information not available"}
                except Exception as e:
                    logger.error(f"Error fetching weather: {e}")
                    weather_info = {"status": "error", "message": "Failed to fetch weather information"}

            todo = TodoItem(
                id=todo_id,
                title=title,
                description=description,
                due_date=due_date_obj,
                priority=priority,
                status="pending",
                category=category,
                location=location,
                weather_check=weather_check,
                notifications=notifications or ["email"],
                created_at=datetime.now(),
                last_modified=datetime.now()
            )

            # Create calendar event if due date exists and calendar service is available
            if due_date_obj and self.calendar_service:
                try:
                    event = {
                        'summary': title,
                        'description': description,
                        'start': {'dateTime': due_date_obj.isoformat()},
                        'end': {'dateTime': (due_date_obj + timedelta(hours=1)).isoformat()},
                    }
                    if location:
                        event['location'] = location
                    
                    calendar_event = self.calendar_service.events().insert(
                        calendarId='primary',
                        body=event
                    ).execute()
                    todo.calendar_event_id = calendar_event['id']
                    logger.info(f"Calendar event created successfully: {calendar_event['id']}")
                except Exception as e:
                    logger.error(f"Failed to create calendar event: {e}")
                    # Continue with todo creation even if calendar fails

            # Save todo
            self.todos[todo_id] = todo
            self._save_todos()

            # Schedule email notification
            if "email" in (notifications or ["email"]):
                try:
                    await self._schedule_email_notification(todo)
                except Exception as e:
                    logger.error(f"Failed to schedule email notification: {e}")

            # Return serializable response
            return {
                "status": "success",
                "message": "Todo created successfully",
                "todo": todo.to_dict(),
                "weather_info": weather_info
            }
        except Exception as e:
            logger.error(f"Error creating todo: {e}")
            return {"status": "error", "message": str(e)}

    async def _schedule_email_notification(self, todo: TodoItem):
        """Schedule email notification for todo"""
        if todo.due_date:
            reminder_time = todo.due_date - timedelta(hours=1)
            if reminder_time > datetime.now():
                weather_info = ""
                if todo.weather_check and todo.location:
                    try:
                        weather = await get_weather(todo.location)
                        if weather["status"] == "success":
                            weather_info = f"\nWeather at location: {weather['data']}"
                    except Exception as e:
                        logger.error(f"Failed to get weather for notification: {e}")
                        weather_info = "\nWeather information unavailable"
                
                await self.ai_service.send_email(
                    to={"kejimone30@gmail.com"},  # Replace with user's email
                    subject=f"Reminder: {todo.title}",
                    body=f"""
                    Reminder for your upcoming task:
                    Title: {todo.title}
                    Due: {todo.due_date.strftime('%Y-%m-%d %H:%M')}
                    Location: {todo.location if todo.location else 'Not specified'}
                    {weather_info}
                    
                    Description:
                    {todo.description}
                    """
                )

    async def process_natural_language_request(self, request: str) -> Dict[str, Any]:
        """Process natural language todo requests"""
        try:
            # Get tomorrow's date for the AI prompt
            tomorrow = datetime.now() + timedelta(days=1)
            tomorrow_str = tomorrow.strftime('%Y-%m-%d')
            
            # Format prompt for better response structure
            prompt = f"""
            Convert this todo request into structured data:
            "{request}"
            
            Today's date is {datetime.now().strftime('%Y-%m-%d')}.
            If the request mentions "tomorrow", use {tomorrow_str}.
            For location, return the full city name.
            
            Return ONLY valid JSON with actual dates (not placeholders) in this format:
            {{
                "title": "extracted title",
                "description": "detailed description",
                "due_date": "YYYY-MM-DDTHH:MM:SS",
                "priority": "low/medium/high",
                "category": "task/reminder/meeting",
                "location": "full city name",
                "weather_check": true/false
            }}"""
            
            # Generate response using Gemini
            response = self.ai_service.gemini_model.generate_content(prompt)
            
            if not response or not response.text:
                logger.error("Empty response from AI model")
                return {"status": "error", "message": "Failed to get AI response"}
            
            # Extract and clean JSON from response
            response_text = response.text.strip()
            logger.debug(f"AI Response: {response_text}")
            
            # Attempt to extract JSON
            try:
                start_index = response_text.find("{")
                end_index = response_text.rfind("}") + 1
                if start_index != -1 and end_index != -1:
                    json_str = response_text[start_index:end_index]
                else:
                    json_str = response_text

                task_info = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response: {e}")
                logger.debug(f"Raw response: {response_text}")
                return {
                    "status": "error",
                    "message": f"Failed to parse AI response: {e}. Raw response: {response_text}",
                }

            # Create todo with parsed information
            return await self.create_todo(
                title=task_info["title"],
                description=task_info["description"],
                due_date=task_info["due_date"],
                priority=task_info.get("priority", "medium"),
                category=task_info.get("category", "task"),
                location=task_info.get("location"),
                weather_check=task_info.get("weather_check", False),
            )
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"status": "error", "message": str(e)}

    def _save_todos(self) -> None:
        """Save todos to storage"""
        with open(self.storage_path, 'w') as f:
            json.dump({
                id_: todo.to_dict()
                for id_, todo in self.todos.items()
            }, f, indent=2)

    def _load_todos(self) -> Dict[str, TodoItem]:
        """Load todos from storage"""
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                data = json.load(f)
                return {
                    id_: TodoItem(**{
                        **item,
                        "due_date": datetime.fromisoformat(item["due_date"]) if item["due_date"] else None,
                        "created_at": datetime.fromisoformat(item["created_at"]),
                        "last_modified": datetime.fromisoformat(item["last_modified"])
                    }) for id_, item in data.items()
                }
        return {}

todo_manager = TodoManager()

async def main():
    # Example usage
    result = await todo_manager.process_natural_language_request(
        "Remind me to buy groceries tomorrow at 5pm at Walmart in New York City, check weather before going"
    )
    # Use custom encoder for JSON serialization
    print(json.dumps(result, indent=2, cls=DateTimeEncoder))

if __name__ == "__main__":
    asyncio.run(main())