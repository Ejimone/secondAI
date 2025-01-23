import os
import asyncio
import logging
from ai import (
    initialize_openai, 
    initialize_gemini, 
    real_time_search, 
    scrape_webpages_with_serpapi, 
    handle_task_creation, 
    analyze_prompt_and_route_task,
    create_task
)
import uuid
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_initialize_openai():
    """Test OpenAI initialization"""
    try:
        llm, is_working = initialize_openai()
        logger.info(f"OpenAI Initialization Successful: {is_working}")
        if is_working:
            logger.info(f"LLM Object: {llm}")
        return is_working
    except Exception as e:
        logger.error(f"Error in OpenAI initialization: {e}")
        return False

async def test_initialize_gemini():
    """Test Gemini initialization"""
    try:
        model = initialize_gemini()
        logger.info(f"Gemini Initialization Successful: {model is not None}")
        return model is not None
    except Exception as e:
        logger.error(f"Error in Gemini initialization: {e}")
        return False

async def test_task_creation():
    """Test task creation functionality"""
    test_tasks = [
        {
            "task_type": "email",
            "task_details": {
                "to": "kejimone@gmail.com",
                "subject": "Quarterly Project Review Meeting",
                "content": "Need to schedule a comprehensive review of Q1 projects."
            },
            "due_date": "2024-03-20T14:00:00",
            "priority": "high"
        },
        {
            "task_type": "reminder",
            "task_details": {
                "content": "Prepare presentation slides"
            },
            "due_date": "2024-03-21T10:00:00",
            "priority": "medium"
        },
        {
            "task_type": "todo",
            "task_details": {
                "content": "Website redesign project"
            },
            "due_date": "2024-04-01T17:00:00",
            "priority": "high"
        }
    ]

    for task in test_tasks:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing Task Creation: {task['task_type'].upper()}")
        logger.info(f"{'='*50}")
        
        response = await handle_task_creation(task)
        
        if response["status"] == "success":
            task_data = response["task"]
            
            # Print common task information
            logger.info("\n📋 TASK DETAILS")
            logger.info(f"Task ID: {task_data['task_id']}")
            logger.info(f"Type: {task_data['type']}")
            logger.info(f"Priority: {task_data['priority'].upper()}")
            logger.info(f"Due Date: {task_data['due_date']}")
            logger.info(f"Status: {task_data['status'].upper()}")
            
            # Print task-specific information
            if task_data['type'] == 'email':
                logger.info("\n📧 EMAIL CONTENT")
                logger.info(f"To: {task_data['email_content']['to']}")
                logger.info(f"Subject: {task_data['email_content']['subject']}")
                logger.info("\nGenerated Email:")
                logger.info(f"{task_data['email_content']['generated_content']}")
                
            elif task_data['type'] == 'reminder':
                logger.info("\n⏰ REMINDER DETAILS")
                logger.info(f"Content: {task_data['reminder_details']['content']}")
                logger.info("\nGenerated Plan:")
                logger.info(f"{task_data['reminder_details']['generated_plan']}")
                
            elif task_data['type'] == 'todo':
                logger.info("\n📝 TODO DETAILS")
                logger.info(f"Task: {task_data['todo_details']['content']}")
                logger.info("\nGenerated Plan:")
                
                # Format the generated plan for better readability
                plan = task_data['todo_details']['generated_plan']
                for line in plan.split('\n'):
                    if line.startswith('##'):  # Main sections
                        logger.info(f"\n{line.strip('#').strip()}")
                    elif line.startswith('**'):  # Sub-sections
                        logger.info(f"\n{line.strip('*')}")
                    elif line.startswith('-'):  # List items
                        logger.info(f"  {line}")
                    else:
                        logger.info(line)
            
            logger.info(f"\n{'='*50}\n")
        else:
            logger.error(f"Task creation failed: {response['message']}")
            logger.info(f"{'='*50}\n")

async def test_prompt_analysis():
    """Test prompt analysis and routing"""
    test_prompts = [
        "Send an email to kejimone@gmail.com about the project meeting tomorrow",
        "Remind me to prepare presentation slides by next Tuesday",
        "Create a todo list for the website redesign project",
    ]

    for prompt in test_prompts:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing Prompt: {prompt}")
        logger.info(f"{'='*50}")
        
        response = await analyze_prompt_and_route_task(prompt)
        
        if response["status"] == "success":
            logger.info("\n🎯 PROMPT ANALYSIS")
            logger.info(f"Original Prompt: {response['original_prompt']}")
            
            interpreted = response['interpreted_task']
            logger.info("\n📋 INTERPRETED TASK")
            logger.info(f"Task Type: {interpreted['task_type'].upper()}")
            logger.info(f"Priority: {interpreted['priority'].upper()}")
            if 'due_date' in interpreted:
                logger.info(f"Due Date: {interpreted['due_date']}")
            
            logger.info("\n📝 TASK DETAILS")
            for key, value in interpreted['task_details'].items():
                logger.info(f"{key.capitalize()}: {value}")
                
        else:
            logger.error(f"Prompt analysis failed: {response['message']}")
        
        logger.info(f"\n{'='*50}")

async def test_search_functionality():
    """Test search functionality"""
    test_queries = [
        "Latest developments in AI",
        "Best practices in software testing"
    ]

    for query in test_queries:
        logger.info(f"\nTesting search for: {query}")
        results = await real_time_search(query)
        logger.info(f"Search Results: {results}")
        logger.info("="*50)
        await asyncio.sleep(2)  # Rate limiting

async def test_weather_functionality():
    """Test weather functionality"""
    test_queries = [
        "What's the weather in London?",
        "Current weather in New York",
        "Weather forecast for Tokyo"
    ]

    for query in test_queries:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing Weather Query: {query}")
        logger.info(f"{'='*50}")
        
        response = await real_time_search(query)
        
        if response["status"] == "success":
            logger.info("\n🌤️ WEATHER DETAILS")
            logger.info(f"Weather Data: {response['data']}")
        else:
            logger.error(f"Weather retrieval failed: {response['message']}")
        
        logger.info(f"\n{'='*50}")

async def main():
    """Main test execution function"""
    try:
        # Test API initializations
        logger.info("\n=== Testing API Initializations ===")
        await test_initialize_openai()
        await test_initialize_gemini()

        # Test core functionalities
        logger.info("\n=== Testing Task Creation ===")
        await test_task_creation()

        logger.info("\n=== Testing Prompt Analysis ===")
        await test_prompt_analysis()

        logger.info("\n=== Testing Weather Functionality ===")
        await test_weather_functionality()

        logger.info("\n=== Testing Search Functionality ===")
        await test_search_functionality()

    except Exception as e:
        logger.error(f"Error in test execution: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())