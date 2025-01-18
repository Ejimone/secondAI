import os
import asyncio
import logging
from ai import initialize_openai, initialize_gemini, real_time_search, scrape_webpages_with_serpapi, handle_task_creation
import uuid
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for testing
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

async def test_initialize_openai():
    try:
        llm, is_working = initialize_openai()
        logger.info(f"OpenAI Initialization Successful: {is_working}")
        if is_working:
            logger.info(f"LLM Object: {llm}")
    except Exception as e:
        logger.error(f"Error in OpenAI initialization: {e}")

async def test_initialize_gemini():
    try:
        model = initialize_gemini()
        logger.info(f"Gemini Initialization Successful: {model is not None}")
        return model
    except Exception as e:
        logger.error(f"Error in Gemini initialization: {e}")
        return None

async def test_real_time_search(query):
    try:
        logger.info(f"Performing search for query: {query}")
        results = await real_time_search(query)
        
        # Add separators for better readability in logs
        logger.info("\n" + "="*50)
        logger.info("Search Results and Analysis:")
        logger.info("="*50)
        logger.info("\n%s", results)
        logger.info("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Error in real-time search: {e}")

async def test_scrape_webpages_with_serpapi(query):
    try:
        logger.info(f"\nInitiating comprehensive analysis for query: {query}")
        logger.info("="*50)
        
        results = await scrape_webpages_with_serpapi(query)
        
        # Format and display results with clear sections
        logger.info("\n" + "="*50)
        logger.info("COMPREHENSIVE ANALYSIS RESULTS")
        logger.info("="*50)
        logger.info("\n%s", results)
        logger.info("\n" + "="*50)
        
        return results
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        return None

async def test_task_creation():
    """
    Test function for task creation functionality
    """
    try:
        test_tasks = [
            {
                "task_type": "email",
                "task_details": {
                    "to": "john@example.com",
                    "subject": "Quarterly Project Review Meeting",
                    "content": "Need to schedule a comprehensive review of Q1 projects, discuss milestones, and plan for Q2."
                },
                "due_date": "2024-03-20T14:00:00",
                "priority": "high"
            },
            {
                "task_type": "reminder",
                "task_details": {
                    "content": "Prepare presentation slides for client meeting"
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
            logger.info(f"\nTesting task creation for: {task['task_type']}")
            logger.info("="*50)
            
            response = await handle_task_creation(task)
            
            if response["status"] == "success":
                task_data = response["task"]
                
                logger.info(f"Task ID: {task_data['task_id']}")
                logger.info(f"Type: {task_data['type']}")
                logger.info(f"Priority: {task_data['priority']}")
                logger.info(f"Due Date: {task_data['due_date']}")
                
                if "email_content" in task_data:
                    logger.info("\nEmail Content:")
                    logger.info(f"To: {task_data['email_content']['to']}")
                    logger.info(f"Subject: {task_data['email_content']['subject']}")
                    logger.info("\nGenerated Content:")
                    logger.info(task_data['email_content']['generated_content'])
                
                elif "reminder_details" in task_data:
                    logger.info("\nReminder Details:")
                    logger.info(task_data['reminder_details']['generated_plan'])
                
                elif "todo_details" in task_data:
                    logger.info("\nTodo Details:")
                    logger.info(task_data['todo_details']['generated_plan'])
            else:
                logger.error(f"Task creation failed: {response['message']}")
            
            logger.info("="*50)

    except Exception as e:
        logger.error(f"Error in test_task_creation: {str(e)}")

async def main():
    try:
        # Initialize APIs
        await test_initialize_openai()
        await test_initialize_gemini()
        
        # Test task creation
        logger.info("\nTesting Task Creation Functionality")
        await test_task_creation()
        
        # Test search functionality if needed
        test_queries = [
            "Latest developments in quantum computing",
            "Impact of artificial intelligence on healthcare"
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting search for: {query}")
            await test_scrape_webpages_with_serpapi(query)
            await asyncio.sleep(2)  # Add delay between queries

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())