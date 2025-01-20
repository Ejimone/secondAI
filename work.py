import os
import asyncio
import logging
import datetime
import atexit
from contextlib import contextmanager
from google.cloud import storage
from ai import (
    send_email,
    initialize_openai, 
    initialize_gemini, 
    real_time_search, 
    scrape_webpages_with_serpapi, 
    handle_task_creation, 
    analyze_prompt_and_route_task,
    create_task
)
import uuid

# Add this at the top after imports
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Configure logging before anything else
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@contextmanager
def gemini_session():
    """Context manager for Gemini session"""
    try:
        model = initialize_gemini()
        yield model
    finally:
        if 'genai' in globals():
            genai.close()  # Clean up Gemini resources

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
        with gemini_session() as model:
            is_working = model is not None
            logger.info(f"Gemini Initialization Successful: {is_working}")
            return is_working
    except Exception as e:
        logger.error(f"Error in Gemini initialization: {e}")
        return False

async def test_task_creation():
    """Test task creation functionality"""
    try:
        test_tasks = [
            {
                "task_type": "email",
                "task_details": {
                    "to": "kejimone30@gmail.com",
                    "subject": "Quarterly Project Review Meeting",
                    "content": "Need to schedule a comprehensive review of Q1 projects.",
                    "sender_name": "Test Sender",
                    "recipient_name": "Test Recipient"
                },
                "due_date": datetime.datetime.now().isoformat(),
                "priority": "high"
            },
            {
                "task_type": "reminder",
                "task_details": {
                    "content": "Prepare presentation slides",
                    "reminder_time": datetime.datetime.now().isoformat()
                },
                "due_date": datetime.datetime.now().isoformat(),
                "priority": "medium"
            },
            {
                "task_type": "todo",
                "task_details": {
                    "content": "Website redesign project",
                    "steps": ["Planning", "Design", "Development", "Testing"]
                },
                "due_date": datetime.datetime.now().isoformat(),
                "priority": "high"
            }
        ]

        for task in test_tasks:
            logger.info("\n==================================================")
            logger.info(f"Testing Task Creation: {task['task_type'].upper()}")
            logger.info("==================================================")
            
            result = await create_task(
                task_type=task['task_type'],
                task_details=task['task_details'],
                priority=task['priority']
            )
            
            if result['status'] == 'success':
                logger.info("\n📋 TASK DETAILS")
                logger.info(f"Task ID: {result['task']['task_id']}")
                logger.info(f"Type: {result['task']['type']}")
                logger.info(f"Priority: {result['task']['priority']}")
                
                if 'email_content' in result['task']:
                    logger.info("\n📧 EMAIL DETAILS")
                    logger.info(f"To: {result['task']['email_content']['to']}")
                    logger.info(f"Subject: {result['task']['email_content']['subject']}")
                    logger.info(f"Content: {result['task']['email_content']['generated_content']}")
            else:
                logger.error(f"Task creation failed: {result['message']}")
                
    except Exception as e:
        logger.error(f"Error in task creation test: {str(e)}")
        raise

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

async def test_email_functionality():
    """Test email sending functionality"""
    logger.info("\n=== Testing Email Functionality ===")
    
    test_emails = [
        {
            "prompt": "Send an email to kejimone30@gmail.com about the project meeting tomorrow at 2pm",
            "expected_type": "email"
        },
        {
            "prompt": "Draft an email to kejimone@gmail.com regarding the quarterly review",
            "expected_type": "email"
        }
    ]

    for test in test_emails:
        logger.info(f"\nTesting email prompt: {test['prompt']}")
        try:
            # Test prompt analysis first
            response = await analyze_prompt_and_route_task(test["prompt"])
            
            if response["status"] == "success":
                logger.info("✅ Prompt analysis successful")
                logger.info("\n📧 Generated Email Details:")
                
                if response.get("request_type") == "TASK_CREATION":
                    task_details = response.get("task_result", {}).get("task", {})
                    email_content = task_details.get("email_content", {})
                    
                    logger.info(f"To: {email_content.get('to')}")
                    logger.info(f"Subject: {email_content.get('subject')}")
                    logger.info("\nGenerated Content:")
                    logger.info(f"{email_content.get('generated_content')}")
                    
                    send_result = await send_email(
                        to=email_content["to"],
                        subject=email_content["subject"],
                        body=email_content["generated_content"]
                    )
                    logger.info(f"\nEmail Send Result: {send_result}")
                else:
                    logger.error("❌ Response type mismatch - expected email task")
            else:
                logger.error(f"❌ Prompt analysis failed: {response.get('message')}")
                
        except Exception as e:
            logger.error(f"❌ Error testing email functionality: {str(e)}")
        
        logger.info("\n" + "="*50)

# Modify main() to use context manager
async def main():
    """Main test execution function"""
    try:
        # Test API initializations
        logger.info("\n=== Testing API Initializations ===")
        await test_initialize_openai()
        
        with gemini_session():
            await test_initialize_gemini()
            await test_email_functionality()
            await test_task_creation()
            await test_prompt_analysis()
            await test_search_functionality()

    except Exception as e:
        logger.error(f"Error in test execution: {str(e)}")
    finally:
        # Cleanup
        if 'genai' in globals():
            genai.close()

# Add cleanup handler
def cleanup():
    if 'genai' in globals():
        genai.close()

atexit.register(cleanup)

if __name__ == "__main__":
    asyncio.run(main())