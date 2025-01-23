import asyncio
import logging
from ai import create_task

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_email_functionality():
    """Test email functionality"""
    test_email = {
        "task_type": "email",
        "task_details": {
            "to": "kejimone@gmail.com",
            "subject": "Quarterly Project Review Meeting",
            "content": "Need to schedule a comprehensive review of Q1 projects."
        },
        "priority": "high"
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Testing Email Task Creation")
    logger.info(f"{'='*50}")

    response = await create_task(
        task_type=test_email["task_type"],
        task_details=test_email["task_details"],
        priority=test_email["priority"]
    )

    if response["status"] == "success":
        logger.info("\n📧 EMAIL TASK DETAILS")
        logger.info(f"Task ID: {response['task']['task_id']}")
        logger.info(f"To: {response['task']['email_content']['to']}")
        logger.info(f"Subject: {response['task']['email_content']['subject']}")
        logger.info(f"Generated Email: {response['task']['email_content']['generated_content']}")
    else:
        logger.error(f"Email task creation failed: {response['message']}")

if __name__ == "__main__":
    asyncio.run(test_email_functionality())
