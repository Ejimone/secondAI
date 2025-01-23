import asyncio
import logging
from ai import real_time_search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_real_time_search():
    """Test real-time information retrieval functionality"""
    test_queries = [
        "What's the weather in London?",
        "Current weather in New York",
        "Weather forecast for Tokyo"
    ]

    for query in test_queries:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing Real-Time Search: {query}")
        logger.info(f"{'='*50}")
        
        response = await real_time_search(query)
        
        if response.get("status") == "success":
            logger.info("\n🌤️ REAL-TIME WEATHER DETAILS")
            logger.info(f"Weather Data: {response.get('data', 'No data available')}")
        else:
            logger.error(f"Real-Time Weather retrieval failed: {response.get('message', 'No message available')}")
        
        logger.info(f"\n{'='*50}")

if __name__ == "__main__":
    asyncio.run(test_real_time_search())
