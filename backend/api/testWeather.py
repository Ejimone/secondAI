import asyncio
import logging
import sys
import os
from typing import Dict, Any
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weather import get_weather 
from webScrapeAndProcess import scrape_webpages_with_serpapi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

async def test_weather_functionality():
    """Test weather functionality"""
    test_locations = [
        "London",
        "New York", 
        "Tokyo"
    ]

    for location in test_locations:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing Weather Query for: {location}")
        logger.info(f"{'='*50}")
        
        try:
            response = await get_weather(location)
            
            if response["status"] == "success":
                logger.info("\n🌤️ WEATHER DETAILS")
                logger.info(f"Weather Data: {response['data']}")
            else:
                logger.error(f"Weather retrieval failed: {response.get('message')}")
                
        except Exception as e:
            logger.error(f"Error getting weather for {location}: {str(e)}")
            
        await asyncio.sleep(1)  # Rate limiting
        logger.info(f"\n{'='*50}")

async def test_scrape_webpages_with_serpapi():
    """Test scraping functionality with SerpAPI"""
    test_queries = [
        "current weather in London",
        "weather forecast for New York",
        "Tokyo weather today"
    ]

    for query in test_queries:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing Scraping Query for: {query}")
        logger.info(f"{'='*50}")

        try:
            response = await scrape_webpages_with_serpapi(query)
            
            if isinstance(response, dict) and response.get("status") == "error":
                logger.error(f"Scraping failed: {response['message']}")
            else:
                logger.info("\n🌐 SCRAPING DETAILS")
                logger.info(f"Scraped Data: {response['data']}")
                
                # Check if the scraped data is not empty
                if response['data']:
                    logger.info("Successfully scraped weather information.")
                else:
                    logger.warning("No weather information was scraped.")
                
        except Exception as e:
            logger.error(f"Error scraping data for query '{query}': {str(e)}")
            
        await asyncio.sleep(2)  # Rate limiting
        logger.info(f"\n{'='*50}")

def main():
    """Run all tests"""
    try:
        asyncio.run(test_weather_functionality())
        # asyncio.run(test_scrape_webpages_with_serpapi())
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")

if __name__ == "__main__":
    main()
