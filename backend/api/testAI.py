import os
import asyncio
from ai import initialize_openai, initialize_gemini, real_time_search, scrape_webpages_with_serpapi

# Set environment variables for testing
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

async def test_initialize_openai():
    llm, is_working = initialize_openai()
    print("OpenAI Initialization Successful:", is_working)
    if is_working:
        print("LLM Object:", llm)

async def test_initialize_gemini():
    model = initialize_gemini()
    print("Gemini Initialization Successful:", model is not None)

async def test_real_time_search(query):
    results = await real_time_search(query)
    print("Real-Time Search Results:\n", results)

async def test_scrape_webpages_with_serpapi(query):
    scraped_content = await scrape_webpages_with_serpapi(query)
    print("Scraped Content:\n", scraped_content)

async def main():
    await test_initialize_openai()
    await test_initialize_gemini()
    await test_real_time_search("How long does it take to get to the moon?")  # Example query
    await test_scrape_webpages_with_serpapi("Apollo mission to the moon")

if __name__ == "__main__":
    asyncio.run(main())