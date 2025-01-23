import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from webScrapeAndProcess import web_search

async def run_test():
    prompt = input("Enter your search prompt: ")
    result = await web_search(prompt)
    print(result)


if __name__ == "__main__":
    asyncio.run(run_test())

