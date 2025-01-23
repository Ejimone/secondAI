# this is for testing the realtime search function, the user will be able to search for anything, and the function will return the result in a structured format
import requests
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realtimeSearch import real_time_search

async def test_realtime_search():
    user_prompt = "what is the weather in new york"
    response = real_time_search(user_prompt)
    print(response)
    assert response["status"] == "success"


if __name__ == "__main__":
    test_realtime_search()