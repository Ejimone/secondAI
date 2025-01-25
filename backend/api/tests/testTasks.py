# import route task from ../tasks.py
from tasks import route_task

import asyncio

async def main():
	# Example usage
	response = await route_task("Send an email to kejimone30@gmail.com telling him to check out the new OpenCode model")
	print(response)

asyncio.run(main())