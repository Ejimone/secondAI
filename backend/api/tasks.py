import json
import uuid
from ai import initialize_gemini
from webScrapeAndProcess import web_search


async def analyze_prompt_and_route_task(user_prompt: str):
    """Analyzes user prompt and routes to appropriate function"""
    try:
        gemini_model = initialize_gemini()
        if not gemini_model:
            return {"status": "error", "message": "Could not initialize Gemini model"}

        # First, determine if this is a task creation request
        classification_prompt = f"""
        Classify this request. Return EXACTLY in this format without any additional text:
        {{
            "type": "TASK_CREATION",
            "category": "email",
            "query_type": "email_task"
        }}
        
        Request: "{user_prompt}"
        """
        
        response = gemini_model.generate_content(classification_prompt)
        classification = json.loads(response.text.strip())
        
        if classification["type"] == "TASK_CREATION" and classification["category"] == "email":
            # Extract email details using a specific prompt
            email_prompt = f"""
            Extract email details from this request and return a JSON object.
            Request: "{user_prompt}"
            
            Return EXACTLY in this format without any additional text:
            {{
                "task_type": "email",
                "priority": "medium",
                "task_details": {{
                    "to": "recipient email",
                    "subject": "Meeting Discussion",
                    "content": "email body content",
                    "sender_name": "OpenCodeHq.Agent",
            "recipient_name": task_details.get('recipient_name', 'Banxs'),
            "sender_name": task_details.get('sender_name', 'Erico')
                }}
            }}
            """
            
            email_response = gemini_model.generate_content(email_prompt)
            email_info = json.loads(email_response.text.strip())
            
            # Create email content with proper formatting
            email_content = f"""
Dear {email_info['task_details']['recipient_name']},

{email_info['task_details']['content']}

Best regards,
{email_info['task_details']['sender_name']}
            """
            
            # Update the task details with formatted content
            email_info['task_details']['content'] = email_content.strip()
            
            # Create the task
            result = {
                "status": "success",
                "task": {
                    "task_id": str(uuid.uuid4()),
                    "type": "email",
                    "priority": email_info.get("priority", "medium"),
                    "email_content": {
                        "to": email_info['task_details']['to'],
                        "subject": email_info['task_details']['subject'],
                        "generated_content": email_content
                    }
                }
            }
            
            return {
                "status": "success",
                "original_prompt": user_prompt,  # Include original prompt
                "request_type": "TASK_CREATION",
                "interpreted_task": email_info,
                "task_result": result
            }
            
        elif classification["type"] in ["REAL_TIME_INFO", "WEB_SEARCH"]:
            return await web_search(user_prompt)
            
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        return {
            "status": "error",
            "message": "Failed to parse task details. Please try rephrasing your request."
        }
    except Exception as e:
        print(f"Error in analyze_prompt_and_route_task: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing prompt: {str(e)}"
        }

