import google.generativeai as genai
import speech_recognition as sr
import os
import asyncio
import logging
from faster_whisper import WhisperModel
import pyaudio
import time
from openai import OpenAI
from dotenv import load_dotenv
import google.cloud.speech_v1p1beta1 as speech
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
from google.oauth2 import service_account
from ai import send_email, handle_task_creation, create_task, get_gmail_service, scrape_url, summarize_content
from docsprocessing import RAGProcessor # Import the RAGProcessor class from docsprocessing.py, RAGPprocessor it has functions to process_documentsm \_process_urls, \_process_pdf, ask_question, stream_answer, \_stram_web_search and the main function which was used to test the class, which could be modified to be used to be used
from realtimeSearch import get_current_time, real_time_search
from weather import get_weather
from sendEmail import * # Import the AIService class from sendEmail.py, AIService has functions to send_email, test_service, get_gmail_service, and the main function which was used to test the class, which could be modified to be used
from tasks import TaskRouter
from todo import TodoManager
from webScrapeAndProcess import web_search, scrape_url, summarize_content, scrape_webpages_with_serpapi



load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
credentials_path = "./credentials.json"
if not os.path.exists(credentials_path):
    raise FileNotFoundError(f"Credentials file not found at {credentials_path}")

credentials = service_account.Credentials.from_service_account_file(credentials_path)
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

speech_client = speech.SpeechClient(credentials=credentials)
tts_client = texttospeech.TextToSpeechClient(credentials=credentials)

wakeword = "boom"
listening_for_wakeword = True

client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

whisper_size = "base"  # or "small", "medium", "large-v1", "large-v2"
num_cores = os.cpu_count()
whisper_model = WhisperModel(whisper_size, device="cpu", compute_type="int8", cpu_threads=num_cores, num_workers=num_cores)




num_cores = os.cpu_count()
whisper_model = WhisperModel(
    whisper_size,
    device="cpu",
    compute_type="int8",
    cpu_threads=num_cores,
    num_workers=num_cores
)
generation_config = {
    "temperature":0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
]


model = genai.GenerativeModel(
    'gemini-1.5-flash',
    generation_config=generation_config,
    safety_settings=safety_settings
)
r = sr.Recognizer()
source = sr.Microphone()
convo= model.start_chat()
system_message =  '''INSTRUCTIONS: Do not respond to messages in a way that would reveal personal information, or a too long  format response, you can also be affirmative or negative, this if for token generation purposes.
SYSTEM MESSAGE: You're a being used for Voice Assistant and AI agent ans should respond as so., As an agent, you should be able to respond to any question or statement that is asked of you or tasked to you. You generate words in a user-friendly manner. You can also ask questions to the user to get more information, be playful alsyou generate workds of valur prioritising logic and facts'''
system_message= system_message.replace("\n", "")

def speak(text):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", 
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )
    
    response = tts_client.synthesize_speech(
        input=synthesis_input, 
        voice=voice, 
        audio_config=audio_config
    )
    
    # Play audio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, 
                    channels=1, 
                    rate=24000, 
                    output=True)
    stream.write(response.audio_content)
    stream.stop_stream()
    stream.close()
    p.terminate()


def wav_to_text(audio_path):
    segments,_ = whisper_model.transcribe(audio_path)
    text = "".join(segment.text for segment in segments)
    text = text.replace("*", "")  # Remove asterisks from the transcribed text
    return text

def listen_for_wake_word(audio):
    audio_content = speech.RecognitionAudio(content=audio.get_wav_data())
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US"
    )
    
    response = speech_client.recognize(config=config, audio=audio_content)
    
    for result in response.results:
        transcript = result.alternatives[0].transcript
        print(f"Detected transcript: {transcript}")  # Debug print
        if wakeword.lower() in transcript.lower():
            return True
    return False

async def listen_and_route_tasks(audio_input: str) -> None:
    """Route voice commands to appropriate functions"""
    try:
        task_router = TaskRouter()
        
        # Analyze the prompt
        classification = await task_router.analyze_prompt_and_route_task(audio_input)
        
        if classification["type"].upper() == "EMAIL":
            # Handle email special case
            speak("Please type the sender's name:")
            sender_name = input("Sender's name: ")
            
            speak("Please type the sender's email:")
            sender_email = input("Sender's email: ")
            
            speak("Please type the receiver's name:")
            receiver_name = input("Receiver's name: ")
            
            speak("Please type the receiver's email:")
            receiver_email = input("Receiver's email: ")
            
            # Extract email details
            details = classification.get("details", {})
            subject = details.get("subject", "No Subject")
            body = details.get("body", "")
            
            # Format email body with names
            formatted_body = f"From: {sender_name}\n\n{body}\n\nBest regards,\n{sender_name}"
            
            # Send email using AIService
            email_service = AIService()
            result = await email_service.send_email(
                to=receiver_email,
                subject=subject,
                body=formatted_body
            )
            
            if result.get("status") == "success":
                speak("Email sent successfully!")
            else:
                speak("Failed to send email. Please try again.")
                
        else:
            # Route other tasks
            match classification["type"].upper():
                case "WEATHER":
                    result = await get_weather(classification["details"]["location"])
                    if "speak" in audio_input.lower():
                        speak(str(result))
                        
                case "REALTIME":
                    result = await real_time_search(audio_input)
                    if "speak" in audio_input.lower():
                        speak(str(result))
                        
                case "TODO":
                    todo_manager = TodoManager()
                    result = await todo_manager.process_natural_language_request(
                        classification["details"]["query"]
                    )
                    if "speak" in audio_input.lower():
                        speak("Todo task created successfully!")
                        
                case "WEBSEARCH":
                    result = await web_search(classification["details"]["query"])
                    if "speak" in audio_input.lower():
                        speak(str(result))
                        
                case _:
                    speak("I'm not sure how to handle that request.")
                    
    except Exception as e:
        logger.error(f"Error in task routing: {e}")
        speak("Sorry, there was an error processing your request.")

# Update prompt_gpt function to use new router
def prompt_gpt(audio):
    global listening_for_wakeword
    try:
        prompt_audio_path = "prompt.wav"
        with open(prompt_audio_path, "wb") as f:
            f.write(audio.get_wav_data())
        prompt_text = wav_to_text(prompt_audio_path)

        if not prompt_text.strip():
            speak("Empty prompt, please speak again")
            return

        print("User: ", prompt_text)
        
        # Use new task router instead of default conversation
        asyncio.run(listen_and_route_tasks(prompt_text))
        
        if "thank you for your help" in prompt_text.lower():
            print("Conversation ended by user.")
            speak("You're welcome! Have a great day!")
            listening_for_wakeword = True
        else:
            print(f"\nSay {wakeword} to wake me up")
            
    except Exception as e:
        print("Error: ", e)
        speak("I am sorry, I could not understand you, please try again")

def callback(recognizer, audio):
    global listening_for_wakeword
    if listening_for_wakeword:
        if listen_for_wake_word(audio):
            print("Wake word detected, ready for your command.")
            listening_for_wakeword = False
    else:
        prompt_gpt(audio)

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print("Say", wakeword, "to wake me up")
    r.listen_in_background(source, callback)
    while True:
        time.sleep(0.1)



#importing the modules from the other files, here, the bot will be able to use the functions from the other files, it will listen to the wakeword, if a command is given it would analyse the prompt, check for the right functions to call and assign that to the neccessaty function, the bot will be able to send emails, create tasks, summarize content, scrape webpages, get the weather, search in real time, process documents, ask questions, stream answers, stream web search, analyze prompts and route tasks, manage todos, and send emails, else, it would answer the default system message, and the user can end the conversation by saying "thank you for your help", the bot will respond with "You're welcome! Have a great day!" and the conversation will end, the bot will be able to respond to any question or statement that is asked of it or tasked to it, it will generate words in a user-friendly manner, it can also ask questions to the user to get more information, be playful and generate words of value prioritising logic and facts

# creating a function that will listen and route the tasks, but for the email, it has ti use the test_email function from the sendEmail.py file, the function will be able to send an email, test the service, get the gmail service, and the main function which was used to test the class, which could be modified to be used, the for the input of the name of the sender, name of the reciever, and the email address of the reciever, the bout will request the user to manually type it, then the audio functionality will continue
# the function will also be able to process the documents, process the urls, process the pdf, ask questions, stream answers, stream web search, and the main function which was used to test the class, which could be modified to be used

if __name__ == "__main__":
    start_listening()