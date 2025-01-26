import os
import wave
import tempfile
import pyaudio
import google.generativeai as genai
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
from ai import send_email, handle_task_creation, create_task, get_gmail_service, scrape_url, summarize_content
from docsprocessing import RAGProcessor # Import the RAGProcessor class from docsprocessing.py, RAGPprocessor it has functions to process_documentsm _process_urls, _process_pdf, ask_question, stream_answer, _stram_web_search and the main function which was used to test the class, which could be modified to be used to be used
from realtimeSearch import get_current_time, real_time_search
from weather import get_weather
from sendEmail import test_service, send_email
from tasks import analyze_prompt_and_route_task
from todo import TodoManager
from webScrapeAndProcess import web_search, scrape_url, summarize_content, scrape_webpages_with_serpapi
load_dotenv()

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"

# Initialize Gemini client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class AudioSystem:
    def __init__(self):
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.record_seconds = 5
        
        # Initialize Google Speech-to-Text client
        self.speech_client = speech.SpeechClient()
        
        # Initialize Google Text-to-Speech client
        self.tts_client = texttospeech.TextToSpeechClient()
        
        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-pro')

    def record_audio(self):
        """Record audio from microphone"""
        audio = pyaudio.PyAudio()
        stream = audio.open(format=self.format, channels=self.channels,
                          rate=self.rate, input=True,
                          frames_per_buffer=self.chunk)
        
        print("Recording...")
        frames = []
        
        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)
            
        print("Recording finished")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Save recording temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            wf = wave.open(temp_audio.name, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            return temp_audio.name

    def speech_to_text(self, audio_file):
        """Convert speech to text using Google Speech-to-Text"""
        try:
            with open(audio_file, "rb") as audio_content:
                content = audio_content.read()
            
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.rate,
                language_code="en-US"
            )
            
            response = self.speech_client.recognize(config=config, audio=audio)
            
            # Extract the first transcription
            transcript = response.results[0].alternatives[0].transcript if response.results else ""
            return transcript
        
        except Exception as e:
            print(f"Speech-to-text error: {e}")
            return None

    def text_to_gemini(self, text):
        """Generate response using Gemini"""
        try:
            response = self.model.generate_content(text)
            return response.text
        except Exception as e:
            print(f"Gemini generation error: {e}")
            return None

    def text_to_speech(self, text):
        """Convert text to speech using Google Cloud TTS"""
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            # Play audio using pydub
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                temp_audio.write(response.audio_content)
                audio = AudioSegment.from_mp3(temp_audio.name)
                play(audio)
                os.unlink(temp_audio.name)
        except Exception as e:
            print(f"Text-to-speech error: {e}")

