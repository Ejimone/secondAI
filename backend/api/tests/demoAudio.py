import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional
import numpy as np
import sounddevice as sd
import soundfile as sf
from google.cloud import speech_v1, texttospeech
import tempfile
from pathlib import Path
import wave

# Import local modules
from sendEmail import AIService
from weather import get_weather
from docsprocessing import RAGProcessor
from realtimeSearch import real_time_search
from webScrapeAndProcess import web_search, summarize_content
from tasks import analyze_prompt_and_route_task

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.ai_service = AIService()
        self.sample_rate = 16000
        self.channels = 1
        self.speech_client = speech_v1.SpeechClient()
        self.tts_client = texttospeech.TextToSpeechClient()
        self.rag_processor = RAGProcessor()
        
    async def record_audio(self, duration: int = 5) -> Optional[np.ndarray]:
        """Record audio from microphone"""
        try:
            logger.info("Recording started...")
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.int16
            )
            sd.wait()
            logger.info("Recording finished")
            return recording
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return None

    async def speech_to_text(self, audio_data: np.ndarray) -> Optional[str]:
        """Convert speech to text"""
        try:
            # Save audio data temporarily
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                sf.write(temp_audio.name, audio_data, self.sample_rate)
                
                # Read the audio file
                with open(temp_audio.name, "rb") as audio_file:
                    content = audio_file.read()

            # Configure audio and recognition settings
            audio = speech_v1.RecognitionAudio(content=content)
            config = speech_v1.RecognitionConfig(
                encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code="en-US",
                model="command_and_search",
            )

            # Perform the transcription
            response = self.speech_client.recognize(config=config, audio=audio)
            
            if response.results:
                return response.results[0].alternatives[0].transcript
            return None

        except Exception as e:
            logger.error(f"Speech to text failed: {e}")
            return None

    async def text_to_speech(self, text: str) -> bool:
        """Convert text to speech and play it"""
        try:
            # Configure voice settings
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16
            )

            # Generate speech
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            # Play audio
            audio_data = np.frombuffer(response.audio_content, dtype=np.int16)
            sd.play(audio_data, self.sample_rate)
            sd.wait()
            return True

        except Exception as e:
            logger.error(f"Text to speech failed: {e}")
            return False

    async def process_command(self, text: str) -> Dict[str, Any]:
        """Process voice command and route to appropriate function"""
        try:
            # Use Gemini to classify the command
            classification_prompt = f"""
            Analyze this voice command and classify it:
            "{text}"
            
            Return EXACTLY in this format:
            {{
                "type": "email/weather/search/todo/doc_process",
                "action": "specific action",
                "parameters": {{
                    "location": "if applicable",
                    "query": "if applicable",
                    "url": "if applicable"
                }}
            }}
            """
            
            response = self.ai_service.gemini_model.generate_content(classification_prompt)
            command = json.loads(response.text)
            
            # Route to appropriate handler
            if command["type"] == "email":
                return await analyze_prompt_and_route_task(text)
                
            elif command["type"] == "weather":
                return await get_weather(command["parameters"].get("location", ""))
                
            elif command["type"] == "search":
                return await web_search(command["parameters"].get("query", ""))
                
            elif command["type"] == "doc_process":
                if "url" in command["parameters"]:
                    return await summarize_content(command["parameters"]["url"], 
                                                self.ai_service.gemini_model)
                                                
            elif command["type"] == "realtime":
                return await real_time_search(text)

            return {"status": "error", "message": "Unknown command type"}

        except Exception as e:
            logger.error(f"Command processing error: {e}")
            return {"status": "error", "message": str(e)}

    async def run(self):
        """Main interaction loop"""
        logger.info("Starting voice assistant...")
        await self.text_to_speech("Hello! I'm ready to help you.")
        
        while True:
            try:
                # Record audio
                audio_data = await self.record_audio()
                if audio_data is None:
                    continue
                
                # Convert to text
                text = await self.speech_to_text(audio_data)
                if not text:
                    continue
                
                logger.info(f"Recognized: {text}")
                
                # Process command
                result = await self.process_command(text)
                
                # Convert result to speech
                response_text = result.get("data", result.get("message", ""))
                if response_text:
                    await self.text_to_speech(response_text)
                
            except Exception as e:
                logger.error(f"Runtime error: {e}")
            
            await asyncio.sleep(0.1)

async def main():
    processor = AudioProcessor()
    await processor.run()

if __name__ == "__main__":
    asyncio.run(main())