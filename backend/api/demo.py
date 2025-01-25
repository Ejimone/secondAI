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

def test_audio_system():
    """Test the audio interaction system"""
    system = AudioSystem()
    
    print("Starting test...")
    # Record audio
    audio_file = system.record_audio()
    
    # Convert speech to text
    if audio_file:
        text = system.speech_to_text(audio_file)
        print(f"Transcribed text: {text}")
        
        if text:
            # Process with Gemini
            response = system.text_to_gemini(text)
            print(f"AI Response: {response}")
            
            if response:
                # Convert response to speech
                system.text_to_speech(response)
        
        # Cleanup
        os.unlink(audio_file)

if __name__ == "__main__":
    test_audio_system()