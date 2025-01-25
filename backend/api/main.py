from audio import AudioSystem


async def main():
    system = AudioSystem()
    while True:
        audio_file = await system.record_audio()
        if audio_file:
            text = await system.speech_to_text(audio_file)
            if text:
                result = await system.process_command(text)
                if result.get("status") == "success":
                    await system.text_to_speech(result.get("data"))