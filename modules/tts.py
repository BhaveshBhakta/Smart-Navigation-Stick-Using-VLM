from gtts import gTTS
import os

def text_to_speech(text, output_path="outputs/audio/output.mp3"):
    
    # Create audio folder if not exists
    os.makedirs("outputs/audio", exist_ok=True)

    # Convert text to speech
    tts = gTTS(text=text, lang="en")

    # Save audio file
    tts.save(output_path)

    return output_path