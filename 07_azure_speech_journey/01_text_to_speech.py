# Topic: Text to Speech (TTS) using Azure AI
# Source: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-text-to-speech
# Summary:
#   1. Set up Azure SpeechConfig with your Key and Region.
#   2. Choose a specific neural voice (e.g., a natural sounding English or Turkish voice).
#   3. Use SpeechSynthesizer to convert printed text into audio.
#   4. The audio is played on the default speaker and saved as a WAV file.
#
# Note: Azure's Neural TTS voices are nearly indistinguishable from humans.

import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# Load secrets from .env file
load_dotenv()

# =======================================================
# ⚠️ GET YOUR KEYS FROM AZURE PORTAL AND PUT IN .env FILE
# See README.md for instructions on how to get these for free.
# =======================================================
SPEECH_KEY = os.getenv("SPEECH_KEY", "YOUR_SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION", "YOUR_SPEECH_REGION")


def main():
    print("\n=======================================================")
    print("  Azure AI: Text to Speech (TTS)")
    print("=======================================================\n")

    if SPEECH_KEY == "YOUR_SPEECH_KEY":
        print("❌ ERROR: You haven't added your Azure API Key yet!")
        print("   Please read README.md to learn how to get a free key.")
        return

    # 1. Configure the Speech Service
    print("[*] Initializing Azure Speech Service...")
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    
    # Optional: Select a specific neural human voice. 
    # Example for US English: "en-US-JennyNeural" or "en-US-GuyNeural"
    # Example for Turkish: "tr-TR-EmelNeural" or "tr-TR-AhmetNeural"
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

    # 2. Configure output (Play to speaker AND save to file)
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    audio_file_path = os.path.join(output_dir, "speech_output.wav")
    
    # AudioOutputConfig defines where the synthesis goes
    # We create two configurations to do both: play out loud & save locally.
    audio_config_file = speechsdk.audio.AudioOutputConfig(filename=audio_file_path)
    
    # We will just use the default speaker output for the synthesizer
    audio_config_speaker = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    # Note: To do *both* easily, we'll synthesize to file first, then you can listen to the resulting WAV, 
    # OR we synthesize directly to the speaker default! We will just do speaker default here.
    # To keep it simple, let's output to the speaker first.
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config_speaker)

    # 3. Read input text limits
    print("📝 Type some text you want the AI to read out loud (Press Enter to stop):")
    text = input("> ")

    if not text.strip():
        print("No text provided. Exiting.")
        return

    # 4. Synthesize!
    print(f"\n[*] Sending text to Azure '{speech_config.speech_synthesis_voice_name}' voice...")
    
    # speak_text_async converts the text to audio synchronously on .get()
    result = synthesizer.speak_text_async(text).get()

    # 5. Handle Results
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"✅ Success! Speech synthesized for text: '{text}'")
        
        # Let's save a copy to the file as well!
        # Re-synthesize silently to the file path.
        file_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config_file)
        file_synthesizer.speak_text_async(text).get()
        print(f"💾 Audio copy saved to: {audio_file_path}")
        
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"❌ Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"   Error details: {cancellation_details.error_details}")
            print("   Did you enter the correct API Key and Region?")

if __name__ == "__main__":
    main()
