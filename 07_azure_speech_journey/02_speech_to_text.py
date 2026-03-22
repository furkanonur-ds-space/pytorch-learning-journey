# Topic: Speech to Text (STT) Real-time Transcription
# Source: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-speech-to-text
# Summary:
#   1. Set up Azure SpeechConfig.
#   2. Use the default microphone as the audio input.
#   3. Wait for the user to speak a single phrase.
#   4. Send the audio to Azure and receive the transcript back immediately.

import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# Load secrets from .env file
load_dotenv()

# =======================================================
# ⚠️ GET YOUR KEYS FROM AZURE PORTAL AND PUT IN .env FILE
# =======================================================
SPEECH_KEY = os.getenv("SPEECH_KEY", "YOUR_SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION", "YOUR_SPEECH_REGION")

def main():
    print("\n=======================================================")
    print("  Azure AI: Speech to Text (STT)")
    print("=======================================================\n")

    if SPEECH_KEY == "YOUR_SPEECH_KEY":
        print("❌ ERROR: You haven't added your Azure API Key yet!")
        print("   Please read README.md to learn how to get a free key.")
        return

    # 1. Setup Configure Speech Service
    print("[*] Initializing Azure STT Service...")
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    
    # Optional: Set the spoken language you expect the user to be speaking. 
    # Default is en-US. If you want to speak Turkish, change this to "tr-TR".
    speech_config.speech_recognition_language = "en-US"

    # 2. Configure microphone
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    
    # 3. Create the Recognizer
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # 4. Listen!
    print("\n🎙️  Microphone is ON! Please speak a clear sentence into your mic...")
    
    # recognize_once_async() listens for a single utterance (until you pause speaking)
    result = recognizer.recognize_once_async().get()

    # 5. Process result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"\n✅ Recognized Text: '{result.text}'")
        
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("\n🤔 No speech could be recognized. Did you say something?")
        
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"\n❌ Speech Recognition canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"   Error details: {cancellation_details.error_details}")

if __name__ == "__main__":
    main()
