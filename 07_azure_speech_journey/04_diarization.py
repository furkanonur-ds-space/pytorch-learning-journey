# Topic: Conversation Transcription & Diarization
# Source: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-stt-diarization
# Summary:
#   1. Diarization is the process of distinguishing between different speakers 
#      in an audio stream (i.e. Speaker 1 vs Speaker 2).
#   2. Use ConversationTranscriber configured with standard STT keys.
#   3. Start continuous recognition on an audio file or microphone.
#   4. Event-based callbacks trigger when Azure identifies a speaker's sentence.

import os
import time
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
    print("  Azure AI: Diarization (Speaker Recognition)")
    print("=======================================================\n")

    if SPEECH_KEY == "YOUR_SPEECH_KEY":
        print("❌ ERROR: You haven't added your Azure API Key yet!")
        return

    # 1. Setup Service
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = "en-US"

    # NOTE: Diarization works best with Stereo Audio Files.
    # To test this via microphone, grab a friend and speak back and forth!
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    # 2. Conversation Transcriber 
    # This specific object handles tracking WHO is speaking.
    transcriber = speechsdk.transcription.ConversationTranscriber(
        speech_config=speech_config, audio_config=audio_config
    )

    # 3. Define Callback Functions
    # Conversation transcription is a continuous process. You must attach functions
    # to events like "Transcribed" which fires every time a speaker finishes a sentence.
    
    def transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            # Look here! `evt.result.speaker_id` tells us WHO spoke!
            speaker = evt.result.speaker_id if evt.result.speaker_id else "Unknown"
            print(f"🗣️ [{speaker}]: {evt.result.text}")

    def canceled_cb(evt: speechsdk.SessionEventArgs):
        print(f"\n❌ Transcription canceled: {evt.cancellation_details.reason}")
        if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {evt.cancellation_details.error_details}")

    # Attach callbacks
    transcriber.transcribed.connect(transcribed_cb)
    transcriber.canceled.connect(canceled_cb)
    transcriber.session_stopped.connect(lambda evt: print("\n🛑 Session stopped."))

    # 4. Start Listening Loop
    print("\n🎧 Starting Conversation Transcription...")
    print("   (Have a conversation with someone else. Azure will try to label 'Guest-1', 'Guest-2')")
    print("   >> Press Ctrl+C at any time to stop.\n")
    print("-" * 50)
    
    transcriber.start_transcribing_async().get()

    try:
        # Keep the main thread alive while callbacks run in the background
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping transcription...")
        transcriber.stop_transcribing_async().get()
    
    print("✅ Diarization gracefully stopped.")

if __name__ == "__main__":
    main()
