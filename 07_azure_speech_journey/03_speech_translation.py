# Topic: Real-Time Speech Translation
# Source: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-speech-translation
# Summary:
#   1. Set up Azure SpeechTranslationConfig.
#   2. Define the exact language you will speak (e.g., 'en-US' or 'tr-TR').
#   3. Define multiple target languages you want the text translated into.
#   4. Use TranslationRecognizer to capture mic audio and translate it instantly.

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
    print("  Azure AI: Real-Time Speech Translation")
    print("=======================================================\n")

    if SPEECH_KEY == "YOUR_SPEECH_KEY":
        print("❌ ERROR: You haven't added your Azure API Key yet!")
        return

    # 1. Setup Configure Speech Translation Service
    print("[*] Initializing Azure Translation Service...")
    translation_config = speechsdk.translation.SpeechTranslationConfig(
        subscription=SPEECH_KEY, region=SPEECH_REGION
    )
    
    # 2. DEFINING LANGUAGES
    # The language you will SPEAK (Source Language)
    # E.g. English (en-US) or Turkish (tr-TR)
    translation_config.speech_recognition_language = "en-US"
    
    # The languages you want it translated TO
    # We can request multiple languages simultaneously!
    translation_config.add_target_language("tr") # Turkish
    translation_config.add_target_language("de") # German
    translation_config.add_target_language("ja") # Japanese

    # 3. Configure microphone & Recognizer
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config, audio_config=audio_config
    )

    # 4. Listen!
    print(f"\n🎙️  Speak an English sentence into your mic now...")
    print(f"    (Azure will translate it to Turkish, German, and Japanese)\n")
    
    result = recognizer.recognize_once_async().get()

    # 5. Process result
    if result.reason == speechsdk.ResultReason.TranslatedSpeech:
        print(f"✅ Original (English): '{result.text}'\n")
        
        # Pull translations from the dictionary
        print("🌍 Translations:")
        print(f"  🇹🇷 Turkish : {result.translations['tr']}")
        print(f"  🇩🇪 German  : {result.translations['de']}")
        print(f"  🇯🇵 Japanese: {result.translations['ja']}")
        
    elif result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"✅ Recognized: '{result.text}' (But no translation was produced)")
        
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("🤔 No speech could be recognized.")
        
    elif result.reason == speechsdk.ResultReason.Canceled:
        print(f"❌ Recognition canceled: {result.cancellation_details.reason}")
        if result.cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"   Error details: {result.cancellation_details.error_details}")

if __name__ == "__main__":
    main()
