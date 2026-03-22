# Topic: Azure AI Speech Registration Guide & UX Links
# Source: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/
# Summary: Guide on how to get Azure API Keys and explore Advanced UX tryouts.

# 🔑 HOW TO GET AZURE SPEECH API KEYS (Ücretsiz Kayıt Rehberi)

To run the Python scripts in this folder, you need an **Azure Speech Resource Key** and a **Region**. Follow these steps to get them for free:

1. **Go to Azure Portal**: Visit [portal.azure.com](https://portal.azure.com/).
2. **Create an Account**: If you don't have a Microsoft account, register. Microsoft gives you **$200 free credit** for the first month and **free tiers** for many services for 12 months.
3. **Search for Speech**: In the top search bar, type `Speech` and select **Speech services** under cognitive services.
4. **Create a Resource**:
   - Click **Create**.
   - Select your Subscription (Free Trial or Azure for Students).
   - Create a new Resource Group (e.g. `speech-learning-rg`).
   - Select a Region close to you (e.g. `northeurope` or `westeurope` or `eastus`). **Remember this Region name!**
   - Give it a Name (e.g. `my-awesome-speech-ai`).
   - Pricing Tier: Select **Free F0** (this gives you 5 hours of free speech processing per month forever!).
   - Click **Review + Create**, then **Create**.
5. **Get Your Keys**:
   - Once deployment is complete, click **Go to resource**.
   - On the left menu, click **Keys and Endpoint**.
   - Copy **KEY 1** and your **Location/Region**.
6. **Paste Them in Code**: Open the Python scripts in this folder and replace `YOUR_SPEECH_KEY` and `YOUR_SPEECH_REGION` with your copied values!

---

# 🌐 ADVANCED WEB PLAYGROUNDS (UX Tryouts)

Your professor shared two amazing visual playgrounds for advanced Speech applications. These are best experienced directly in the browser rather than via terminal scripts:

### 1. Pronunciation Assessment (Telaffuz Değerlendirme)
**Link:** [Azure AI Speech - Pronunciation Assessment](https://ai.azure.com/explore/models/aiservices/Azure-AI-Speech/version/1/registry/azureml-cogsvc/tryout?NewUX=true&Trigger=AutoRedirect_NoSpeechResources#pronunciationassessment)
- **What it does:** You read a text into the microphone, and the AI grades your pronunciation, fluency, accuracy, and even points out exactly which syllables you mispronounced! Great for language learning apps.

### 2. Custom Speech (Özel Konuşma / Ses Modeli Eğitimi)
**Link:** [Azure AI Speech - Custom Speech](https://ai.azure.com/explore/models/aiservices/Azure-AI-Speech/version/1/registry/azureml-cogsvc/tryout?NewUX=true&Trigger=AutoRedirect_NoSpeechResources#customspeech)
- **What it does:** Allows you to upload your own voice recordings to teach the AI *your specific voice* or very specific industry jargon (medical, acoustic environments).
