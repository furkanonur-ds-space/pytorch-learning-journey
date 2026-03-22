# Topic: Text-to-Image with FLUX.1-dev (Zero-Download Cloud API)
# Source: https://huggingface.co/docs/api-inference/detailed_parameters
# Summary:
#   1. Sending prompts directly to Hugging Face's extremely powerful servers.
#   2. We don't download the 23GB model, we don't use our own GPU.
#   3. FLUX.1-dev requires an HF_TOKEN from a user who has accepted the model license.
#   4. The response is raw image bytes, which we save as a .jpg file.

import os
import requests
from dotenv import load_dotenv

# Load the secret HF_TOKEN from the .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN")

# The endpoint for the massive 12 Billion parameter FLUX model
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def generate_image(prompt: str, output_filename: str):
    """Sends a text prompt to Hugging Face API and saves the resulting image."""
    
    print(f"[*] Sending prompt: '{prompt}'")
    print("[*] Waiting for Hugging Face giant servers to generate the image...")
    
    # We send a POST request with our JSON payload
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        # The API returns the raw binary image data
        with open(output_filename, "wb") as file:
            file.write(response.content)
        print(f"✅ Success! Image saved as '{output_filename}'")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        if response.status_code == 401:
            print("\n   👉 Check your HF_TOKEN in the .env file!")
        elif response.status_code == 403:
            print("\n   👉 You must go to https://huggingface.co/black-forest-labs/FLUX.1-dev")
            print("      and click 'Agree and access repository' to accept the FLUX license!")

def main():
    print("\n=======================================================")
    print("  Hugging Face AI: FLUX.1-dev (Serverless API)")
    print("=======================================================\n")

    if HF_TOKEN == "YOUR_HF_TOKEN" or not HF_TOKEN:
        print("❌ ERROR: You haven't added your HF_TOKEN yet!")
        print("   Please rename .env.example to .env and add your token.")
        return

    print("📝 Type an English description of what you want to draw:")
    print("   (Example: 'A futuristic cyberpunk cat wearing neon sunglasses in Tokyo')")
    
    prompt = input("> ")
    if not prompt.strip():
        print("No prompt provided. Exiting.")
        return

    # Let's create an 'outputs' folder to stay organized
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the file with a clean name based on the prompt
    clean_prompt = "".join(x for x in prompt if x.isalnum() or x in " ").strip()
    filename = os.path.join(output_dir, f"flux_{clean_prompt[:20].replace(' ', '_')}.jpg")

    generate_image(prompt, filename)

if __name__ == "__main__":
    main()
