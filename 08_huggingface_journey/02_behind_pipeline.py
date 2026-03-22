# Topic: Behind the Pipeline (Tokenizers & Models)
# Source: https://huggingface.co/learn/nlp-course/chapter2/2
# Summary:
#   1. A pipeline hides 3 steps: Tokenizer -> Model -> Postprocessing.
#   2. Tokenizer converts text into numerical IDs.
#   3. Model takes IDs and outputs high-dimensional raw scores (Logits).
#   4. Postprocessing (Softmax) converts raw Logits into readable Probabilities.

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    print("\n=======================================================")
    print("  Hugging Face NLP Course: 2. Behind the Pipeline")
    print("=======================================================\n")

    # The exact same model default used by pipeline("sentiment-analysis")
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

    print(f"[*] Downloading/Loading Tokenizer and Model: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    # Input data
    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    print(f"\nRaw Inputs:\n 1. {raw_inputs[0]}\n 2. {raw_inputs[1]}")

    # ---------------------------------------------------------
    # STEP 1: PREPROCESSING (Tokenizer)
    # ---------------------------------------------------------
    # padding=True: Adds 0s so both sentences are the same length.
    # truncation=True: Cuts off words if they exceed the model's max limit.
    # return_tensors="pt": Return PyTorch Tensors instead of standard Python lists.
    print("\n--- Step 1: Tokenization ---")
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    
    print("Input IDs (Tokens converted to dictionary IDs):")
    print(inputs["input_ids"])
    print("\nAttention Mask (1=Real word, 0=Padding to ignore):")
    print(inputs["attention_mask"])

    # ---------------------------------------------------------
    # STEP 2: GOING THROUGH THE MODEL
    # ---------------------------------------------------------
    print("\n--- Step 2: Model Inference ---")
    # We pass the tokenized dictionary directly to the model as keyword arguments
    with torch.no_grad(): # We don't need gradients for inference, saves memory!
        outputs = model(**inputs)

    # The output is raw, unnormalized scores called "Logits"
    print("Raw Logits (Unnormalized scores):")
    print(outputs.logits)
    # E.g. tensor([[-1.56, 1.61], [4.16, -3.34]])

    # ---------------------------------------------------------
    # STEP 3: POSTPROCESSING (Softmax)
    # ---------------------------------------------------------
    print("\n--- Step 3: Postprocessing (Softmax) ---")
    # Softmax converts logits into probabilities that sum up to 1 over the classes.
    predictions = F.softmax(outputs.logits, dim=-1)
    print("Probabilities (After Softmax):")
    print(predictions)

    # Let's map these probabilities to human-readable labels!
    print("\n--- Final Results Map ---")
    labels = model.config.id2label # Gives us {0: 'NEGATIVE', 1: 'POSITIVE'}
    
    for i in range(len(raw_inputs)):
        pred_probs = predictions[i]
        predicted_class_id = torch.argmax(pred_probs).item()
        predicted_label = labels[predicted_class_id]
        confidence = pred_probs[predicted_class_id].item()
        
        print(f"Sentence {i+1}: Evaluated as {predicted_label} (Confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()
