# Topic: What can Transformers do? (Pipelines)
# Source: https://huggingface.co/learn/nlp-course/chapter1/3
# Summary:
#   1. The `pipeline()` function is the most high-level API in Hugging Face.
#   2. It connects a model with its necessary preprocessing (tokenizer) and postprocessing steps.
#   3. You can use it for sentiment analysis, text generation, zero-shot classification, etc.

from transformers import pipeline

def main():
    print("\n=======================================================")
    print("  Hugging Face NLP Course: 1. Pipelines")
    print("=======================================================\n")

    # ---------------------------------------------------------
    # 1. Sentiment Analysis
    # ---------------------------------------------------------
    print("\n--- 1. Sentiment Analysis ---")
    classifier = pipeline("sentiment-analysis")
    # This automatically downloads a default model (usually DistilBERT)
    sentences = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!"
    ]
    results = classifier(sentences)
    for text, result in zip(sentences, results):
        print(f"Text: '{text}'")
        print(f" -> Result: {result['label']}, Score: {result['score']:.4f}\n")

    # ---------------------------------------------------------
    # 2. Zero-Shot Classification
    # ---------------------------------------------------------
    # Zero-shot means the model hasn't been explicitly fine-tuned on these specific labels!
    print("--- 2. Zero-Shot Classification ---")
    classifier_zero_shot = pipeline("zero-shot-classification")
    text = "This is a course about the Transformers library and natural language processing."
    candidate_labels = ["education", "politics", "business"]
    
    print(f"Text: '{text}'")
    print(f"Candidate Labels: {candidate_labels}")
    
    result_zs = classifier_zero_shot(text, candidate_labels=candidate_labels)
    for label, score in zip(result_zs['labels'], result_zs['scores']):
        print(f" -> Label: {label}, Score: {score:.4f}")

    # ---------------------------------------------------------
    # 3. Text Generation
    # ---------------------------------------------------------
    print("\n--- 3. Text Generation ---")
    # We explicitly specify a smaller model (distilgpt2) so it downloads quickly
    generator = pipeline("text-generation", model="distilgpt2")
    prompt = "In this course, we will teach you how to"
    print(f"Prompt: '{prompt}...'")
    
    results_gen = generator(
        prompt, 
        max_length=30, # Generate up to 30 tokens
        num_return_sequences=2 # Give me 2 different variations
    )
    
    for i, res in enumerate(results_gen):
        print(f" Variant {i+1}: {res['generated_text']}")

if __name__ == "__main__":
    main()
