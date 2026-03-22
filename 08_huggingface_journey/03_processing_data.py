# Topic: Processing Datasets
# Source: https://huggingface.co/learn/nlp-course/chapter3/2
# Summary:
#   1. Load a massive dataset directly from the Hugging Face Hub using `datasets`.
#   2. Inspect the dataset features (training vs validation splits, features, rows).
#   3. Process the dataset using a tokenizer.
#   4. Use the `map()` method for insanely fast, memory-mapped batch tokenization.

from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    print("\n=======================================================")
    print("  Hugging Face NLP Course: 3. Processing Datasets")
    print("=======================================================\n")

    # ---------------------------------------------------------
    # 1. Loading a Dataset
    # ---------------------------------------------------------
    print("\n--- 1. Loading the MRPC Dataset ---")
    # MRPC (Microsoft Research Paraphrase Corpus) is a standard benchmark dataset
    # It contains pairs of sentences and a label indicating if they mean the same thing.
    raw_datasets = load_dataset("glue", "mrpc")
    
    print("Dataset Metadata:")
    print(raw_datasets)

    # Let's peek at the first row in the training set
    print("\nFirst row of Training Data:")
    raw_train_dataset = raw_datasets["train"]
    print(raw_train_dataset[0])
    
    # We can see 'sentence1', 'sentence2', 'label' (0 or 1), and 'idx' (ID).
    
    # ---------------------------------------------------------
    # 2. Tokenizing
    # ---------------------------------------------------------
    print("\n--- 2. Setting up the Tokenizer ---")
    checkpoint = "bert-base-uncased" # A standard BERT model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # We need a function that takes a batch of data and tokenizes sentence1 and sentence2
    # Hugging Face tokenizers can automatically handle pairs of sentences!
    def tokenize_function(example):
        # We don't pad here yet (we'll do that dynamically later in the course during batching), 
        # but we TRUNCATE so we don't exceed BERT's maximum length.
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    print("\nTest Tokenizing a single pair:")
    example_tokens = tokenize_function(raw_train_dataset[0])
    print(f"Input IDs: {example_tokens['input_ids'][:10]}... (truncated for display)")
    # Notice the token IDs include special tokens [CLS] and [SEP] exactly where they belong!

    # ---------------------------------------------------------
    # 3. Applying to the Full Dataset (The .map() function)
    # ---------------------------------------------------------
    print("\n--- 3. Mapping Tokenizer across entire Dataset ---")
    # `map()` applies our function to every row. 
    # `batched=True` speeds it up tremendously by processing multiple elements at once (multithreading enabled!)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    print("\nTokenized Dataset Metadata:")
    print(tokenized_datasets)

    print("\nFirst row of Tokenized Training Data (Notice the new 'input_ids' column!):")
    print(tokenized_datasets["train"][0].keys())
    
    # Normally, we would now remove 'sentence1', 'sentence2', etc. before passing them to PyTorch
    # because the Model only cares about 'input_ids', 'attention_mask' and 'label'.
    print("\n✅ Dataset successfully downloaded and preprocessed in memory-efficient Arrow format!")

if __name__ == "__main__":
    main()
