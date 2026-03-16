# Topic: NLP From Scratch - Data Preprocessing
# Source: https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
# Summary:
#   1. Download and extract the dataset (thousands of surnames from 18 languages).
#   2. Convert Unicode strings to ASCII (e.g., Ślusàrski -> Slusarski).
#   3. Build dictionaries to organize names by language.
#   4. Convert words (strings) into PyTorch Tensors using One-Hot Encoding.

import os
import urllib.request
import zipfile
import glob
import unicodedata
import string
import torch

print("\n==========================================")
print("1. DOWNLOAD & EXTRACT DATASET")
print("==========================================")

data_url = "https://download.pytorch.org/tutorial/data.zip"
data_dir = "./data"

# Download and extract if data doesn't exist
if not os.path.exists(os.path.join(data_dir, "names")):
    print("Downloading dataset...")
    urllib.request.urlretrieve(data_url, "data.zip")
    print("Extracting dataset...")
    with zipfile.ZipFile("data.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    os.remove("data.zip")
    print("✅ Dataset ready!")
else:
    print("✅ Dataset already exists.")

# ==========================================
# 2. ALPHABET & UNICODE CONVERSION
# ==========================================
# We define the vocabulary: all ASCII letters plus a few punctuation marks.
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII
# Example: "Ślusàrski" -> "Slusarski"
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(f"Alphabet length: {n_letters}")
print(f"Test Unicode to ASCII ('Ślusàrski'): {unicodeToAscii('Ślusàrski')}")

# ==========================================
# 3. LOAD DATA INTO DICTIONARIES
# ==========================================
# Build the category_lines dictionary, a list of names per language
# category_lines = {'English': ['Smith', 'Jones'], 'Turkish': ['Yilmaz', 'Kaya']}
category_lines = {}
all_categories = []

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in glob.glob('data/names/*.txt'):
    # Extract the language name from the filename (e.g., 'data/names/Turkish.txt' -> 'Turkish')
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
print(f"\nLoaded {n_categories} categories (languages).")
print(f"Sample English names: {category_lines['English'][:5]}")
# ==========================================
# 4. CONVERTING NAMES TO TENSORS (ONE-HOT)
# ==========================================
print("\n==========================================")
print("4. TENSOR CONVERSION (ONE-HOT ENCODING)")
print("==========================================")

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line (word) into a <line_length x 1 x n_letters> Tensor
# This is an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

test_word = "Jones"
word_tensor = lineToTensor(test_word)
print(f"Word: '{test_word}'")
print(f"Tensor Shape: {word_tensor.shape} -> (sequence_length, batch_size, input_size)")
print(f"Tensor representation of 'J':\n{word_tensor[0]}")

print("\n✅ Data Preprocessing script finished successfully.")