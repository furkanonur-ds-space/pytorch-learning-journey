# Topic: NLP From Scratch - Evaluating / Predicting
# Source: https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
# Summary:
#   1. Loading the saved model weights (.pth file).
#   2. Creating a prediction function that outputs the top 3 languages.
#   3. Building an interactive terminal loop for the user to play with.

import torch
import torch.nn as nn
from data_prep import all_categories, n_categories, n_letters, lineToTensor
from rnn_model import RNN

# ==========================================
# 1. LOAD THE TRAINED MODEL
# ==========================================
print("Loading trained model from 'char_rnn.pth'...")

n_hidden = 128
# 1. Instantiate the exact same architecture
rnn = RNN(n_letters, n_hidden, n_categories)

# 2. Load the weights from the file into our model
try:
    rnn.load_state_dict(torch.load('char_rnn.pth', weights_only=True))
    rnn.eval() # Set the model to evaluation mode (important for some layers like Dropout)
    print("✅ Model loaded successfully!\n")
except FileNotFoundError:
    print("❌ ERROR: 'char_rnn.pth' not found. Please run 03_train.py first to save the model!")
    exit()

# ==========================================
# 2. THE PREDICTION FUNCTION
# ==========================================
def predict(name, n_predictions=3):
    print(f"\n> Analyzing name: '{name}'")
    
    # Turn the name into a tensor
    tensor = lineToTensor(name)
    
    # Initialize blank memory
    hidden = rnn.initHidden()

    # We don't need to track gradients for prediction, so we use no_grad()
    with torch.no_grad():
        # Feed the name to the model letter by letter
        for i in range(tensor.size()[0]):
            output, hidden = rnn(tensor[i], hidden)

        # Get the top N categories
        # .topk(3) returns the 3 highest values and their indices
        topv, topi = output.topk(n_predictions, 1, True)
        
        # Print the results
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            # Convert logarithmic value back to a rough percentage-like confidence score
            confidence = math.exp(value) * 100 
            print(f"  {(i+1)}. {all_categories[category_index]:12s} (Confidence: {confidence:.2f}%)")

import math

# ==========================================
# 3. INTERACTIVE USER LOOP
# ==========================================
print("==========================================")
print("🤖 RNN NAME CLASSIFIER IS READY!")
print("Type 'quit' or 'q' to exit.")
print("==========================================")

while True:
    user_input = input("\nEnter a name (e.g., Satoshi, Yilmaz, MacDonald): ")
    if user_input.lower() in ['quit', 'q', 'exit']:
        print("Goodbye!")
        break
    
    if len(user_input) > 0:
        predict(user_input)