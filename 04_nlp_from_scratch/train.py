# Topic: NLP From Scratch - Training the RNN
# Source: https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
# Summary:
#   1. Importing our custom data and model.
#   2. Creating helper functions to fetch random training examples.
#   3. Defining the training loop (processing words letter by letter).
#   4. Using NLLLoss and SGD Optimizer to train the network.

import torch
import torch.nn as nn
import random
import time
import math

# Import our custom modules!
from data_prep import all_categories, n_categories, category_lines, n_letters, lineToTensor
from rnn_model import RNN

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
# Find the predicted category (language) from the network's 1x18 output
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

# Get a random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random language and a random name from that language
def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    
    # Create the target tensor (the correct language index)
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    # Create the input tensor (the word converted to one-hot vectors)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

# ==========================================
# 2. SETUP MODEL, LOSS, AND OPTIMIZER
# ==========================================
n_hidden = 128
# Create the RNN using our imported class
rnn = RNN(n_letters, n_hidden, n_categories)

# We use NLLLoss (Negative Log Likelihood Loss) because our model's 
# last layer is LogSoftmax. (LogSoftmax + NLLLoss = CrossEntropyLoss)
criterion = nn.NLLLoss()

# We use SGD optimizer instead of manual weight updates
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

# ==========================================
# 3. THE TRAINING FUNCTION
# ==========================================
def train(category_tensor, line_tensor):
    # 1. Clear memory before reading a new word
    hidden = rnn.initHidden()
    
    # 2. Zero the gradients
    optimizer.zero_grad()
    
    # 3. FORWARD PASS: Read the word letter by letter!
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
        
    # 'output' is now the prediction after reading the VERY LAST letter
    # 4. Compute Loss
    loss = criterion(output, category_tensor)
    
    # 5. BACKWARD PASS
    loss.backward()
    
    # 6. Update weights
    optimizer.step()
    
    return output, loss.item()

# ==========================================
# 4. THE MAIN TRAINING LOOP
# ==========================================
n_iters = 100000
print_every = 5000

print("\n==========================================")
print("🚀 STARTING TRAINING (100,000 Iterations)")
print("==========================================")

start = time.time()
current_loss = 0

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s:.0f}s'

for iter in range(1, n_iters + 1):
    # Get a random training example
    category, line, category_tensor, line_tensor = randomTrainingExample()
    
    # Train the model on this example
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    
    # Print progress
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else f'✗ ({category})'
        print(f"Iter: {iter:6d} | {(iter/n_iters)*100:.0f}% | Time: {timeSince(start)} | Loss: {loss:.4f} | Name: {line:15s} | Guess: {guess:10s} {correct}")

print("✅ Training complete!")

# Save the trained model weights to a file
torch.save(rnn.state_dict(), 'char_rnn.pth')
print("💾 Model saved to 'char_rnn.pth'")