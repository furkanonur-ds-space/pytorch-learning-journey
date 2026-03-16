# Topic: NLP From Scratch - Building the RNN
# Source: https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
# Summary:
#   1. Creating a custom RNN cell from scratch using nn.Module.
#   2. Combining the current input (letter) with the previous hidden state (memory).
#   3. Creating two linear layers: one for the new memory, one for the output prediction.
#   4. Applying LogSoftmax to get probabilities.

import torch
import torch.nn as nn

# ==========================================
# 1. DEFINE THE RNN ARCHITECTURE
# ==========================================
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        input_size: Number of characters in our alphabet (57)
        hidden_size: Size of our "memory" vector (e.g., 128)
        output_size: Number of languages/categories (18)
        """
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        # We create two Linear layers. 
        # Notice the input to both is (input_size + hidden_size).
        # This is because we will CONCATENATE the current letter and the past memory!
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # Input to Hidden (New Memory)
        self.i2o = nn.Linear(input_size + hidden_size, output_size) # Input to Output (Prediction)
        
        # LogSoftmax turns our raw output scores into probabilities (0 to 1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """
        This function runs FOR EVERY SINGLE LETTER in the word.
        """
        # Step 1: Combine current letter and previous memory side-by-side
        combined = torch.cat((input, hidden), 1)

        # Step 2: Calculate the NEW memory for the next step
        hidden = self.i2h(combined)

        # Step 3: Calculate the prediction for the current step
        output = self.i2o(combined)
        output = self.softmax(output)

        # We return both the prediction AND the new memory!
        return output, hidden

    def initHidden(self):
        """
        Before we start reading a new word, we must clear our memory.
        This returns a tensor of zeros for the initial hidden state.
        """
        return torch.zeros(1, self.hidden_size)

# ==========================================
# 2. QUICK SANITY CHECK (TESTING THE MODEL)
# ==========================================
if __name__ == "__main__":
    print("\n==========================================")
    print("TESTING THE RNN ARCHITECTURE")
    print("==========================================")
    
    # Mock parameters based on our dataset
    n_letters = 57      # ASCII letters + punctuation
    n_hidden = 128      # Arbitrary memory size
    n_categories = 18   # Number of languages
    
    # Instantiate the model
    rnn = RNN(n_letters, n_hidden, n_categories)
    print(f"Model Architecture:\n{rnn}")
    
    # Create a fake 'letter' (1x57 tensor) and a blank 'memory' (1x128 tensor)
    mock_input = torch.zeros(1, n_letters)
    mock_input[0][10] = 1 # Let's pretend this is the letter 'k'
    
    mock_hidden = rnn.initHidden()
    
    # Pass them through the model ONCE (simulating reading the first letter)
    output, next_hidden = rnn(mock_input, mock_hidden)
    
    print(f"\nInput shape: {mock_input.shape}")
    print(f"Hidden shape: {mock_hidden.shape}")
    print(f"Output shape: {output.shape} -> (Prediction for 18 languages)")
    print(f"Next Hidden shape: {next_hidden.shape} -> (Memory for the next letter)")
    print("\n✅ RNN Architecture script finished successfully.")