# Topic: Save and Load the Model
# Source: https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
# Summary: 
#   1. Saving and Loading Model Weights (state_dict) -> The Recommended Way.
#   2. Saving and Loading Models with Shapes (The whole object).

import torch
import torchvision.models as models
from torch import nn

# ==========================================
# 0. SETUP: DEFINING THE MODEL
# ==========================================
# To load weights, we first need to create an instance of the model structure.
# (This is the same model we used in previous tutorials)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Initialize the model
model = NeuralNetwork()
print("\n--- Model Created (Random Weights) ---")


# ==========================================
# 1. SAVING AND LOADING MODEL WEIGHTS (RECOMMENDED)
# ==========================================
# PyTorch models store learned parameters in an internal state dictionary, called ``state_dict``.
# This is just a Python dictionary that maps each layer to its parameter tensor.

print("\n--- 1. Saving Model Weights (state_dict) ---")

# Let's save the model's weights to a file called 'model_weights.pth'
torch.save(model.state_dict(), 'model_weights.pth')
print("Model weights saved to 'model_weights.pth'")


print("\n--- 2. Loading Model Weights ---")
# To load weights, we must first create a new instance of the same model class.
model2 = NeuralNetwork() 

# Now we load the dictionary we saved into this new model.
model2.load_state_dict(torch.load('model_weights.pth', weights_only=True))

# IMPORTANT: Call .eval() before inference!
# This sets dropout and batch normalization layers to evaluation mode.
# If you forget this, you might get inconsistent inference results.
model2.eval() 

print("Model weights loaded successfully into 'model2'.")


# ==========================================
# 2. SAVING AND LOADING MODELS WITH SHAPES
# ==========================================
# When loading model weights, we needed to instantiate the model class first, 
# because the class defines the structure of a network. 
# We might want to save the structure of this class together with the model.

print("\n--- 3. Saving Entire Model (Structure + Weights) ---")

# We pass the whole 'model' object, not just 'model.state_dict()'
torch.save(model, 'model_full.pth')
print("Entire model saved to 'model_full.pth'")


print("\n--- 4. Loading Entire Model ---")
# We don't need to create an instance of NeuralNetwork() first.
# The load function brings back the class structure AND the weights.
model3 = torch.load('model_full.pth', weights_only=False)
model3.eval()

print("Entire model loaded successfully as 'model3'.")
print(f"Model Structure: \n{model3}")

print("\nCONGRATULATIONS! You have completed the PyTorch Basics Series!")