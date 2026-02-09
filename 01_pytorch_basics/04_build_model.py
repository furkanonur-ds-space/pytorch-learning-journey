# Topic: Build the Neural Network
# Source: https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# Summary: 
#   1. Defining the Neural Network class (subclassing nn.Module).
#   2. Initializing layers (Linear, ReLU, Flatten).
#   3. Defining the forward pass (how data flows).
#   4. Inspecting model structure and parameters.

import torch
from torch import nn

# ==========================================
# 1. GET DEVICE FOR TRAINING
# ==========================================
# We want to train our model on a hardware accelerator like GPU (CUDA/MPS) if available.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# ==========================================
# 2. DEFINE THE CLASS
# ==========================================
# Every PyTorch model must inherit from nn.Module.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() # Initialize the parent class (nn.Module)
        
        # 1. Flatten: Converts 2D image (28x28) into a 1D array (784 pixels)
        self.flatten = nn.Flatten()
        
        # 2. Sequential Stack: A container that chains layers together.
        # Data flows through them in the order defined here.
        self.linear_relu_stack = nn.Sequential(
            # Layer 1: Takes 784 inputs -> Outputs 512 features
            nn.Linear(28*28, 512),
            # Activation: Adds non-linearity (helps learn complex patterns)
            nn.ReLU(),
            
            # Layer 2: Takes 512 inputs -> Outputs 512 features
            nn.Linear(512, 512),
            nn.ReLU(),
            
            # Output Layer: Takes 512 inputs -> Outputs 10 scores (Logits)
            # Why 10? Because we have 10 classes (T-shirt, Dress, etc.)
            nn.Linear(512, 10)
        )

    # The forward function defines how the data moves through the network.
    # PyTorch automatically handles the backward pass (gradients) for us.
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Create an instance of the model and move it to the device (GPU)
model = NeuralNetwork().to(device)
print(f"\nModel Structure: \n{model}\n")


# ==========================================
# 3. TEST WITH RANDOM DATA (PREDICTION)
# ==========================================
print("--- Testing the Model with Random Input ---")

# Create a dummy input (1 image, 1 channel, 28x28 pixels)
X = torch.rand(1, 28, 28, device=device)

# Pass the data through the model (This calls the forward() method automatically)
logits = model(X)

# The output is "Logits" (Raw scores from -infinity to +infinity).
# We use Softmax to convert them into probabilities (0 to 1).
pred_probab = nn.Softmax(dim=1)(logits)

# Get the predicted class (the index with the highest probability)
y_pred = pred_probab.argmax(1)

print(f"Predicted class: {y_pred.item()}")


# ==========================================
# 4. DEEP DIVE: STEP-BY-STEP LAYER ANALYSIS
# ==========================================
# Let's break down exactly what happens inside the 'forward' method.
print("\n--- Deep Dive: What happens inside the layers? ---")

# Create a mini-batch of 3 images (3, 28, 28)
input_image = torch.rand(3, 28, 28)
print(f"1. Input Shape: {input_image.size()}")

# Step A: Flatten
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(f"2. After Flatten: {flat_image.size()}") 
# Result: [3, 784] -> 3 images, each is a long line of 784 pixels.

# Step B: Linear Layer
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(f"3. After Linear Layer: {hidden1.size()}")
# Result: [3, 20] -> Features are compressed from 784 to 20.

# Step C: ReLU (Activation)
# ReLU turns all negative numbers to 0. It doesn't change the shape.
print(f"   (Before ReLU values): {hidden1[0][:5].detach().numpy()}")
hidden1 = nn.ReLU()(hidden1)
print(f"   (After ReLU values):  {hidden1[0][:5].detach().numpy()}")


# ==========================================
# 5. MODEL PARAMETERS
# ==========================================
# Neural Networks learn by adjusting "Weights" and "Biases".
# Let's see how many parameters our model has.

print("\n--- Model Parameters ---")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n") 

print("Model built and inspected successfully.")