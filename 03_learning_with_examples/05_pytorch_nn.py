# Topic: PyTorch: nn
# Source: https://docs.pytorch.org/tutorials/beginner/examples_nn/polynomial_nn.html
# Summary:
#   1. Replacing manual operations with the PyTorch `nn` module.
#   2. Using `nn.Sequential` to build a network of layers.
#   3. Using `nn.Linear` to handle the weights and biases automatically.
#   4. Using `nn.MSELoss` to compute the loss function.

import torch
import math

# ==========================================
# 0. SET DEVICE & DATA
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float
print(f"🚀 Running on device: {device}")

# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# For this example, the output y is a linear function of (x, x^2, x^3), so
# we can consider it as a linear layer neural network. Let's prepare the
# tensor (x, x^2, x^3).
p = torch.tensor([1, 2, 3], device=device)
xx = x.unsqueeze(-1).pow(p)
# xx now has shape (2000, 3). It contains [x, x^2, x^3] for every point.

# ==========================================
# 1. DEFINE THE MODEL USING nn.Sequential
# ==========================================
# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. 
# The Linear Module computes output from input using a linear function: y = xA^T + b
# The Flatten layer flattens the output to match the shape of `y`.
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
).to(device) 

# ==========================================
# 2. DEFINE THE LOSS FUNCTION
# ==========================================
# The nn package also contains definitions of popular loss functions.
# We will use Mean Squared Error (MSE) as our loss function with reduction='sum'.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
print("Starting training with PyTorch nn package...\n")

# ==========================================
# 3. TRAINING LOOP
# ==========================================
for t in range(2000):
    # Forward pass: compute predicted y by passing xx to the model. 
    y_pred = model(xx)

    # Compute and print loss. 
    loss = loss_fn(y_pred, y)
    
    if t % 100 == 99:
        print(f'Epoch {t+1}: loss = {loss.item():.4f}')

    # Zero the gradients before running the backward pass.
    # We no longer need to zero each variable individually!
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. 
    loss.backward()

    # Update the weights using gradient descent. 
    # We iterate over model.parameters() so we don't have to update a, b, c, d manually.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# Access the first layer of `model` to print the results
linear_layer = model[0]

# For a linear layer, its parameters are stored as `weight` and `bias`.
# bias = a, weight[0] = b, weight[1] = c, weight[2] = d
print(f"\nResult: y = {linear_layer.bias.item():.2f} + {linear_layer.weight[:, 0].item():.2f}x + {linear_layer.weight[:, 1].item():.2f}x^2 + {linear_layer.weight[:, 2].item():.2f}x^3")
print("✅ PyTorch nn script finished.")