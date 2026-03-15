# Topic: PyTorch: optim
# Source: https://docs.pytorch.org/tutorials/beginner/examples_nn/polynomial_optim.html
# Summary:
#   1. Using the `torch.optim` package to update the weights automatically.
#   2. Replacing manual gradient descent with the RMSprop optimizer.
#   3. Simplifying the training loop drastically.

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

# Prepare the polynomial tensor (x, x^2, x^3).
p = torch.tensor([1, 2, 3], device=device)
xx = x.unsqueeze(-1).pow(p)

# ==========================================
# 1. DEFINE THE MODEL & LOSS FUNCTION
# ==========================================
# We use nn.Sequential to define our model.
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
).to(device)

loss_fn = torch.nn.MSELoss(reduction='sum')

# ==========================================
# 2. DEFINE THE OPTIMIZER (The Magic Engine)
# ==========================================
# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use RMSprop; the optim package contains many other
# optimization algorithms (like SGD, Adam, AdaGrad, etc.).
# Note: Learning rate is now 1e-3, much larger than the 1e-6 we used before!
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

print("Starting training with PyTorch optim package...\n")

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

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This replaces model.zero_grad() or param.grad = None.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters.
    # THIS REPLACES THE ENTIRE `with torch.no_grad():` BLOCK!
    optimizer.step()

# Access the first layer to print the final weights
linear_layer = model[0]
print(f"\nResult: y = {linear_layer.bias.item():.2f} + {linear_layer.weight[:, 0].item():.2f}x + {linear_layer.weight[:, 1].item():.2f}x^2 + {linear_layer.weight[:, 2].item():.2f}x^3")
print("✅ PyTorch optim script finished.")