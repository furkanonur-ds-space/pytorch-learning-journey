# Topic: PyTorch: Tensors
# Source: https://docs.pytorch.org/tutorials/beginner/examples_tensor/polynomial_tensor.html
# Summary:
#   1. Implementing the same 3rd order polynomial to fit y=sin(x).
#   2. Replacing NumPy arrays with PyTorch Tensors.
#   3. Utilizing the GPU for faster computations.
#   4. Still manually computing gradients (No Autograd yet).

import torch
import math

# ==========================================
# 0. SET DEVICE (CPU OR GPU)
# ==========================================
dtype = torch.float
# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on device: {device}")

# ==========================================
# 1. CREATE RANDOM DATA ON DEVICE
# ==========================================
# Create random input and output data
# We must specify the dtype and device when creating the tensors
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
# y = a + b*x + c*x^2 + d*x^3
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
print(f"Initial weights: a={a.item():.2f}, b={b.item():.2f}, c={c.item():.2f}, d={d.item():.2f}")
print("Starting training with PyTorch Tensors (Manual Backprop)...\n")

# ==========================================
# 2. TRAINING LOOP
# ==========================================
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    # loss is a Tensor of shape (), so we use .item() to get the Python number
    loss = (y_pred - y).pow(2).sum().item()
    
    if t % 100 == 99:
        print(f'Epoch {t+1}: loss = {loss:.4f}')

    # Backprop to compute gradients of a, b, c, d with respect to loss
    # STILL DOING THIS MANUALLY!
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f"\nResult: y = {a.item():.2f} + {b.item():.2f}x + {c.item():.2f}x^2 + {d.item():.2f}x^3")
print("✅ PyTorch Tensors (Manual) script finished.")