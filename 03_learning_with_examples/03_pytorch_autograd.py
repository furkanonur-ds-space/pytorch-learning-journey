# Topic: PyTorch: Tensors and Autograd
# Source: https://docs.pytorch.org/tutorials/beginner/examples_autograd/polynomial_autograd.html
# Summary:
#   1. Using PyTorch Tensors with requires_grad=True to track operations.
#   2. Automatically computing gradients using loss.backward().
#   3. Updating weights inside a torch.no_grad() block.
#   4. Zeroing out gradients after each update.

import torch
import math

# ==========================================
# 0. SET DEVICE (CPU OR GPU)
# ==========================================
dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on device: {device}")

# ==========================================
# 1. CREATE RANDOM DATA AND WEIGHTS
# ==========================================
# Create input and output data. 
# By default, requires_grad=False, which is what we want for data.
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Create random weights for y = a + b*x + c*x^2 + d*x^3
# CRITICAL: We set requires_grad=True to tell PyTorch to track all operations 
# that happen to these tensors so it can compute gradients later.
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
print(f"Initial weights: a={a.item():.2f}, b={b.item():.2f}, c={c.item():.2f}, d={d.item():.2f}")
print("Starting training with PyTorch Autograd...\n")

# ==========================================
# 2. TRAINING LOOP (AUTOMATIC BACKPROP)
# ==========================================
for t in range(2000):
    # Forward pass: compute predicted y
    # Because a, b, c, d have requires_grad=True, PyTorch builds a 
    # "Computational Graph" in the background as we do these math operations.
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    
    if t % 100 == 99:
        print(f'Epoch {t+1}: loss = {loss.item():.4f}')

    # BACKWARD PASS: The Magic Happens Here!
    # This computes the gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad, c.grad and d.grad will be Tensors holding the gradient
    # of the loss with respect to a, b, c, d respectively.
    loss.backward()

    # Update weights using gradient descent
    # We wrap this in torch.no_grad() because weights have requires_grad=True, 
    # but we don't need to track this specific update operation in autograd.
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Manually zero the gradients after updating weights!
        # If we don't do this, gradients will accumulate (add up) across iterations.
        a.grad.zero_()
        b.grad.zero_()
        c.grad.zero_()
        d.grad.zero_()

print(f"\nResult: y = {a.item():.2f} + {b.item():.2f}x + {c.item():.2f}x^2 + {d.item():.2f}x^3")
print("✅ PyTorch Autograd script finished.")