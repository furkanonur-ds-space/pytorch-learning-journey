# Topic: Warm-up: numpy
# Source: https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html
# Summary:
#   1. Implementing a 3rd order polynomial to fit y=sin(x).
#   2. Using pure NumPy (No PyTorch, No Autograd).
#   3. Manually computing gradients (Backpropagation via Math).
#   4. Manually updating weights.

import numpy as np
import math

# ==========================================
# 1. CREATE RANDOM DATA
# ==========================================
# Create random input and output data
# x: range from -pi to +pi
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Randomly initialize weights
# We want to fit the function: y = a + b*x + c*x^2 + d*x^3
# So we need 4 parameters: a, b, c, d
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
print(f"Initial weights: a={a:.2f}, b={b:.2f}, c={c:.2f}, d={d:.2f}")
print("Starting training with pure NumPy...\n")

# ==========================================
# 2. TRAINING LOOP (MANUAL BACKPROP)
# ==========================================
for t in range(2000):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    # Loss = Sum of squared errors
    loss = np.square(y_pred - y).sum()
    
    if t % 100 == 99:
        print(f'Epoch {t+1}: loss = {loss:.4f}')

    # Backprop to compute gradients of a, b, c, d with respect to loss
    # WE HAVE TO DO THE MATH MANUALLY HERE! 
    # d(loss)/d(y_pred) = 2 * (y_pred - y)
    grad_y_pred = 2.0 * (y_pred - y)
    
    # Chain rule:
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    # w = w - learning_rate * gradient
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f"\nResult: y = {a:.2f} + {b:.2f}x + {c:.2f}x^2 + {d:.2f}x^3")
print("✅ Numpy Warm-up finished.")