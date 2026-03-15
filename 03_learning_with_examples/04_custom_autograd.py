# Topic: PyTorch: Defining New autograd Functions
# Source: https://docs.pytorch.org/tutorials/beginner/examples_autograd/polynomial_custom_function.html
# Summary:
#   1. Creating a custom autograd function by subclassing torch.autograd.Function.
#   2. Implementing the forward pass and saving context for backward.
#   3. Implementing the backward pass (manual derivative calculation).
#   4. Using the custom function in the training loop.

import torch
import math

# ==========================================
# 1. DEFINE A CUSTOM AUTOGRAD FUNCTION
# ==========================================
class LegendrePolynomial3(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation.
        
        Math: P3(x) = 0.5 * (5x^3 - 3x)
        """
        # Save the input tensor for use in the backward pass
        ctx.save_for_backward(input)
        
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        
        Math Derivative: P3'(x) = 1.5 * (5x^2 - 1)
        """
        # Retrieve the input tensor saved in the forward pass
        input, = ctx.saved_tensors
        
        # Chain Rule: grad_output * local_derivative
        return grad_output * 1.5 * (5 * input ** 2 - 1)


# ==========================================
# 0. SET DEVICE & DATA
# ==========================================
dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on device: {device}")

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# We initialize weights not too far from the correct result to ensure convergence.
# The formula is: y = a + b * P3(c + d * x)
a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

learning_rate = 5e-6
print(f"Initial weights: a={a.item():.2f}, b={b.item():.2f}, c={c.item():.2f}, d={d.item():.2f}")
print("Starting training with Custom Autograd Function...\n")

# ==========================================
# 2. TRAINING LOOP
# ==========================================
# Alias for our custom function
P3 = LegendrePolynomial3.apply

for t in range(2000):
    # Forward pass: compute predicted y using our custom autograd operation.
    y_pred = a + b * P3(c + d * x)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    
    if t % 100 == 99:
        print(f'Epoch {t+1}: loss = {loss.item():.4f}')

    # Use autograd to compute the backward pass.
    # PyTorch will call our custom backward() function!
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f"\nResult: y = {a.item():.2f} + {b.item():.2f} * P3({c.item():.2f} + {d.item():.2f}x)")
print("✅ Custom Autograd Function script finished.")