# Topic: Automatic Differentiation with torch.autograd
# Source: https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
# Summary: 
#   1. Understanding how PyTorch tracks operations (Computational Graph).
#   2. Calculating gradients (Backpropagation) automatically.
#   3. Disabling gradient tracking (for inference/testing).

import torch

print(f"PyTorch Version: {torch.__version__}")

# ==========================================
# 1. SETUP: A SIMPLE ONE-LAYER NETWORK
# ==========================================
# Let's simulate a very simple neural network training step.
# We have input (x), target (y), and parameters (w, b).

# Input tensor (Fixed data, does not need gradients)
x = torch.ones(5)  # input: [1, 1, 1, 1, 1]
y = torch.zeros(3) # expected output: [0, 0, 0]

# Parameters (Weights and Biases)
# These are what the model needs to LEARN.
# We MUST set `requires_grad=True` so PyTorch knows to track changes here.
w = torch.randn(5, 3, requires_grad=True) # Weight matrix
b = torch.randn(3, requires_grad=True)    # Bias vector

print("\n--- 1. Tensors Created ---")
print(f"w: {w.shape} (requires_grad={w.requires_grad})")
print(f"b: {b.shape} (requires_grad={b.requires_grad})")


# ==========================================
# 2. FORWARD PASS & COMPUTATIONAL GRAPH
# ==========================================
# We connect the tensors mathematically.
# PyTorch builds a "Computational Graph" (DAG) in the background.

# Step A: Linear transformation -> z = x * w + b
z = torch.matmul(x, w) + b

# Step B: Loss calculation (Binary Cross Entropy)
# This compares our output (z) with the target (y) to measure error.
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print("\n--- 2. Forward Pass ---")
print(f"Loss value: {loss.item()}")

# Inspecting the Gradient Function
# Since 'z' and 'loss' were created by operations on tensors with requires_grad=True,
# they have a `grad_fn` attribute (The function that created them).
print(f"Gradient Function for z: {z.grad_fn}")       # AddBackward (from + b)
print(f"Gradient Function for loss: {loss.grad_fn}") # BinaryCrossEntropyWithLogitsBackward


# ==========================================
# 3. BACKWARD PASS (THE MAGIC)
# ==========================================
# Now we calculate the gradients.
# This computes d(loss)/dw and d(loss)/db.
# In simple terms: "How much did 'w' and 'b' contribute to the error?"

loss.backward()

print("\n--- 3. Backward Pass (Gradients Calculated) ---")
# The gradients are stored in the `.grad` attribute of the parameters.
print(f"Gradient for b (d_loss/d_b): \n{b.grad}")
print(f"Gradient for w (d_loss/d_w): \n{w.grad}")

# NOTE: We can only call .backward() once per graph unless input retain_graph=True.


# ==========================================
# 4. DISABLING GRADIENT TRACKING
# ==========================================
# Sometimes we don't want to calculate gradients (e.g., during testing/inference).
# Tracking gradients consumes memory.

print("\n--- 4. Disabling Gradient Tracking ---")

# Method A: torch.no_grad() Context Manager (Most common)
z = torch.matmul(x, w) + b
print(f"Does z require grad? {z.requires_grad}") # True

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(f"Does z require grad (inside no_grad)? {z.requires_grad}") # False

# Method B: .detach()
# Creates a new tensor that shares content but is detached from the graph.
z = torch.matmul(x, w) + b
z_det = z.detach()
print(f"Original z requires_grad: {z.requires_grad}") # True
print(f"Detached z requires_grad: {z_det.requires_grad}") # False

print("\nAutograd tutorial completed.")