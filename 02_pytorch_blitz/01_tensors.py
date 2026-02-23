# Topic: What is PyTorch? (Tensors in 60 Minute Blitz)
# Source: https://docs.pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
# Summary: 
#   1. Tensor initialization variations (empty, zeros, rand).
#   2. Tensor operations (in-place additions).
#   3. Reshaping tensors using .view() (Crucial for Deep Learning!).
#   4. NumPy Bridge (Sharing memory).

import torch
import numpy as np

print("\n==========================================")
print("1. TENSOR INITIALIZATION")
print("==========================================")

# Construct a 5x3 matrix, uninitialized
# Note: Whatever values were in the allocated memory will appear as the initial values.
x_empty = torch.empty(5, 3)
print(f"Empty Tensor:\n{x_empty}\n")

# Construct a randomly initialized matrix
x_rand = torch.rand(5, 3)
print(f"Random Tensor:\n{x_rand}\n")

# Construct a matrix filled zeros and of dtype long
x_zeros = torch.zeros(5, 3, dtype=torch.long)
print(f"Zeros Tensor (dtype=long):\n{x_zeros}\n")

# Construct a tensor directly from data
x_data = torch.tensor([5.5, 3])
print(f"Tensor from data:\n{x_data}\n")

# Create a tensor based on an existing tensor
# These methods will reuse properties of the input tensor (e.g. dtype) unless new values are provided
x_ones = x_empty.new_ones(5, 3, dtype=torch.double)
print(f"New Ones Tensor (based on empty):\n{x_ones}\n")

# Override dtype but keep the same size
x_randn = torch.randn_like(x_ones, dtype=torch.float)
print(f"Random Normal Tensor (same shape as ones):\n{x_randn}\n")
print(f"Size of x_randn: {x_randn.size()}") # .size() is similar to .shape


print("\n==========================================")
print("2. TENSOR OPERATIONS")
print("==========================================")

y = torch.rand(5, 3)
print(f"Tensor X:\n{x_randn}")
print(f"Tensor Y:\n{y}\n")

# Addition: Syntax 1
print(f"Addition (X + Y):\n{x_randn + y}\n")

# Addition: Syntax 2
print(f"Addition (torch.add):\n{torch.add(x_randn, y)}\n")

# Addition: In-place
# Any operation that mutates a tensor in-place is post-fixed with an '_'.
# For example: x.copy_(y), x.t_(), will change x.
y.add_(x_randn)
print(f"In-place Addition (y.add_(x)) -> Modifies Y directly:\n{y}\n")


print("\n==========================================")
print("3. RESHAPING TENSORS (Very Important!)")
print("==========================================")
# If you want to resize/reshape tensor, you can use torch.view


x = torch.randn(4, 4)

# Reshape to a 1D vector of 16 elements
y = x.view(16)

# The size -1 is inferred from other dimensions
# Here, we want 8 columns, so PyTorch figures out it needs 2 rows (16 / 8 = 2)
z = x.view(-1, 8) 

print(f"Original shape (x): {x.size()}")
print(f"Reshaped to 1D (y): {y.size()}")
print(f"Reshaped with -1 (z): {z.size()}\n")


print("\n==========================================")
print("4. ONE-ELEMENT TENSORS")
print("==========================================")

x_single = torch.randn(1)
print(f"Single element tensor: {x_single}")
print(f"Extracted Python number: {x_single.item()}")

print("\n✅ Blitz Tensors script completed successfully.")