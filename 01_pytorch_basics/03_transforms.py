# Topic: Transforms (Preparing Data for Training)
# Source: https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
# Summary: 
#   1. Using `ToTensor` to convert images to normalized tensors (0-1 range).
#   2. Using `Lambda` transforms to convert labels to One-Hot Encoding.

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# ==========================================
# 1. LOADING DATA WITH TRANSFORMS
# ==========================================
# We load the dataset again, but this time we focus on the 'transform' parameters.
# transform: Modifies the features (images).
# target_transform: Modifies the labels (integers).

print("\n--- 1. Loading Dataset with Transforms ---")

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    
    # A) ToTensor(): Converts a PIL Image or NumPy ndarray into a FloatTensor.
    # It scales the image's pixel intensity values in the range [0., 1.]
    transform=ToTensor(),

    # B) Lambda Transforms: Apply any user-defined lambda function.
    # Here, we turn the integer label (e.g., 9) into a One-Hot Encoded tensor.
    # Example: If label is 2 (Pullover) and we have 10 classes:
    # Result: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

print("Dataset loaded with transformations.")


# ==========================================
# 2. INSPECTING THE TRANSFORMATIONS
# ==========================================
# Let's grab one sample to see what these transforms actually did to the data.

print("\n--- 2. Inspecting the Result ---")

# Get the first sample (Image and Label)
image, label = ds[0]

# CHECK 1: The Image (Feature)
print(f"Original Image Shape: {image.shape}")
print(f"Data Type: {image.dtype}")
print(f"Pixel Values Range: [{image.min()}, {image.max()}]") 
# Notice the range is now 0.0 to 1.0 (Thanks to ToTensor)

print("-" * 30)

# CHECK 2: The Label (Target)
# The raw label for the first item is usually 9 (Ankle Boot).
# Let's see what the Lambda transform converted it into.
print(f"Transformed Label (One-Hot): \n{label}")
print(f"Label Shape: {label.shape}")
print(f"Label Type: {label.dtype}")

# Explanation of Scatter:
# torch.zeros(10) created: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# scatter_ put a '1' at index 9.
# Result: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

print("\nScript completed. Data is ready for the Neural Network.")