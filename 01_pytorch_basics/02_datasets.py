# Topic: Datasets and DataLoaders (The Full Guide)
# Source: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# Summary: 
#   1. Loading pre-defined datasets (FashionMNIST).
#   2. Visualizing data.
#   3. Creating a Custom Dataset class (Blueprint).
#   4. Preparing data for training with DataLoaders.
#   5. Iterating through the DataLoader.

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import decode_image

# ==========================================
# 1. LOADING A DATASET
# ==========================================
print("\n--- 1. Loading Datasets ---")
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
print("Datasets loaded successfully.")


# ==========================================
# 2. ITERATING AND VISUALIZING THE DATASET
# ==========================================
print("\n--- 2. Visualizing Data (Random Samples) ---")
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

print("Displaying random samples... (Close window to continue)")
plt.show()


# ==========================================
# 3. CREATING A CUSTOM DATASET (BLUEPRINT)
# ==========================================
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

print("\n--- 3. Custom Dataset Class defined (Skipped execution) ---")


# ==========================================
# 4. PREPARING DATA FOR TRAINING WITH DATALOADERS
# ==========================================
print("\n--- 4. Creating DataLoaders ---")
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
print("DataLoaders ready.")


# ==========================================
# 5. ITERATE THROUGH THE DATALOADER
# ==========================================
# This part corresponds to the explicit visualization in the tutorial.
# We pull ONE batch to inspect its dimensions and data.

print("\n--- 5. Iterate through the DataLoader ---")

# Method A: Using next(iter()) to get a single batch manually
train_features, train_labels = next(iter(train_dataloader))

print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# Explanation:
# [64, 1, 28, 28] -> We have 64 images, 1 Color Channel (Gray), 28x28 pixels.

# Display the first image in this batch
img = train_features[0].squeeze()
label = train_labels[0]
plt.figure(figsize=(4, 4))
plt.imshow(img, cmap="gray")
plt.title(f"Batch Sample Label: {label} ({labels_map[label.item()]})")
print(f"Label: {label} -> {labels_map[label.item()]}")
print("Displaying batch sample... (Close window to finish)")
plt.show()

# Method B: How the training loop actually iterates (Example)
print("\n[Demo] How the loop works in training:")
for batch_idx, (X, y) in enumerate(train_dataloader):
    print(f" -> Batch {batch_idx+1} loaded. X shape: {X.shape}, y shape: {y.shape}")
    if batch_idx == 2: # Stop after 3 batches just for demo
        print(" -> ... (Stopping demo loop here)")
        break

print("\nAll steps completed successfully.")