# Topic: A Gentle Introduction to torch.autograd (Blitz)
# Source: https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
# Summary:
#   1. Using a pre-trained model (ResNet18) for a single training step.
#   2. Forward pass, loss calculation, and backward pass.
#   3. Updating weights with an Optimizer.
#   4. Freezing parameters (useful for Fine-Tuning / Transfer Learning).

import torch
from torchvision.models import resnet18, ResNet18_Weights

print("\n==========================================")
print("1. FORWARD PASS WITH RESNET18")
print("==========================================")

# Load a pre-trained ResNet18 model
# We use weights=ResNet18_Weights.DEFAULT to download a model already trained on ImageNet
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Create dummy data: 1 image, 3 color channels (RGB), 64x64 pixels
data = torch.rand(1, 3, 64, 64)

# Create dummy labels: 1 label with shape (1, 1000) because ResNet18 outputs 1000 classes
labels = torch.rand(1, 1000)

# Forward pass: push data through the model to make a prediction
prediction = model(data)
print(f"Prediction shape: {prediction.shape}")
print("Forward pass completed. The computational graph is built.")


print("\n==========================================")
print("2. BACKWARD PASS & OPTIMIZATION")
print("==========================================")

# Calculate a dummy loss and trigger backpropagation
loss = (prediction - labels).sum()
print(f"Calculated Loss: {loss.item():.4f}")

# Autograd calculates the gradients of the loss w.r.t. model parameters
loss.backward()
print("Backward pass completed. Gradients are stored in model parameters (.grad).")

# Load an optimizer (Stochastic Gradient Descent)
# Learning rate (lr) is 1e-2, momentum is 0.9
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# Initiate Gradient Descent: Update the weights based on the calculated gradients
optimizer.step()
print("Optimizer step completed. Weights are updated!")


print("\n==========================================")
print("3. FREEZING PARAMETERS (TRANSFER LEARNING)")
print("==========================================")

# Sometimes we only want to train specific layers (e.g., the last layer)
# We can freeze the rest of the network by setting requires_grad = False

for param in model.parameters():
    param.requires_grad = False # Freeze all existing layers

# Replace the last layer (fc = fully connected) with a new one
# We change it from 1000 classes to 10 classes.
# Note: New layers have requires_grad=True by default!
model.fc = torch.nn.Linear(512, 10)

# Now, only the parameters of model.fc will be updated during training
# The rest of the network acts as a fixed feature extractor.
optimizer_finetune = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

print("✅ Successfully froze base layers and replaced the final classifier.")
print("✅ Only the new 'fc' layer will learn. The rest is frozen.")

print("\n✅ Blitz Autograd script completed successfully.")