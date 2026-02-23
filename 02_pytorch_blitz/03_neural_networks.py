# Topic: Neural Networks (Blitz)
# Source: https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# Summary:
#   1. Define a Convolutional Neural Network (CNN).
#   2. Process inputs through the network (Forward pass).
#   3. Compute the loss (MSELoss).
#   4. Backpropagate gradients into the network's parameters.
#   5. Update the weights using an Optimizer (SGD).

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print("\n==========================================")
print("1. DEFINING THE NETWORK (LeNet-5 Architecture)")
print("==========================================")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # 1. Convolutional Layers (Extracting visual features)
        # 1 input image channel (grayscale), 6 output channels (filters), 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        
        # 6 input channels (from conv1), 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # 2. Fully Connected (Linear) Layers (Making the final decision)
        # Why 16 * 5 * 5? Because the 32x32 image shrinks down to 5x5 after convolutions and pooling.
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10) # 10 classes output

    def forward(self, x):
        # Convolution 1 -> ReLU -> Max Pooling (2x2 window)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        
        # Convolution 2 -> ReLU -> Max Pooling (2x2 window)
        # If the size is a square, you can specify it with a single number (2 means 2x2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        # Flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1) 
        
        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Final output layer (No ReLU here, just raw scores/logits)
        x = self.fc3(x)
        return x

net = Net()
print(net)


print("\n==========================================")
print("2. FORWARD PASS (PROCESSING DATA)")
print("==========================================")

# Create a dummy input representing a batch of 1 image, 1 channel (grayscale), 32x32 pixels
input_data = torch.randn(1, 1, 32, 32)
print(f"Input Shape: {input_data.shape}")

# Pass the data through the network
out = net(input_data)
print(f"Output Shape (Logits): {out.shape}")
print(f"Output Values:\n{out}")


print("\n==========================================")
print("3. LOSS CALCULATION")
print("==========================================")

# Create a dummy target (label) for our dummy input
target = torch.randn(10)  
target = target.view(1, -1) # Make it the same shape as output: (1, 10)

# We use Mean Squared Error (MSE) Loss for this example
criterion = nn.MSELoss()

# Calculate how far off our prediction (out) is from the target
loss = criterion(out, target)
print(f"Calculated MSE Loss: {loss.item():.4f}")


print("\n==========================================")
print("4. BACKPROPAGATION & WEIGHT UPDATE")
print("==========================================")

# Zero the gradient buffers of all parameters (CRITICAL STEP!)
# Otherwise, gradients from previous runs would accumulate.
net.zero_grad()     

print(f"conv1.bias.grad BEFORE backward: {net.conv1.bias.grad}")

# Trigger backpropagation
loss.backward()

print(f"conv1.bias.grad AFTER backward: \n{net.conv1.bias.grad}")

# Create an Optimizer (Stochastic Gradient Descent)
# Learning rate is 0.01
optimizer = optim.SGD(net.parameters(), lr=0.01)

# In the training loop, you would do this:
optimizer.zero_grad()   # 1. Zero gradients
out = net(input_data)   # 2. Forward pass
loss = criterion(out, target) # 3. Calculate loss
loss.backward()         # 4. Backward pass
optimizer.step()        # 5. Update weights

print("\n Optimizer step completed. Weights are updated successfully!")
print("Blitz Neural Networks script finished.")