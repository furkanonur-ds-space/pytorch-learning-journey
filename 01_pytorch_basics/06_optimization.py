# Topic: Optimization Loop (Training the Model)
# Source: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
# Summary: 
#   1. Setting Hyperparameters (Learning Rate, Batch Size, Epochs).
#   2. Defining Loss Function (CrossEntropy) and Optimizer (SGD).
#   3. Implementing the Training Loop (Forward -> Backward -> Update).
#   4. Implementing the Test Loop (Evaluation).

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# ==========================================
# 0. SETUP (DATA & MODEL)
# ==========================================
# We need to re-create the pieces we built in previous tutorials.

# Device configuration
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 1. Load Data
training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

# 2. Create DataLoaders
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 3. Define Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)


# ==========================================
# 1. HYPERPARAMETERS
# ==========================================
# These are the "knobs" we turn to tune the training process.

learning_rate = 1e-3  # How much to update models parameters at each batch.
batch_size = 64       # Number of data samples propagated through the network before updating parameters.
epochs = 5            # Number of times to iterate over the dataset.

print(f"\n--- Hyperparameters ---")
print(f"Learning Rate: {learning_rate}")
print(f"Batch Size: {batch_size}")
print(f"Epochs: {epochs}")


# ==========================================
# 2. LOSS FUNCTION & OPTIMIZER
# ==========================================
# Loss Function: Measures how wrong the model is.
# Optimizer: Adjusts the model parameters to reduce the error.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print("\n--- Setup Complete. Starting Training... ---")


# ==========================================
# 3. TRAINING LOOP (IMPLEMENTATION)
# ==========================================
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        # Move data to GPU
        X, y = X.to(device), y.to(device)
        
        # 1. Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # 2. Backpropagation
        loss.backward()       # Calculate gradients
        optimizer.step()      # Update weights (Adjust parameters)
        optimizer.zero_grad() # Reset gradients for the next batch

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# ==========================================
# 4. TEST LOOP (IMPLEMENTATION)
# ==========================================
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed 
    # during test mode and also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True.
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# ==========================================
# 5. EXECUTION
# ==========================================
# We loop through the epochs. Each epoch consists of a training loop and a validation loop.

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")