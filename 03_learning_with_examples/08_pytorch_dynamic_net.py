# Topic: PyTorch: Control Flow + Weight Sharing
# Source: https://docs.pytorch.org/tutorials/beginner/examples_nn/dynamic_net.html
# Summary:
#   1. Creating a fully dynamic neural network.
#   2. Using standard Python control flow (for loops, random) inside the forward pass.
#   3. Sharing weights across multiple layers/operations (Weight Sharing).
#   4. Demonstrating PyTorch's "define-by-run" dynamic graph creation.

import random
import torch
import math

# ==========================================
# 0. SET DEVICE & DATA
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on device: {device}")

# Create input and output tensors
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=torch.float)
y = torch.sin(x)

# ==========================================
# 1. DEFINE THE DYNAMIC MODULE
# ==========================================
class DynamicNet(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn((), device=device))
        self.b = torch.nn.Parameter(torch.randn((), device=device))
        self.c = torch.nn.Parameter(torch.randn((), device=device))
        self.d = torch.nn.Parameter(torch.randn((), device=device))
        self.e = torch.nn.Parameter(torch.randn((), device=device))

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 4, 5
        and reuse the e parameter to compute the contribution of these orders.
        
        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements.
        """
        # Base polynomial: y = a + b*x + c*x^2 + d*x^3
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        
        # DYNAMIC CONTROL FLOW & WEIGHT SHARING:
        # random.randint(4, 6) returns 4, 5 or 6.
        # If 4: loop doesn't run. (Order 3)
        # If 5: runs once (exp=4). Adds e * x^4. (Order 4)
        # If 6: runs twice (exp=4, exp=5). Adds e * x^4 + e * x^5. (Order 5)
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x ** exp
            
        return y

    def string(self):
        """
        Custom method to print the model's weights.
        """
        return f'y = {self.a.item():.2f} + {self.b.item():.2f} x + {self.c.item():.2f} x^2 + {self.d.item():.2f} x^3 + {self.e.item():.2f} x^4 ? + {self.e.item():.2f} x^5 ?'

# Instantiate the dynamic model
model = DynamicNet().to(device)

# ==========================================
# 2. DEFINE LOSS FUNCTION & OPTIMIZER
# ==========================================
criterion = torch.nn.MSELoss(reduction='sum')

# CRITICAL: We use a VERY small learning rate (1e-8) and momentum.
# Why? Because x^4 and x^5 create MASSIVE numbers. Gradients will explode 
# easily if we don't brake hard!
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)

print("Starting training with Dynamic Network...\n")

# ==========================================
# 3. TRAINING LOOP
# ==========================================
for t in range(2000):
    # Forward pass: Compute predicted y
    y_pred = model(x)

    # Compute loss
    loss = criterion(y_pred, y)
    
    if t % 100 == 99:
        print(f'Epoch {t+1}: loss = {loss.item():.4f}')

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"\nResult: {model.string()}")
print("✅ PyTorch Dynamic Network script finished.")