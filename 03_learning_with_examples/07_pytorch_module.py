# Topic: PyTorch: Custom nn Modules
# Source: https://docs.pytorch.org/tutorials/beginner/examples_nn/polynomial_module.html
# Summary:
#   1. Defining a custom model by subclassing `torch.nn.Module`.
#   2. Using `nn.Parameter` to register tensors as model weights.
#   3. Defining the `forward` function for complex operations.
#   4. Reaping the benefits of PyTorch's ecosystem with custom architecture.

import torch
import math

# ==========================================
# 0. SET DEVICE & DATA
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on device: {device}")

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=torch.float)
y = torch.sin(x)

# ==========================================
# 1. DEFINE THE CUSTOM MODULE
# ==========================================
class Polynomial3(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters. We use nn.Parameter to tell PyTorch that these 
        should be learned during training.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn((), device=device))
        self.b = torch.nn.Parameter(torch.randn((), device=device))
        self.c = torch.nn.Parameter(torch.randn((), device=device))
        self.d = torch.nn.Parameter(torch.randn((), device=device))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        """
        Just like any class in Python, you can also define custom methods on PyTorch modules!
        """
        return f'y = {self.a.item():.2f} + {self.b.item():.2f} x + {self.c.item():.2f} x^2 + {self.d.item():.2f} x^3'

# Construct our model by instantiating the class defined above
model = Polynomial3().to(device)

# ==========================================
# 2. DEFINE LOSS FUNCTION & OPTIMIZER
# ==========================================
# We use MSELoss. For this specific custom architecture, SGD is sufficient to converge.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)


print("Starting training with Custom nn.Module...\n")

# ==========================================
# 3. TRAINING LOOP
# ==========================================
for t in range(2000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    
    if t % 100 == 99:
        print(f'Epoch {t+1}: loss = {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# We can call the custom method we created inside the module!
print(f"\nResult: {model.string()}")
print("✅ PyTorch Custom nn.Module script finished.")