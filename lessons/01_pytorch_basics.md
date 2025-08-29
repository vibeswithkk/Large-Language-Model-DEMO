# Lesson 1: PyTorch Basics for Deep Learning

## Overview
This lesson introduces the fundamental concepts of PyTorch, the deep learning framework we'll use throughout this course. We'll cover tensors, automatic differentiation, and basic neural network components.

## Learning Objectives
By the end of this lesson, you should be able to:
- Understand what tensors are and how they differ from NumPy arrays
- Create and manipulate tensors in PyTorch
- Use PyTorch's automatic differentiation system
- Build simple neural network components
- Move computations between CPU and GPU

## 1. Introduction to Tensors

### What are Tensors?
Tensors are the fundamental data structure in PyTorch, similar to NumPy arrays but with additional capabilities for deep learning:
- Multi-dimensional arrays (0D scalars, 1D vectors, 2D matrices, etc.)
- Support for GPU acceleration
- Automatic differentiation capabilities
- Optimized for deep learning operations

### Creating Tensors
```python
import torch
import numpy as np

# Creating tensors from Python data types
scalar = torch.tensor(5.0)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Creating tensors with specific shapes
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
random = torch.rand(2, 3)
normal = torch.randn(2, 3)

# Creating tensors from NumPy arrays
numpy_array = np.array([1, 2, 3, 4])
tensor_from_numpy = torch.from_numpy(numpy_array)

# Specifying data types
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
long_tensor = torch.tensor([1, 2, 3], dtype=torch.long)
```

### Tensor Properties
```python
tensor = torch.randn(3, 4, 5)

print(f"Shape: {tensor.shape}")
print(f"Data type: {tensor.dtype}")
print(f"Device: {tensor.device}")
print(f"Number of elements: {tensor.numel()}")
print(f"Number of dimensions: {tensor.dim()}")
```

## 2. Tensor Operations

### Basic Operations
```python
# Arithmetic operations
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

addition = a + b
subtraction = a - b
multiplication = a * b
division = a / b

# Matrix operations
matrix_a = torch.randn(3, 4)
matrix_b = torch.randn(4, 2)
matrix_product = torch.mm(matrix_a, matrix_b)  # or matrix_a @ matrix_b

# Reduction operations
tensor_2d = torch.randn(3, 4)
sum_all = tensor_2d.sum()
sum_dim0 = tensor_2d.sum(dim=0)  # Sum along rows
sum_dim1 = tensor_2d.sum(dim=1)  # Sum along columns
```

### Indexing and Slicing
```python
tensor_3d = torch.randn(2, 3, 4)

# Basic indexing
first_element = tensor_3d[0, 0, 0]
first_matrix = tensor_3d[0]

# Slicing
subset = tensor_3d[:, 1:3, :2]  # Rows 1-2, first 2 columns of all matrices

# Advanced indexing
indices = torch.tensor([0, 2])
selected_rows = tensor_3d[indices]

# Boolean indexing
mask = tensor_3d > 0
positive_elements = tensor_3d[mask]
```

### Reshaping Operations
```python
tensor = torch.randn(6, 4)

# Reshape to different dimensions
reshaped1 = tensor.view(3, 8)
reshaped2 = tensor.view(2, 2, 6)
reshaped3 = tensor.reshape(12, 2)  # reshape() can handle non-contiguous tensors

# Transpose and permute
transposed = tensor.t()  # For 2D tensors
permuted = tensor_3d.permute(2, 0, 1)  # Change dimension order
```

## 3. Automatic Differentiation

### Gradient Computation
```python
# Enable gradient tracking
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Define a computation
z = x**2 + 2*y + 3*x*y

# Compute gradients
z.backward()

# Access gradients
print(f"Gradient of x: {x.grad}")
print(f"Gradient of y: {y.grad}")
```

### Computational Graph
```python
# For more complex computations
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

# Build computation graph
a = x * y
b = a + x
c = b**2
loss = c.mean()

# Backpropagate
loss.backward()

print(f"Gradient of x: {x.grad}")
print(f"Gradient of y: {y.grad}")
```

### Gradient Management
```python
# Disable gradient computation (for inference)
with torch.no_grad():
    result = x**2 + y**2

# Temporarily disable gradient tracking
with torch.set_grad_enabled(False):
    result = x * y

# Zero gradients (important for training loops)
optimizer = torch.optim.SGD([x, y], lr=0.01)
optimizer.zero_grad()  # Clear gradients before backward pass
```

## 4. Neural Network Basics

### Using torch.nn
```python
import torch.nn as nn

# Linear layer
linear = nn.Linear(in_features=10, out_features=5)

# Activation functions
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

# Loss functions
mse_loss = nn.MSELoss()
cross_entropy = nn.CrossEntropyLoss()

# Sequential models
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.Softmax(dim=1)
)
```

### Custom Modules
```python
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate and use the model
model = SimpleMLP(784, 128, 10)
input_data = torch.randn(32, 784)
output = model(input_data)
```

## 5. GPU Acceleration

### Device Management
```python
# Check for GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Move tensors to GPU
tensor = torch.randn(1000, 1000)
tensor_gpu = tensor.to(device)

# Move models to GPU
model = SimpleMLP(784, 128, 10)
model_gpu = model.to(device)

# Ensure data and model are on the same device
input_data = torch.randn(32, 784).to(device)
output = model_gpu(input_data)
```

### Memory Management
```python
# Check GPU memory usage
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# Clear GPU cache
torch.cuda.empty_cache()
```

## 6. Practical Examples

### Linear Regression with PyTorch
```python
# Generate synthetic data
X = torch.randn(100, 1)
y = 3 * X + 2 + torch.randn(100, 1) * 0.1

# Define model
model = nn.Linear(1, 1)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print(f"Learned weight: {model.weight.item():.4f}")
print(f"Learned bias: {model.bias.item():.4f}")
```

## Summary

In this lesson, we covered the fundamental concepts of PyTorch:
- Tensors as the core data structure
- Tensor operations and manipulations
- Automatic differentiation for gradient computation
- Basic neural network components using torch.nn
- GPU acceleration for faster computations

These concepts form the foundation for building and training deep learning models, including transformers, which we'll explore in subsequent lessons.

## Next Steps

In the next lesson, we'll dive into the Transformer architecture, understanding its components and how they work together to process sequential data.