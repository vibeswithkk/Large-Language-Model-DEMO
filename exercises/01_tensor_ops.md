# Exercise 1: Tensor Operations

## Objective
Practice fundamental tensor operations in PyTorch to build a strong foundation for working with transformer models.

## Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Jupyter Notebook (optional but recommended)

## Instructions

### Part 1: Basic Tensor Creation and Properties

1. Create tensors of different dimensions:
   - Create a scalar tensor with value 5.0
   - Create a 1D tensor with values [1.0, 2.0, 3.0, 4.0]
   - Create a 2D tensor (matrix) with values [[1.0, 2.0], [3.0, 4.0]]
   - Create a 3D tensor with random values of shape (2, 3, 4)

2. Examine tensor properties:
   - Print the shape, data type, and device of each tensor
   - Move one of your tensors to GPU if available
   - Convert a float tensor to integer type

### Part 2: Tensor Operations

1. Perform basic arithmetic operations:
   - Create two 2D tensors of the same shape
   - Add, subtract, multiply, and divide them element-wise
   - Perform matrix multiplication using `torch.mm()` and `@` operator

2. Practice broadcasting:
   - Create a 2D tensor of shape (3, 4) and a 1D tensor of shape (4,)
   - Add them together and observe the broadcasting behavior
   - Try adding a 2D tensor of shape (3, 1) to a 2D tensor of shape (1, 4)

3. Indexing and slicing:
   - Create a 3D tensor of shape (2, 3, 4) with sequential values
   - Extract the first row of the first matrix
   - Extract the last column of the second matrix
   - Extract a sub-tensor using advanced indexing

### Part 3: Advanced Operations

1. Reshaping operations:
   - Create a tensor of shape (6, 4)
   - Reshape it to (3, 8) and then to (2, 2, 6)
   - Use `view()`, `reshape()`, and `permute()` to manipulate dimensions

2. Reduction operations:
   - Create a 2D tensor with random values
   - Calculate the sum, mean, max, and min along different dimensions
   - Use `argmax()` and `argmin()` to find indices of extreme values

3. Statistical operations:
   - Generate a tensor with normal distribution values
   - Calculate its standard deviation and variance
   - Normalize the tensor to have zero mean and unit variance

## Challenge Problems

1. **Memory Efficiency**: Create a large tensor (1000x1000) and perform operations without running out of memory. Use techniques like in-place operations and proper memory management.

2. **GPU Optimization**: If you have access to a GPU, compare the performance of tensor operations on CPU vs GPU. Measure execution time for large matrix multiplications.

3. **Broadcasting Puzzle**: Create three tensors with shapes (5, 1, 3), (1, 4, 3), and (5, 4, 1). Determine what shape the result will have when you add all three together.

## Submission

Create a Python script or Jupyter notebook that demonstrates all the operations above. Include comments explaining what each operation does and why it's useful in the context of neural networks.

## Resources

- [PyTorch Tensor Documentation](https://pytorch.org/docs/stable/tensors.html)
- [PyTorch Broadcasting Semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)
- [Tensor Operations Tutorial](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)