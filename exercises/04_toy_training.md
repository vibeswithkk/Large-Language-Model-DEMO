# Exercise 4: Toy Model Training

## Objective
Train a small transformer model on a toy dataset to understand the training process and optimization techniques.

## Prerequisites
- Completion of Exercises 1-3
- Understanding of transformer architecture
- Knowledge of PyTorch training loops

## Instructions

### Part 1: Dataset Preparation

1. Create a synthetic dataset:
   - Generate text data with simple patterns (e.g., "The [adjective] [noun] [verb] [adverb]")
   - Create vocabulary with ~1000 tokens
   - Generate 10,000 training examples
   - Split into train/validation sets (90/10)

2. Data preprocessing:
   - Tokenize the text data using your tokenizer from Exercise 2
   - Convert to PyTorch tensors
   - Create attention masks for padded sequences
   - Implement a PyTorch Dataset class

3. Data loading:
   - Create DataLoader with appropriate batch size
   - Implement shuffling and batching
   - Handle variable sequence lengths with padding

### Part 2: Model Configuration

1. Configure a small transformer model:
   - Hidden size: 256
   - Number of attention heads: 4
   - Number of layers: 4
   - Intermediate size: 512
   - Vocabulary size: 1000
   - Maximum sequence length: 128

2. Initialize model weights:
   - Implement proper weight initialization
   - Use Xavier/He initialization for linear layers
   - Initialize embedding layers appropriately

3. Device placement:
   - Move model to GPU if available
   - Ensure data is moved to the same device

### Part 3: Training Loop

1. Loss function and optimizer:
   - Use cross-entropy loss with ignore_index for padding
   - Implement Adam optimizer with weight decay
   - Add learning rate scheduling (cosine annealing or linear decay)

2. Training implementation:
   - Implement training loop with epochs
   - Calculate and log training loss
   - Implement gradient clipping
   - Handle gradient accumulation for larger effective batch sizes

3. Validation:
   - Implement validation loop
   - Calculate validation loss
   - Early stopping based on validation performance
   - Save best model checkpoints

### Part 4: Monitoring and Evaluation

1. Logging:
   - Log training and validation metrics
   - Track learning rate, loss, and perplexity
   - Implement TensorBoard logging or simple CSV logging

2. Evaluation metrics:
   - Calculate perplexity on validation set
   - Implement text generation for qualitative evaluation
   - Compare generated text with expected patterns

3. Model analysis:
   - Visualize attention weights
   - Analyze learned embeddings
   - Check for overfitting or underfitting

### Part 5: Experimentation

1. Hyperparameter tuning:
   - Experiment with different learning rates
   - Try different batch sizes
   - Test various optimizer configurations

2. Regularization techniques:
   - Implement dropout and test different rates
   - Try label smoothing
   - Experiment with gradient clipping values

3. Training strategies:
   - Implement learning rate warmup
   - Test different learning rate schedules
   - Try different initialization methods

## Challenge Problems

1. **Curriculum Learning**: Implement a curriculum learning approach:
   - Start with shorter sequences and gradually increase length
   - Begin with simpler patterns and add complexity over time
   - Measure if this improves convergence

2. **Mixed Precision Training**: Implement mixed precision training:
   - Use PyTorch's autocast for automatic mixed precision
   - Compare memory usage and training speed
   - Ensure numerical stability

3. **Distributed Training**: Implement data parallel training:
   - Use `torch.nn.DataParallel` for multi-GPU training
   - Measure speedup compared to single GPU
   - Handle multi-GPU checkpointing

## Submission

Create a Python script `train_toy.py` that includes:
- Dataset generation and preprocessing
- Model configuration and initialization
- Complete training loop with validation
- Logging and checkpointing
- Example generation and evaluation
- Documentation and comments explaining key components

## Resources

- [PyTorch Training Loop Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Transformer Training Tips](https://arxiv.org/abs/2007.11749)
- [Learning Rate Scheduling](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)