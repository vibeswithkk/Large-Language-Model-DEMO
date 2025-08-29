# Exercise 3: Build a Simple Transformer

## Objective
Implement a simplified Transformer model from scratch to understand the core components and architecture.

## Prerequisites
- Completion of Exercises 1 and 2
- Understanding of tensor operations and tokenization
- Knowledge of PyTorch neural network modules

## Instructions

### Part 1: Multi-Head Attention

1. Implement a `MultiHeadAttention` class:
   - Initialize with `hidden_size`, `num_heads`, and `dropout`
   - Implement the `forward` method
   - Handle mask for padded sequences

2. Key components:
   - Linear projections for Query, Key, and Value
   - Split heads for parallel attention computation
   - Scaled dot-product attention
   - Combine heads and final linear projection
   - Dropout for regularization

3. Attention mechanism:
   - Compute attention scores: `Q @ K^T / sqrt(d_k)`
   - Apply mask to prevent attending to padding tokens
   - Apply softmax to get attention weights
   - Apply attention weights to values: `weights @ V`

### Part 2: Transformer Layer

1. Implement a `TransformerLayer` class:
   - Include multi-head attention with residual connection
   - Implement layer normalization
   - Add position-wise feed-forward network
   - Include dropout and residual connections

2. Feed-forward network:
   - Two linear layers with activation function (GELU or ReLU)
   - Expansion to intermediate size and back
   - Dropout for regularization

3. Residual connections and layer normalization:
   - Apply LayerNorm after attention and feed-forward
   - Add residual connections around both components

### Part 3: Complete Transformer Model

1. Implement a `SimpleTransformer` class:
   - Embedding layer for token embeddings
   - Positional encoding (learned or fixed)
   - Stack of transformer layers
   - Final layer normalization
   - Output projection to vocabulary size

2. Positional encoding:
   - Implement learned positional embeddings
   - Optional: Implement sinusoidal positional encoding

3. Model interface:
   - `forward` method for training
   - `generate` method for text generation
   - Handle input tensors and optional attention masks

### Part 4: Training Setup

1. Loss function:
   - Implement cross-entropy loss for language modeling
   - Handle padding tokens in loss computation

2. Optimizer:
   - Use Adam optimizer with learning rate scheduling
   - Implement learning rate warmup

3. Training loop:
   - Batch processing with padding
   - Gradient accumulation for large sequences
   - Logging and checkpointing

## Challenge Problems

1. **Efficiency Optimization**: Optimize your transformer implementation:
   - Implement attention with PyTorch's `scaled_dot_product_attention` (if available)
   - Use `torch.compile` for performance improvements
   - Profile memory usage and optimize accordingly

2. **Advanced Attention**: Implement more sophisticated attention mechanisms:
   - Relative positional encoding
   - Rotary positional embeddings (RoPE)
   - Sparse attention patterns

3. **Model Scaling**: Extend your model to handle larger contexts:
   - Implement gradient checkpointing
   - Add mixed precision training
   - Optimize memory usage for long sequences

## Submission

Create a Python module `simple_transformer.py` with your implementation. Include:
- All required classes: `MultiHeadAttention`, `TransformerLayer`, `SimpleTransformer`
- Example usage with training and generation
- Unit tests for core components
- Documentation for each class and method

## Resources

- [Attention Is All You Need Paper](https://arxiv.org/abs/1706.03762)
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)