# Lesson 2: Transformer Architecture Fundamentals

## Overview
This lesson introduces the Transformer architecture, which has revolutionized natural language processing and forms the foundation of modern large language models. We'll explore the key components and mechanisms that make Transformers so effective.

## Learning Objectives
By the end of this lesson, you should be able to:
- Understand the motivation behind the Transformer architecture
- Explain the self-attention mechanism and its mathematical formulation
- Describe the components of a Transformer encoder and decoder
- Understand positional encoding and its importance
- Appreciate the advantages of Transformers over RNNs and CNNs

## 1. Introduction to Transformers

### The Evolution of Sequence Models
Before Transformers, sequence modeling relied primarily on:
- **Recurrent Neural Networks (RNNs)**: Process sequences step-by-step, maintaining hidden state
- **Convolutional Neural Networks (CNNs)**: Use convolutions with varying kernel sizes to capture context

**Limitations of previous approaches:**
- RNNs suffer from vanishing gradients and sequential processing (slow training)
- CNNs require large kernels or deep networks to capture long-range dependencies
- Both struggle with modeling complex relationships between distant elements

### The Transformer Breakthrough
The Transformer architecture, introduced in "Attention Is All You Need" (2017), addressed these limitations by:
- Replacing recurrence with self-attention mechanisms
- Enabling parallel processing of all sequence elements
- Capturing long-range dependencies effectively
- Providing better interpretability through attention weights

## 2. Self-Attention Mechanism

### Intuition
Self-attention allows each position in the sequence to attend to all positions, computing a weighted representation based on relevance.

### Mathematical Formulation
The scaled dot-product attention is computed as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- Q (Query): Represents what we're looking for
- K (Key): Represents what we can offer
- V (Value): Actual content to be aggregated
- d_k: Dimension of keys (scaling factor)

### Implementation Steps
1. **Linear Projections**: Transform input embeddings to Q, K, V
2. **Compute Attention Scores**: Calculate QK^T
3. **Scale**: Divide by √d_k to stabilize gradients
4. **Softmax**: Normalize scores to get attention weights
5. **Weighted Sum**: Apply weights to values

### Code Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
```

## 3. Multi-Head Attention

### Concept
Instead of computing a single attention function, multi-head attention runs multiple attention "heads" in parallel, allowing the model to jointly attend to information from different representation subspaces.

### Benefits
- Provides multiple "representation subspaces"
- Allows the model to focus on different aspects of the input
- Increases model capacity without significantly increasing computation

### Mathematical Formulation
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Code Example
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        
        # Linear projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_length, self.hidden_size)
        output = self.output(attention_output)
        
        return output, attention_weights
```

## 4. Positional Encoding

### The Problem
Self-attention has no inherent notion of sequence order since it processes all positions simultaneously.

### Solution
Positional encodings are added to the input embeddings to provide order information:
- **Learned**: Trainable positional embeddings
- **Fixed**: Sinusoidal functions based on position and dimension

### Sinusoidal Positional Encoding
```
PE_(pos,2i) = sin(pos / 10000^(2i/d_model))
PE_(pos,2i+1) = cos(pos / 10000^(2i/d_model))
```

### Code Example
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
```

## 5. Transformer Encoder

### Components
A Transformer encoder layer consists of:
1. **Multi-Head Attention** with residual connection and layer normalization
2. **Position-wise Feed-Forward Network** with residual connection and layer normalization

### Feed-Forward Network
A simple MLP applied to each position separately:
```
FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
```

### Code Example
```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_hidden_size, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(hidden_size, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, hidden_size)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention sub-layer
        attn_output, _ = self.self_attention(x)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x
```

## 6. Transformer Decoder

### Additional Components
The decoder includes:
1. **Masked Multi-Head Attention** (prevents attending to future positions)
2. **Encoder-Decoder Attention** (attends to encoder output)
3. **Position-wise Feed-Forward Network**

### Masked Attention
In the decoder, attention to future positions is masked to preserve autoregressive property.

### Code Example
```python
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
```

## 7. Complete Transformer Architecture

### Encoder-Decoder Structure
- **Encoder**: Stack of N identical layers
- **Decoder**: Stack of N identical layers with additional encoder-decoder attention
- **Final Linear and Softmax**: Projects decoder output to vocabulary size

### Advantages of Transformers
1. **Parallelization**: Can process all positions simultaneously
2. **Long-range dependencies**: Direct connections between any two positions
3. **Interpretability**: Attention weights provide insight into model decisions
4. **Scalability**: Easily scaled to larger models and datasets

## Summary

In this lesson, we explored the fundamental components of the Transformer architecture:
- Self-attention mechanism for capturing relationships between sequence elements
- Multi-head attention for richer representations
- Positional encoding to maintain sequence order information
- Encoder and decoder structures with residual connections and layer normalization

These components work together to create a powerful architecture that has become the foundation for modern language models.

## Next Steps

In the next lesson, we'll dive deeper into tokenization techniques, which are crucial for preparing text data for Transformer models.