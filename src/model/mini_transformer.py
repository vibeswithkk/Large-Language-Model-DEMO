"""
Mini Transformer Implementation
==============================

A simplified Transformer model for educational purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class MiniTransformerConfig:
    vocab_size: int = 30522
    hidden_size: int = 512
    num_attention_heads: int = 8
    num_hidden_layers: int = 6
    intermediate_size: int = 2048
    max_position_embeddings: int = 512
    dropout_prob: float = 0.1
    use_cuda: bool = True
    use_cudnn: bool = True
    layer_norm_eps: float = 1e-5
    
    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads

class MiniMultiHeadAttention(nn.Module):
    def __init__(self, config: MiniTransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        
        # Enable CUDA optimizations
        if config.use_cuda and torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
            
        # Enable cuDNN optimizations
        if config.use_cudnn:
            torch.backends.cudnn.benchmark = True
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        
        self.dropout = nn.Dropout(config.dropout_prob)
        
        # Initialize with Xavier uniform for better convergence
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Optimize for GPU computation
        if self.use_cuda and hidden_states.is_cuda:
            hidden_states = hidden_states.contiguous()
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention with optimized memory layout
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores with scaled dot-product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            # Apply attention mask
            attn_scores = attn_scores + attention_mask
            
        # Apply softmax to get attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output

class MiniTransformerLayer(nn.Module):
    def __init__(self, config: MiniTransformerConfig):
        super().__init__()
        # Enable CUDA optimizations
        if config.use_cuda and torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
            
        self.attention = MiniMultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward network with GELU activation
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout_prob)
        )
        
        # Initialize layer norms
        for layer_norm in [self.layer_norm1, self.layer_norm2]:
            if hasattr(layer_norm, 'weight'):
                nn.init.ones_(layer_norm.weight)
                nn.init.zeros_(layer_norm.bias)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Optimize for GPU computation
        if self.use_cuda and hidden_states.is_cuda:
            hidden_states = hidden_states.contiguous()
        
        # Self-attention with residual connection
        attn_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layer_norm1(hidden_states + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.layer_norm2(hidden_states + ffn_output)
        
        return hidden_states

class MiniTransformer(nn.Module):
    def __init__(self, config: MiniTransformerConfig):
        super().__init__()
        self.config = config
        
        # Enable CUDA optimizations
        if config.use_cuda and torch.cuda.is_available():
            self.use_cuda = True
            torch.backends.cudnn.benchmark = True
        else:
            self.use_cuda = False
        
        # Embedding layers
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout_prob)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MiniTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Output layers
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
        
    def post_init(self):
        # Initialize embeddings with normal distribution
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        
        # Initialize linear layers with Xavier uniform
        if hasattr(self.lm_head, 'weight'):
            nn.init.xavier_uniform_(self.lm_head.weight)
            
        # Initialize layer norms
        if hasattr(self.layer_norm, 'weight'):
            nn.init.ones_(self.layer_norm.weight)
            nn.init.zeros_(self.layer_norm.bias)
                
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        
        # Optimize for GPU computation
        if self.use_cuda and input_ids.is_cuda:
            input_ids = input_ids.contiguous()
            if attention_mask is not None:
                attention_mask = attention_mask.contiguous()
            if labels is not None:
                labels = labels.contiguous()
        
        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        
        # Embed inputs
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = token_embeds + position_embeds
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Process through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        # Final layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Prepare output
        output = {"logits": logits, "hidden_states": hidden_states}
        
        # Compute loss if labels are provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            output["loss"] = loss
            
        return output

if __name__ == "__main__":
    config = MiniTransformerConfig()
    model = MiniTransformer(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Number of layers: {config.num_hidden_layers}")
    print(f"Number of attention heads: {config.num_attention_heads}")
    
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Move to GPU if available
    if config.use_cuda and torch.cuda.is_available():
        model = model.cuda()
        input_ids = input_ids.cuda()
    
    outputs = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")