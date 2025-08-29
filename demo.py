"""
Demo Script
===========

This script demonstrates the usage of the upgraded models in the Large Language Model DEMO project.
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.model.mini_transformer import MiniTransformer, MiniTransformerConfig
from src.model.advanced_transformer import AdvancedTransformer, AdvancedTransformerConfig

def demo_mini_transformer():
    """Demonstrate the mini transformer model"""
    print("=== Mini Transformer Demo ===")
    
    # Create configuration
    config = MiniTransformerConfig(
        vocab_size=1000,
        hidden_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=256,
        max_position_embeddings=64
    )
    
    # Create model
    model = MiniTransformer(config)
    
    # Show model information
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Test with sample data
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    outputs = model(input_ids, labels=input_ids)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print()

def demo_advanced_transformer():
    """Demonstrate the advanced transformer model"""
    print("=== Advanced Transformer Demo ===")
    
    # Create configuration
    config = AdvancedTransformerConfig(
        hidden_size=256,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=512,
        max_position_embeddings=64,
        num_modalities=2,
        gpu_acceleration_units=16,
        spiking_neurons=True,
        continuous_learning=True,
        episodic_memory_size=1000,
        semantic_memory_size=5000
    )
    
    # Create model
    model = AdvancedTransformer(config)
    
    # Show model information
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Test with sample data
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    outputs = model(input_ids, labels=input_ids)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print()

def main():
    """Run all demos"""
    print("Large Language Model DEMO - Usage Demonstration")
    print("=" * 50)
    print()
    
    try:
        # Demo mini transformer
        demo_mini_transformer()
        
        # Demo advanced transformer
        demo_advanced_transformer()
        
        print("All demos completed successfully!")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()