"""
Comprehensive Test Script
========================

Test all upgraded components to ensure they work together correctly.
"""

import torch
import sys
from pathlib import Path
import time
import logging

sys.path.append(str(Path(__file__).parent))

from src.model.mini_transformer import MiniTransformer, MiniTransformerConfig
from src.model.advanced_transformer import AdvancedTransformer, AdvancedTransformerConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mini_transformer():
    """Test the mini transformer implementation"""
    logger.info("Testing Mini Transformer...")
    
    # Create configuration
    config = MiniTransformerConfig(
        vocab_size=1000,
        hidden_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=256,
        max_position_embeddings=64,
        use_cuda=True,
        use_cudnn=True
    )
    
    # Create model
    model = MiniTransformer(config)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Test with sample data
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    # Forward pass
    start_time = time.time()
    outputs = model(input_ids, labels=input_ids)
    forward_time = time.time() - start_time
    
    logger.info(f"Mini Transformer test passed!")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Input shape: {input_ids.shape}")
    logger.info(f"  Logits shape: {outputs['logits'].shape}")
    logger.info(f"  Loss: {outputs['loss'].item():.4f}")
    logger.info(f"  Forward time: {forward_time:.4f}s")
    
    return model, outputs

def test_advanced_transformer():
    """Test the advanced transformer implementation"""
    logger.info("Testing Advanced Transformer...")
    
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
        semantic_memory_size=5000,
        use_cuda=True,
        use_cudnn=True
    )
    
    # Create model
    model = AdvancedTransformer(config)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Test with sample data
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 10000, (batch_size, seq_len)).to(device)
    
    # Forward pass
    start_time = time.time()
    outputs = model(input_ids, labels=input_ids)
    forward_time = time.time() - start_time
    
    logger.info(f"Advanced Transformer test passed!")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Input shape: {input_ids.shape}")
    logger.info(f"  Logits shape: {outputs['logits'].shape}")
    logger.info(f"  Loss: {outputs['loss'].item():.4f}")
    logger.info(f"  Forward time: {forward_time:.4f}s")
    
    return model, outputs

def test_cuda_optimizations():
    """Test CUDA optimizations"""
    logger.info("Testing CUDA optimizations...")
    
    if not torch.cuda.is_available():
        logger.info("CUDA not available, skipping CUDA tests")
        return
    
    # Check cuDNN status
    logger.info(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    logger.info(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    # Test TensorFloat-32 support
    logger.info(f"TensorFloat-32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
    logger.info(f"cuDNN TensorFloat-32 enabled: {torch.backends.cudnn.allow_tf32}")

def main():
    """Run all tests"""
    logger.info("Starting comprehensive tests...")
    
    try:
        # Test CUDA optimizations
        test_cuda_optimizations()
        
        # Test mini transformer
        mini_model, mini_outputs = test_mini_transformer()
        
        # Test advanced transformer
        advanced_model, advanced_outputs = test_advanced_transformer()
        
        logger.info("All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main()