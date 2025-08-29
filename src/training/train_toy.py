"""
Toy Training Script
===================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import json
from pathlib import Path
import sys
import time
import logging
from typing import Dict, Any, Optional
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from model.mini_transformer import MiniTransformer, MiniTransformerConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, texts: list, max_length: int = 128, vocab_size: int = 10000):
        self.texts = texts
        self.max_length = max_length
        self.vocab_size = vocab_size
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        # More sophisticated tokenization
        tokens = [hash(c) % self.vocab_size for c in text]
        tokens = tokens[:self.max_length-1]  # Truncate to max_length-1
        tokens = tokens + [0] * (self.max_length - len(tokens))  # Pad to max_length
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {"input_ids": input_ids, "labels": labels}

def create_sample_data(num_samples: int = 10000) -> list:
    base_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models require large amounts of data.",
        "Transformers have revolutionized the field of NLP.",
        "Attention mechanisms allow models to focus on relevant parts.",
        "Python is a popular programming language for machine learning.",
        "PyTorch provides flexible tools for deep learning research.",
        "Data preprocessing is crucial for model performance.",
        "Model evaluation helps measure the effectiveness of algorithms."
    ]
    
    # Generate a larger dataset
    sample_texts = []
    for i in range(num_samples):
        text = base_texts[i % len(base_texts)] + " " + base_texts[(i + 1) % len(base_texts)]
        sample_texts.append(text[:256])  # Limit text length
    
    return sample_texts

def train_model(model, train_dataloader, optimizer, scheduler, device, num_epochs: int = 3) -> None:
    model.train()
    
    # Initialize metrics
    total_steps = 0
    total_samples = 0
    epoch_times = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()
        step_times = []
        
        for batch_idx, batch in enumerate(train_dataloader):
            batch_start_time = time.time()
            
            # Move tensors to device with non-blocking transfer for better performance
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            # Forward pass with AMP-like behavior (mixed precision simulation)
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)  # More memory efficient
            
            # Scale loss for gradient accumulation simulation
            scaled_loss = loss / 1  # No accumulation in this case
            scaled_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            total_samples += input_ids.size(0)
            total_steps += 1
            
            step_time = time.time() - batch_start_time
            step_times.append(step_time)
            
            # Log progress with more detailed metrics
            if batch_idx % 50 == 0:
                avg_step_time = np.mean(step_times[-50:]) if len(step_times) >= 50 else np.mean(step_times)
                current_lr = optimizer.param_groups[0]['lr']
                samples_per_sec = input_ids.size(0) / step_time
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                           f"Loss: {loss.item():.4f}, "
                           f"LR: {current_lr:.6f}, "
                           f"Step Time: {step_time:.2f}s, "
                           f"Avg Step Time: {avg_step_time:.2f}s, "
                           f"Samples/sec: {samples_per_sec:.2f}")
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed. "
                   f"Average Loss: {avg_loss:.4f}, "
                   f"Epoch Time: {epoch_time:.2f}s, "
                   f"Total Steps: {total_steps}, "
                   f"Total Samples: {total_samples}")

def setup_cuda_environment() -> None:
    """Setup CUDA environment for optimal performance"""
    if torch.cuda.is_available():
        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Set memory management
        torch.cuda.empty_cache()
        logger.info(f"CUDA available. Using {torch.cuda.device_count()} GPU(s)")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Set environment variables for better CUDA performance
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    else:
        logger.info("CUDA not available. Training on CPU.")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    
    # Setup CUDA environment
    setup_cuda_environment()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model configuration with CUDA optimizations
    config = MiniTransformerConfig(
        vocab_size=10000,
        hidden_size=256,
        num_attention_heads=4,
        num_hidden_layers=4,
        intermediate_size=1024,
        max_position_embeddings=128,
        use_cuda=True,
        use_cudnn=True
    )
    
    # Create model
    model = MiniTransformer(config)
    model.to(device)
    
    # Use mixed precision training if CUDA is available
    if device.type == 'cuda':
        # Enable TensorFloat-32 for better performance on modern GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create optimizer with weight decay for better generalization
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * 1000,  # Adjust based on your needs
        eta_min=1e-6
    )
    
    # Create sample data
    sample_texts = create_sample_data(args.num_samples)
    
    # Create dataset and dataloader with advanced optimizations
    dataset = TextDataset(sample_texts, max_length=args.max_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=True,  # Enable pinned memory for faster GPU transfer
        num_workers=4,    # Use multiple workers for data loading
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch data for better performance
        drop_last=True    # Drop last incomplete batch for consistent batch sizes
    )
    
    logger.info(f"Created dataset with {len(dataset)} samples")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Using batch size: {args.batch_size}")
    logger.info(f"Using learning rate: {args.learning_rate}")
    logger.info(f"Using max sequence length: {args.max_length}")
    
    # Start training
    logger.info("Starting training with CUDA optimizations...")
    train_start_time = time.time()
    train_model(model, dataloader, optimizer, scheduler, device, num_epochs=args.epochs)
    train_time = time.time() - train_start_time
    
    logger.info(f"Total training time: {train_time:.2f}s")
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "mini_transformer.pth"
    torch.save(model.state_dict(), model_path)
    
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Config saved to {config_path}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()