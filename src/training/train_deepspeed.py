"""
DeepSpeed Training Script
=========================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import deepspeed
import argparse
import json
import os
from pathlib import Path
import sys
import time
import logging
from typing import Dict, Any, Optional
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from model.advanced_transformer import AdvancedTransformer, AdvancedTransformerConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LargeTextDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = 512, vocab_size: int = 100000):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.samples = self.load_data(data_path)
        
    def load_data(self, data_path: str) -> list:
        samples = []
        # Create more realistic sample data
        base_texts = [
            "The field of artificial intelligence has seen tremendous growth in recent years.",
            "Machine learning algorithms can learn patterns from data without explicit programming.",
            "Deep learning models, particularly neural networks, have achieved remarkable results.",
            "Natural language processing enables computers to understand and generate human language.",
            "Computer vision allows machines to interpret and understand visual information.",
            "Reinforcement learning trains agents to make decisions through trial and error.",
            "Data science combines statistics, programming, and domain expertise to extract insights.",
            "Big data technologies handle the storage and processing of massive datasets.",
            "Cloud computing provides scalable resources for machine learning workloads.",
            "Ethical AI ensures that artificial intelligence systems are fair and unbiased."
        ]
        
        # Generate a larger dataset
        for i in range(10000):
            text = base_texts[i % len(base_texts)] + " " + base_texts[(i + 1) % len(base_texts)]
            samples.append(text[:self.max_length * 2])  # Allow for longer texts that will be truncated
            
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.samples[idx]
        # More sophisticated tokenization
        tokens = [hash(c) % self.vocab_size for c in text]
        tokens = tokens[:self.max_length-1]  # Truncate to max_length-1
        tokens = tokens + [0] * (self.max_length - len(tokens))  # Pad to max_length
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {"input_ids": input_ids, "labels": labels}

def create_deepspeed_config(batch_size: int = 32, gradient_accumulation_steps: int = 1) -> Dict[str, Any]:
    return {
        "train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 5e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 5e-5,
                "warmup_num_steps": 1000,
                "total_num_steps": 100000
            }
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "nvme",
                "nvme_path": "/tmp/nvme",
                "pin_memory": True,
                "buffer_count": 4,
                "fast_init": True
            },
            "offload_param": {
                "device": "nvme",
                "nvme_path": "/tmp/nvme",
                "pin_memory": True,
                "buffer_count": 5,
                "buffer_size": 1e8,
                "max_in_cpu": 1e9
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "gather_16bit_weights_on_model_save": True,
            "round_robin_gradients": True
        },
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": True,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 32,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": False
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": False,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        },
        "wall_clock_breakdown": False,
        "flops_profiler": {
            "enabled": True,
            "profile_step": 100,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
            "output_file": "flops_profiler.log"
        },
        "tensorboard": {
            "enabled": True,
            "output_path": "tensorboard_logs",
            "job_name": "deepspeed_training"
        }
    }

def train_model(model_engine, train_dataloader, num_epochs: int = 3, gradient_accumulation_steps: int = 1) -> None:
    model_engine.train()
    
    # Initialize metrics
    total_steps = 0
    total_samples = 0
    epoch_times = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()
        step_times = []
        
        for step, batch in enumerate(train_dataloader):
            step_start_time = time.time()
            
            # Move tensors to device with non-blocking transfer for better performance
            input_ids = batch["input_ids"].to(model_engine.device, non_blocking=True)
            labels = batch["labels"].to(model_engine.device, non_blocking=True)
            
            # Forward pass with AMP (Automatic Mixed Precision)
            outputs = model_engine(input_ids, labels=labels)
            loss = outputs.loss
            
            # Scale loss for gradient accumulation
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            model_engine.backward(loss)
            
            # Step optimizer only after accumulating enough gradients
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                model_engine.step()
                total_steps += 1
            
            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            total_samples += input_ids.size(0)
            
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            
            # Log progress with more detailed metrics
            if step % 50 == 0:
                avg_step_time = np.mean(step_times[-50:]) if len(step_times) >= 50 else np.mean(step_times)
                samples_per_sec = input_ids.size(0) / step_time
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Step {step}, "
                           f"Loss: {loss.item() * gradient_accumulation_steps:.4f}, "
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
    # Enable cuDNN benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Set CUDA device memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set memory fraction to avoid OOM issues
        # torch.cuda.set_per_process_memory_fraction(0.9)
        
        logger.info(f"CUDA available. Using {torch.cuda.device_count()} GPU(s)")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
    # Set environment variables for better CUDA performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CUDA_CACHE_MAXSIZE'] = '2147483648'  # 2GB cache
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Enable cuDNN v8 API

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--data_path", type=str, default="data/training_data.txt")
    parser.add_argument("--output_dir", type=str, default="output/deepspeed_model")
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    
    # Setup CUDA environment
    setup_cuda_environment()
    
    # Initialize DeepSpeed distributed training
    deepspeed.init_distributed()
    
    # Create model configuration with CUDA optimizations
    config = AdvancedTransformerConfig(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=12,
        intermediate_size=8192,
        max_position_embeddings=2048,
        num_modalities=4,
        gpu_acceleration_units=32,
        spiking_neurons=True,
        continuous_learning=True,
        episodic_memory_size=50000,
        semantic_memory_size=250000,
        use_cuda=True,
        use_cudnn=True
    )
    
    # Create model
    model = AdvancedTransformer(config)
    
    # Create DeepSpeed configuration
    ds_config = create_deepspeed_config(
        batch_size=args.batch_size * torch.cuda.device_count() if torch.cuda.is_available() else args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    # Create dataset and dataloader with advanced optimizations
    dataset = LargeTextDataset(args.data_path, max_length=args.max_length)
    train_dataloader = DataLoader(
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
    logger.info(f"Using gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * torch.cuda.device_count() if torch.cuda.is_available() else args.batch_size * args.gradient_accumulation_steps}")
    
    # Start training
    logger.info("Starting DeepSpeed training with CUDA optimizations...")
    train_start_time = time.time()
    train_model(model_engine, train_dataloader, num_epochs=args.epochs, 
                gradient_accumulation_steps=args.gradient_accumulation_steps)
    train_time = time.time() - train_start_time
    
    logger.info(f"Total training time: {train_time:.2f}s")
    
    # Save model
    if model_engine.local_rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model checkpoint
        model_engine.save_checkpoint(output_dir)
        
        # Save configuration
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.__dict__, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
    
    logger.info("DeepSpeed training completed successfully!")

if __name__ == "__main__":
    main()