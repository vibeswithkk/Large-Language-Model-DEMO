"""
Model Utilities for Saving and Loading Checkpoints
=================================================

This module provides utilities for saving and loading model checkpoints
with proper error handling and metadata management.
"""

import torch
import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    model_name: str = "model",
    max_checkpoints: int = 5
) -> str:
    """
    Save model checkpoint with metadata.
    
    Args:
        model: The model to save
        optimizer: The optimizer (optional)
        epoch: Current epoch number
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoint
        model_name: Name of the model
        max_checkpoints: Maximum number of checkpoints to keep
        
    Returns:
        Path to saved checkpoint
    """
    # Create checkpoint directory if it doesn't exist
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "model_config": getattr(model, 'config', None),
        "model_class": model.__class__.__name__
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Manage checkpoint rotation
    _manage_checkpoints(checkpoint_dir, model_name, max_checkpoints)
    
    return checkpoint_path

def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: The model to load checkpoint into
        checkpoint_path: Path to checkpoint file
        optimizer: The optimizer to load state into (optional)
        device: Device to load model on
        
    Returns:
        Checkpoint metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    # Return metadata
    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", float('inf')),
        "model_config": checkpoint.get("model_config", None),
        "model_class": checkpoint.get("model_class", "Unknown")
    }

def _manage_checkpoints(checkpoint_dir: str, model_name: str, max_checkpoints: int):
    """
    Manage checkpoint rotation to limit disk usage.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model_name: Name of the model
        max_checkpoints: Maximum number of checkpoints to keep
    """
    # Find all checkpoints for this model
    pattern = f"{model_name}_epoch_*.pt"
    checkpoints = list(Path(checkpoint_dir).glob(pattern))
    
    # Sort by modification time (oldest first)
    checkpoints.sort(key=lambda x: x.stat().st_mtime)
    
    # Remove oldest checkpoints if we exceed the limit
    if len(checkpoints) > max_checkpoints:
        for checkpoint in checkpoints[:-max_checkpoints]:
            try:
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")

def save_model_for_inference(
    model: torch.nn.Module,
    save_path: str,
    quantize: bool = False
) -> str:
    """
    Save model specifically for inference (without optimizer state).
    
    Args:
        model: The model to save
        save_path: Path to save model
        quantize: Whether to apply quantization for smaller size
        
    Returns:
        Path to saved model
    """
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    if quantize:
        # Apply quantization
        try:
            from torch.quantization import quantize_dynamic
            quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            torch.save(quantized_model.state_dict(), save_path)
            logger.info(f"Quantized model saved to {save_path}")
        except Exception as e:
            logger.warning(f"Quantization failed, saving regular model: {e}")
            torch.save(model.state_dict(), save_path)
    else:
        # Save regular model
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
    
    return save_path

def load_model_for_inference(
    model: torch.nn.Module,
    model_path: str,
    device: str = "cpu"
) -> torch.nn.Module:
    """
    Load model specifically for inference.
    
    Args:
        model: The model class to load state into
        model_path: Path to saved model
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model state
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    
    # Set to evaluation mode
    model.eval()
    
    logger.info(f"Inference model loaded from {model_path}")
    return model

# Example usage
if __name__ == "__main__":
    # This is just an example - you would use this with actual models
    logging.basicConfig(level=logging.INFO)
    logger.info("Model utilities module loaded successfully")