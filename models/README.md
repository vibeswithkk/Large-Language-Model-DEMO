# Model Checkpoints

This directory contains trained model checkpoints and related files.

## Directory Structure

```
models/
├── mini_transformer/
│   ├── checkpoint-1000/
│   ├── checkpoint-2000/
│   └── final/
└── advanced_transformer/
    ├── checkpoint-1000/
    ├── checkpoint-2000/
    └── final/
```

## Model Types

### Mini Transformer
- Located in [src/model/mini_transformer.py](file:///C:/Users/wahyu/Documents/Large-Language-Model-DEMO/src/model/mini_transformer.py)
- Lightweight model for learning and development
- Suitable for CPU execution
- ~500K parameters

### Advanced Transformer
- Located in [src/model/advanced_transformer.py](file:///C:/Users/wahyu/Documents/Large-Language-Model-DEMO/src/model/advanced_transformer.py)
- Full-featured model with advanced capabilities
- Optimized for GPU execution
- Supports CUDA and cuDNN optimizations
- ~56M parameters

## Checkpoint Format

Model checkpoints are saved in PyTorch format and include:
- Model state dictionary
- Optimizer state (for training checkpoints)
- Training configuration
- Training metrics and logs

## Usage

### Loading a Model

```python
import torch
from src.model.mini_transformer import MiniTransformer, MiniTransformerConfig

# Load model configuration
config = MiniTransformerConfig()

# Create model
model = MiniTransformer(config)

# Load checkpoint
checkpoint = torch.load("models/mini_transformer/final/model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Set to evaluation mode
model.eval()
```

### Saving a Model

```python
# Save model checkpoint
torch.save({
    "model_state_dict": model.state_dict(),
    "config": model.config,
    "epoch": epoch,
    "loss": loss
}, "models/mini_transformer/checkpoint-1000/model.pt")
```

## Best Practices

1. **Version Control**: Keep track of model versions and changes
2. **Regular Backups**: Backup important checkpoints to remote storage
3. **Metadata**: Include training configuration and metrics with checkpoints
4. **Compression**: Consider compressing checkpoints for storage efficiency
5. **Validation**: Validate checkpoints before deploying to production