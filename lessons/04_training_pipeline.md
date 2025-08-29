# Lesson 4: Transformer Training Pipeline

## Overview
This lesson covers the complete training pipeline for transformer models, from data preparation to model optimization. We'll explore the key components and best practices for training effective language models.

## Learning Objectives
By the end of this lesson, you should be able to:
- Understand the components of a transformer training pipeline
- Prepare and preprocess data for language model training
- Implement training loops with proper optimization
- Apply regularization techniques to prevent overfitting
- Use distributed training for large-scale models
- Monitor and evaluate training progress

## 1. Data Preparation and Preprocessing

### Dataset Selection
For language model training, datasets can include:
- **Books and Articles**: High-quality, diverse text
- **Web Scraping**: Large-scale but requires cleaning
- **Code Repositories**: For code understanding models
- **Multimodal Data**: Text paired with images, audio, etc.

### Data Cleaning
```python
import re
import torch
from torch.utils.data import Dataset, DataLoader

def clean_text(text):
    """Basic text cleaning function."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Handle special characters (customize based on needs)
    text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)
    return text

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = clean_text(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
```

### Data Loading and Batching
```python
from torch.utils.data import DataLoader

def create_data_loader(texts, tokenizer, batch_size=8, max_length=512):
    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True  # Faster GPU transfer
    )
    return dataloader
```

## 2. Model Configuration and Initialization

### Model Hyperparameters
```python
class ModelConfig:
    def __init__(self):
        # Model architecture
        self.vocab_size = 30522
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.max_position_embeddings = 512
        
        # Training parameters
        self.learning_rate = 5e-5
        self.batch_size = 16
        self.num_epochs = 3
        self.warmup_steps = 1000
        self.weight_decay = 0.01
        
        # Regularization
        self.dropout_rate = 0.1
        self.gradient_clipping = 1.0

config = ModelConfig()
```

### Weight Initialization
```python
import torch.nn as nn

def init_weights(model):
    """Initialize model weights."""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
```

## 3. Optimizer and Scheduler

### AdamW Optimizer
```python
from transformers import AdamW, get_linear_schedule_with_warmup

def create_optimizer_and_scheduler(model, config, train_dataloader):
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        correct_bias=False
    )
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * config.num_epochs
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler
```

## 4. Training Loop Implementation

### Basic Training Loop
```python
import torch.nn.functional as F
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)
```

### Complete Training Function
```python
def train_model(model, train_loader, val_loader, config, device):
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, train_loader)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validation
        val_loss = validate(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("New best model saved!")
```

## 5. Distributed Training

### Data Parallel Training
```python
import torch.nn as nn

def setup_data_parallel(model, device):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)
    return model
```

### Distributed Data Parallel (DDP)
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, world_size):
    """Initialize the distributed environment."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train_ddp(rank, world_size, model, train_loader, config):
    """Training function for DDP."""
    setup_ddp(rank, world_size)
    
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, train_loader)
    
    # Training loop
    for epoch in range(config.num_epochs):
        train_epoch(model, train_loader, optimizer, scheduler, device)
    
    cleanup_ddp()
```

## 6. Mixed Precision Training

### Using PyTorch's Automatic Mixed Precision
```python
from torch.cuda.amp import autocast, GradScaler

def train_epoch_amp(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    scaler = GradScaler()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone()
        
        # Forward pass with autocast
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)
```

## 7. Monitoring and Logging

### TensorBoard Integration
```python
from torch.utils.tensorboard import SummaryWriter

def train_with_logging(model, train_loader, val_loader, config, device):
    writer = SummaryWriter('runs/transformer_experiment')
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, train_loader)
    
    for epoch in range(config.num_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        # Validation
        val_loss = validate(model, val_loader, device)
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    writer.close()
```

### Custom Logging
```python
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def log_training_progress(epoch, train_loss, val_loss, learning_rate):
    """Log training progress."""
    log_data = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': learning_rate,
        'timestamp': time.time()
    }
    
    logging.info(json.dumps(log_data))
```

## 8. Checkpointing and Model Persistence

### Saving Checkpoints
```python
def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, scheduler, filepath):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
```

### Model Export
```python
def export_model(model, tokenizer, export_dir):
    """Export model for inference."""
    # Save model
    model.save_pretrained(export_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(export_dir)
    
    # Save config
    config_dict = {
        'model_type': 'transformer',
        'hidden_size': model.config.hidden_size,
        'num_layers': model.config.num_hidden_layers,
        'num_heads': model.config.num_attention_heads
    }
    
    with open(f'{export_dir}/model_config.json', 'w') as f:
        json.dump(config_dict, f)
```

## 9. Advanced Training Techniques

### Gradient Accumulation
```python
def train_with_gradient_accumulation(model, dataloader, optimizer, scheduler, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Normalize loss by accumulation steps
        loss = outputs.loss / accumulation_steps
        loss.backward()
        
        # Perform optimizer step after accumulation
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    return total_loss / len(dataloader)
```

### Early Stopping
```python
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience
```

## Summary

In this lesson, we covered the complete transformer training pipeline:
- Data preparation and preprocessing
- Model configuration and initialization
- Optimizer and scheduler setup
- Training loop implementation
- Distributed and mixed precision training
- Monitoring and logging
- Checkpointing and model persistence
- Advanced training techniques

These components work together to create an effective training pipeline for transformer models.

## Next Steps

In the next lesson, we'll explore inference and serving techniques for deploying trained transformer models in production environments.