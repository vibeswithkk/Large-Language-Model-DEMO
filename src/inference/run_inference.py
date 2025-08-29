"""
Inference Script
===============
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import json
import time
import logging
from typing import Dict, List, Optional
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from model.mini_transformer import MiniTransformer, MiniTransformerConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextGenerator:
    def __init__(self, model, tokenizer=None, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Enable CUDA optimizations
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            # Use torch.compile for better performance on newer PyTorch versions
            if hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model)
                    logger.info("Model compiled for better performance")
                except Exception as e:
                    logger.warning(f"Could not compile model: {e}")
            
            # Enable TensorFloat-32 for better performance on modern GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
    def generate_text(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> str:
        # Move to device and optimize for inference
        input_tokens = [hash(c) % 10000 for c in prompt]
        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
        
        # Use torch.no_grad() context for inference
        with torch.no_grad():
            generated = input_ids.clone()
            
            # Pre-allocate tensor for better memory management
            generated_tokens = torch.empty((1, max_length + len(input_tokens)), dtype=torch.long, device=self.device)
            generated_tokens[:, :len(input_tokens)] = generated
            
            for i in range(max_length):
                # Forward pass
                outputs = self.model(generated)
                logits = outputs["logits"]
                
                # Apply temperature scaling
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if do_sample:
                    next_token_logits = self._top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
            
            # Convert tokens back to text
            generated_tokens = generated[0].cpu().tolist()
            generated_text = ''.join([chr(token % 128) for token in generated_tokens[len(input_tokens):]])
            
            return prompt + generated_text
    
    def _top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering"""
        assert logits.dim() == 2  # batch_size x vocab_size
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits
    
    def get_next_token_predictions(self, text: str, top_k: int = 5) -> dict:
        # Move to device and optimize for inference
        input_tokens = [hash(c) % 10000 for c in text]
        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
        
        # Use torch.no_grad() context for inference
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs["logits"]
            next_token_logits = logits[:, -1, :]
            
            # Get top-k predictions efficiently
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            
        return {
            "tokens": top_k_indices[0].cpu().tolist(),
            "probabilities": top_k_probs[0].cpu().tolist()
        }

def load_model(model_path: str, config_path: str, device: str = "cpu") -> MiniTransformer:
    # Load configuration
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    config = MiniTransformerConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create model
    model = MiniTransformer(config)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model

def benchmark_inference(model, generator, device) -> Dict[str, float]:
    """Benchmark inference performance"""
    logger.info("Benchmarking inference performance...")
    
    # Test text generation
    prompt = "The future of artificial intelligence"
    
    start_time = time.time()
    generated_text = generator.generate_text(prompt, max_length=30, temperature=0.8)
    generation_time = time.time() - start_time
    
    logger.info(f"Generation time: {generation_time:.4f}s")
    logger.info(f"Generated text: {generated_text[:100]}...")
    
    # Test next token predictions
    start_time = time.time()
    predictions = generator.get_next_token_predictions("Machine learning is", top_k=5)
    prediction_time = time.time() - start_time
    
    logger.info(f"Prediction time: {prediction_time:.4f}s")
    logger.info("Top predictions:")
    for i, (token, prob) in enumerate(zip(predictions["tokens"], predictions["probabilities"])):
        logger.info(f"  {i+1}. Token {token}: {prob:.4f}")
    
    return {
        "generation_time": generation_time,
        "prediction_time": prediction_time,
        "tokens_per_second": len(generated_text) / generation_time
    }

def main() -> None:
    logger.info("Loading model for inference...")
    
    # Set device with CUDA optimizations
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        
        # Enable TensorFloat-32 for better performance on modern GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for inference")
    
    # Create model configuration
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
    
    # Create text generator
    generator = TextGenerator(model, device=device)
    
    # Run benchmark
    benchmark_results = benchmark_inference(model, generator, device)
    
    logger.info("\n=== Text Generation Example ===")
    prompt = "The future of artificial intelligence"
    generated_text = generator.generate_text(prompt, max_length=30, temperature=0.8)
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Generated: {generated_text}")
    
    logger.info("\n=== Next Token Predictions Example ===")
    text = "Machine learning is"
    predictions = generator.get_next_token_predictions(text, top_k=5)
    logger.info(f"Text: {text}")
    logger.info("Top 5 predictions:")
    for i, (token, prob) in enumerate(zip(predictions["tokens"], predictions["probabilities"])):
        logger.info(f"  {i+1}. Token {token}: {prob:.4f}")
    
    logger.info("\n=== Benchmark Results ===")
    for key, value in benchmark_results.items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("\nInference examples completed!")

if __name__ == "__main__":
    main()