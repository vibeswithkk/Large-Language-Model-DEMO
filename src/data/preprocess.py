"""
Data Preprocessing Script
=========================
"""

import torch
from torch.utils.data import Dataset
import json
import re
from typing import List, Dict, Any
from pathlib import Path
import multiprocessing as mp
from functools import partial
import logging
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self, max_length: int = 512, vocab_size: int = 100000):
        self.max_length = max_length
        self.vocab_size = vocab_size
        
    def clean_text(self, text: str) -> str:
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        return text
    
    def tokenize_text(self, text: str) -> List[int]:
        # More sophisticated tokenization using hash for reproducibility
        tokens = [hash(c) % self.vocab_size for c in text]
        return tokens[:self.max_length]
    
    def pad_sequence(self, tokens: List[int], pad_token: int = 0) -> List[int]:
        # Pad or truncate sequence
        if len(tokens) < self.max_length:
            tokens = tokens + [pad_token] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        return tokens
    
    def process_text(self, text: str) -> Dict[str, Any]:
        # Process a single text example
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        padded_tokens = self.pad_sequence(tokens)
        
        return {
            "text": cleaned_text,
            "tokens": padded_tokens,
            "length": len(tokens)
        }

class LLMTextDataset(Dataset):
    def __init__(self, texts: List[str], preprocessor: TextPreprocessor):
        self.texts = texts
        self.preprocessor = preprocessor
        # Preprocess all texts during initialization for better performance
        logger.info("Preprocessing all texts...")
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=min(8, mp.cpu_count())) as executor:
            self.processed_data = list(executor.map(self.preprocessor.process_text, texts))
        logger.info(f"Preprocessed {len(self.processed_data)} texts in {time.time() - start_time:.2f}s")
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        processed = self.processed_data[idx]
        
        tokens = processed["tokens"]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "text": processed["text"]
        }

def load_texts_from_file(file_path: str) -> List[str]:
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.info(f"File {file_path} not found. Creating sample data.")
        return create_sample_data()
    
    texts = []
    
    if file_path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                texts = data
            elif isinstance(data, dict) and "texts" in data:
                texts = data["texts"]
    else:
        # Assume text file with one example per line
        with open(file_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    
    return texts

def create_sample_data(num_samples: int = 50000) -> List[str]:
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
    sample_texts = []
    for i in range(num_samples):
        text = base_texts[i % len(base_texts)] + " " + base_texts[(i + 1) % len(base_texts)]
        sample_texts.append(text[:256])  # Limit text length
    
    return sample_texts

def save_processed_data(dataset: LLMTextDataset, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use efficient serialization with numpy for better performance
    data_dict = {
        "input_ids": [],
        "labels": [],
        "texts": []
    }
    
    logger.info("Converting dataset to numpy arrays for efficient storage...")
    start_time = time.time()
    
    # Process in batches to manage memory
    batch_size = 1000
    for i in range(0, len(dataset), batch_size):
        batch_end = min(i + batch_size, len(dataset))
        batch_data = [dataset[j] for j in range(i, batch_end)]
        
        for item in batch_data:
            data_dict["input_ids"].append(item["input_ids"].numpy())
            data_dict["labels"].append(item["labels"].numpy())
            data_dict["texts"].append(item["text"])
    
    # Convert to numpy arrays for efficient storage
    data_dict["input_ids"] = np.array(data_dict["input_ids"])
    data_dict["labels"] = np.array(data_dict["labels"])
    
    # Save with numpy for better performance
    np.savez_compressed(output_path.with_suffix('.npz'), 
                       input_ids=data_dict["input_ids"],
                       labels=data_dict["labels"],
                       texts=data_dict["texts"])
    
    logger.info(f"Processed data saved to {output_path.with_suffix('.npz')} in {time.time() - start_time:.2f}s")

def parallel_preprocess_texts(texts: List[str], preprocessor: TextPreprocessor, num_workers: int = None) -> List[Dict]:
    """Preprocess texts in parallel for better performance"""
    if num_workers is None:
        num_workers = min(8, mp.cpu_count())
    
    logger.info(f"Preprocessing {len(texts)} texts using {num_workers} workers...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        processed_data = list(executor.map(preprocessor.process_text, texts))
    
    logger.info(f"Preprocessed {len(processed_data)} texts in {time.time() - start_time:.2f}s")
    return processed_data

def main() -> None:
    logger.info("Starting data preprocessing...")
    
    # Create preprocessor
    preprocessor = TextPreprocessor(max_length=128)
    
    # Load texts
    texts = create_sample_data(10000)
    logger.info(f"Loaded {len(texts)} text samples")
    
    # Preprocess in parallel for better performance
    logger.info("Preprocessing texts in parallel...")
    processed_data = parallel_preprocess_texts(texts, preprocessor, num_workers=min(8, mp.cpu_count()))
    
    # Create dataset
    dataset = LLMTextDataset(texts, preprocessor)
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    # Show sample
    sample = dataset[0]
    logger.info(f"Sample input shape: {sample['input_ids'].shape}")
    logger.info(f"Sample text: {sample['text'][:50]}...")
    
    # Save processed data
    save_processed_data(dataset, "data/processed_data.npz")
    
    logger.info("Data preprocessing completed successfully!")

if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    main()