# Lesson 3: Tokenization for Language Models

## Overview
This lesson explores tokenization, the process of converting text into numerical representations that can be processed by machine learning models. We'll examine different tokenization approaches and their impact on model performance.

## Learning Objectives
By the end of this lesson, you should be able to:
- Understand the importance of tokenization in NLP
- Explain different tokenization strategies (character, word, subword)
- Implement basic tokenization techniques
- Understand the trade-offs between different approaches
- Appreciate the role of tokenization in modern language models

## 1. Introduction to Tokenization

### What is Tokenization?
Tokenization is the process of breaking down text into smaller units called tokens, which can be:
- Characters
- Words
- Subwords (parts of words)
- Sentences

These tokens are then converted into numerical representations (token IDs) that models can process.

### Why is Tokenization Important?
1. **Model Input**: Neural networks process numerical data, not raw text
2. **Vocabulary Management**: Controls the size and composition of the model's vocabulary
3. **Generalization**: Affects how well the model handles unseen words
4. **Efficiency**: Impacts model size, training time, and inference speed

## 2. Character-Level Tokenization

### Approach
Treat each character as a separate token.

### Advantages
- Small vocabulary (typically 50-200 tokens)
- Can handle any word, including out-of-vocabulary (OOV) words
- Simple to implement

### Disadvantages
- Very long sequences (more tokens per word)
- Difficult to capture word-level meaning
- Computationally expensive due to long sequences

### Implementation Example
```python
class CharacterTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
    def fit(self, texts):
        # Collect all unique characters
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        # Add special tokens
        self.char_to_idx['<PAD>'] = self.vocab_size
        self.char_to_idx['<UNK>'] = self.vocab_size + 1
        self.idx_to_char[self.vocab_size] = '<PAD>'
        self.idx_to_char[self.vocab_size + 1] = '<UNK>'
        self.vocab_size += 2
        
    def encode(self, text, max_length=None):
        # Convert text to token IDs
        tokens = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in text]
        
        # Handle padding/truncation
        if max_length:
            if len(tokens) < max_length:
                tokens.extend([self.char_to_idx['<PAD>']] * (max_length - len(tokens)))
            else:
                tokens = tokens[:max_length]
                
        return tokens
    
    def decode(self, token_ids):
        # Convert token IDs back to text
        chars = []
        for token_id in token_ids:
            if token_id == self.char_to_idx['<PAD>']:
                break
            chars.append(self.idx_to_char.get(token_id, '<UNK>'))
        return ''.join(chars)

# Example usage
texts = ["Hello world!", "How are you?"]
tokenizer = CharacterTokenizer()
tokenizer.fit(texts)

encoded = tokenizer.encode("Hello", max_length=10)
print(f"Encoded: {encoded}")
decoded = tokenizer.decode(encoded)
print(f"Decoded: {decoded}")
```

## 3. Word-Level Tokenization

### Approach
Treat each word as a separate token.

### Advantages
- Intuitive and linguistically meaningful
- Shorter sequences than character-level
- Easier to interpret attention patterns

### Disadvantages
- Large vocabulary size (can be 10K-100K+ tokens)
- Poor handling of OOV words
- Inflexible with typos and variations

### Implementation Example
```python
import re
from collections import Counter

class WordTokenizer:
    def __init__(self, lowercase=True):
        self.lowercase = lowercase
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def preprocess(self, text):
        # Simple preprocessing
        if self.lowercase:
            text = text.lower()
        # Split on whitespace and punctuation
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def fit(self, texts, max_vocab_size=None):
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = self.preprocess(text)
            word_counts.update(words)
        
        # Sort by frequency and limit vocabulary size
        if max_vocab_size:
            most_common = word_counts.most_common(max_vocab_size - 2)  # Reserve space for special tokens
        else:
            most_common = word_counts.most_common()
        
        # Create mappings
        words = [word for word, _ in most_common]
        self.word_to_idx = {word: idx for idx, word in enumerate(words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        # Add special tokens
        self.word_to_idx['<PAD>'] = self.vocab_size
        self.word_to_idx['<UNK>'] = self.vocab_size + 1
        self.idx_to_word[self.vocab_size] = '<PAD>'
        self.idx_to_word[self.vocab_size + 1] = '<UNK>'
        self.vocab_size += 2
        
    def encode(self, text, max_length=None):
        words = self.preprocess(text)
        tokens = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
        
        # Handle padding/truncation
        if max_length:
            if len(tokens) < max_length:
                tokens.extend([self.word_to_idx['<PAD>']] * (max_length - len(tokens)))
            else:
                tokens = tokens[:max_length]
                
        return tokens
    
    def decode(self, token_ids):
        words = []
        for token_id in token_ids:
            if token_id == self.word_to_idx['<PAD>']:
                break
            words.append(self.idx_to_word.get(token_id, '<UNK>'))
        return ' '.join(words)

# Example usage
texts = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog jumps over the lazy fox"
]
tokenizer = WordTokenizer()
tokenizer.fit(texts, max_vocab_size=100)

encoded = tokenizer.encode("The quick brown fox", max_length=10)
print(f"Encoded: {encoded}")
decoded = tokenizer.decode(encoded)
print(f"Decoded: {decoded}")
```

## 4. Subword Tokenization

### Approach
Break words into subword units that balance the vocabulary size and sequence length.

### Popular Methods
1. **Byte Pair Encoding (BPE)**
2. **WordPiece**
3. **SentencePiece**
4. **Unigram**

### Advantages
- Handles OOV words by breaking them into known subwords
- Smaller vocabulary than word-level (typically 10K-50K tokens)
- Reasonable sequence lengths
- Good generalization to new text

### Disadvantages
- More complex to implement
- Less interpretable than word-level tokenization
- Requires training on representative text corpus

## 5. Byte Pair Encoding (BPE)

### Algorithm
1. Start with character-level tokens
2. Iteratively merge the most frequent adjacent token pairs
3. Continue until desired vocabulary size is reached

### Implementation Example
```python
from collections import Counter, defaultdict

class SimpleBPE:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.bpe_vocab = []
        self.bpe_ranks = {}
        
    def get_pairs(self, word):
        """Get pairs of consecutive symbols in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def merge_vocab(self, pair, vocab):
        """Merge all occurrences of pair in vocabulary."""
        bigram = ''.join(pair)
        new_vocab = {}
        for word in vocab:
            new_word = word.replace(' '.join(pair), bigram)
            new_vocab[new_word] = vocab[word]
        return new_vocab
    
    def fit(self, texts):
        # Initialize vocabulary with characters
        vocab = Counter()
        for text in texts:
            for word in text.split():
                # Add end-of-word symbol
                vocab[' '.join(list(word)) + ' </w>'] += 1
        
        # Learn BPE merges
        num_merges = self.vocab_size - len(set(char for word in vocab for char in word.split()))
        merges = []
        
        for i in range(num_merges):
            # Count pairs
            pairs = Counter()
            for word, freq in vocab.items():
                for pair in self.get_pairs(word.split()):
                    pairs[pair] += freq
            
            if not pairs:
                break
                
            # Get most frequent pair
            best_pair = pairs.most_common(1)[0][0]
            merges.append(best_pair)
            
            # Merge in vocabulary
            vocab = self.merge_vocab(best_pair, vocab)
        
        # Create final vocabulary
        self.bpe_vocab = set(char for word in vocab for char in word.split())
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        
        return self
    
    def encode(self, text):
        # Simple encoding - in practice, this would be more sophisticated
        words = text.split()
        encoded = []
        for word in words:
            # This is a simplified version - real BPE would apply merges
            encoded.extend(list(word))
        return encoded

# Example usage (simplified)
texts = [
    "low lowest newer newest",
    "wide wider widest",
    "deep deeper deepest"
]
bpe = SimpleBPE(vocab_size=100)
bpe.fit(texts)

encoded = bpe.encode("lower")
print(f"Encoded: {encoded}")
```

## 6. Using Hugging Face Tokenizers

### Installation
```bash
pip install tokenizers
```

### Example Usage
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Create a BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Create trainer
trainer = BpeTrainer(
    vocab_size=1000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# Train on files
files = ["path/to/text/file1.txt", "path/to/text/file2.txt"]
tokenizer.train(files, trainer)

# Save tokenizer
tokenizer.save("tokenizer.json")

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

# Encode text
encoding = tokenizer.encode("Hello, how are you?")
print(encoding.ids)
print(encoding.tokens)
```

## 7. Tokenization Best Practices

### Vocabulary Size Considerations
- **Small vocab** (1K-5K): Longer sequences, better OOV handling
- **Medium vocab** (10K-30K): Good balance for most tasks
- **Large vocab** (50K+): Shorter sequences, more parameters

### Special Tokens
- **[PAD]**: Padding for batching sequences
- **[UNK]**: Unknown tokens not in vocabulary
- **[CLS]**: Classification token (for BERT-like models)
- **[SEP]**: Separator token (for BERT-like models)
- **[MASK]**: Mask token (for masked language modeling)

### Preprocessing
- Normalize text (lowercase, remove extra whitespace)
- Handle special characters and punctuation
- Consider language-specific preprocessing
- Apply consistent preprocessing at train and inference time

## 8. Impact on Model Performance

### Sequence Length
- Longer sequences require more memory and computation
- Attention complexity is O(n²) with sequence length
- Very long sequences may need special handling (chunking, sliding window)

### Vocabulary Size
- Larger vocabularies increase model size
- More parameters in embedding and output layers
- May require more training data to learn embeddings well

### Out-of-Vocabulary Handling
- Character/subword approaches handle OOV better
- OOV tokens can significantly impact model performance
- Consider domain-specific vocabulary for specialized tasks

## Summary

In this lesson, we explored various tokenization approaches:
- Character-level: Simple but results in long sequences
- Word-level: Intuitive but struggles with vocabulary size and OOV words
- Subword: Balances vocabulary size and sequence length, handles OOV well

Tokenization is a crucial preprocessing step that significantly impacts model performance, training efficiency, and generalization capabilities.

## Next Steps

In the next lesson, we'll explore the complete training pipeline for transformer models, including data preparation, training loops, and optimization techniques.