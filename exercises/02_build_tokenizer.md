# Exercise 2: Build a Tokenizer

## Objective
Implement a simple tokenizer from scratch to understand how text is converted into numerical representations for transformer models.

## Prerequisites
- Completion of Exercise 1 (Tensor Operations)
- Understanding of Python data structures
- Basic knowledge of text processing

## Instructions

### Part 1: Basic Tokenizer Implementation

1. Create a `SimpleTokenizer` class with the following methods:
   - `__init__(self, vocab_size=1000)`: Initialize with a vocabulary size
   - `fit(self, texts)`: Build vocabulary from a list of texts
   - `encode(self, text)`: Convert text to token IDs
   - `decode(self, token_ids)`: Convert token IDs back to text

2. Vocabulary building:
   - Split texts into words (simple whitespace tokenization)
   - Count word frequencies
   - Keep the most frequent words up to `vocab_size`
   - Assign unique IDs to each word (0 for padding, 1 for unknown words)

3. Encoding and decoding:
   - Implement the `encode` method to convert text to token IDs
   - Handle unknown words by mapping them to a special unknown token
   - Implement the `decode` method to convert token IDs back to text

### Part 2: Advanced Features

1. Subword tokenization (BPE-like):
   - Implement basic Byte Pair Encoding (BPE) algorithm
   - Start with character-level tokens
   - Iteratively merge most frequent pairs to build subword tokens
   - Update the tokenizer to use subword tokens

2. Special tokens:
   - Add support for special tokens:
     - `<PAD>`: Padding token
     - `<UNK>`: Unknown token
     - `<BOS>`: Beginning of sequence
     - `<EOS>`: End of sequence
   - Modify encoding to handle these special tokens

3. Padding and truncation:
   - Add parameters to `encode` method for max_length
   - Implement padding to ensure all sequences have the same length
   - Implement truncation for sequences longer than max_length

### Part 3: Integration with PyTorch

1. Batch processing:
   - Create a method to encode multiple texts into a batch
   - Ensure all sequences in the batch have the same length
   - Return a PyTorch tensor instead of a list

2. Dataset integration:
   - Create a PyTorch Dataset class that uses your tokenizer
   - Implement `__len__` and `__getitem__` methods
   - Return both input IDs and attention masks

## Challenge Problems

1. **Vocabulary Optimization**: Implement a more sophisticated vocabulary building algorithm that considers:
   - Character-level n-grams
   - Word frequency weighting
   - Handling of rare words and typos

2. **Efficiency Challenge**: Optimize your tokenizer for speed:
   - Profile the encoding/decoding process
   - Identify bottlenecks
   - Implement caching for frequently used texts

3. **Multilingual Support**: Extend your tokenizer to handle multiple languages:
   - Support Unicode characters from different scripts
   - Handle language-specific tokenization rules
   - Implement language detection for mixed-language texts

## Submission

Create a Python module `tokenizer.py` with your tokenizer implementation. Include:
- The `SimpleTokenizer` class with all required methods
- Example usage demonstrating each feature
- Unit tests for core functionality
- Documentation for each method

## Resources

- [Byte Pair Encoding Algorithm](https://arxiv.org/abs/1508.07909)
- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/)
- [SentencePiece Library](https://github.com/google/sentencepiece)