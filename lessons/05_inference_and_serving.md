# Lesson 5: Inference and Serving Transformer Models

## Overview
This lesson covers techniques for efficient inference and deployment of transformer models. We'll explore optimization strategies, serving architectures, and best practices for production deployment.

## Learning Objectives
By the end of this lesson, you should be able to:
- Understand different text generation strategies
- Implement efficient inference techniques
- Optimize models for deployment
- Build REST APIs for model serving
- Deploy models using containerization
- Monitor and scale production deployments

## 1. Text Generation Strategies

### Greedy Decoding
Selects the token with the highest probability at each step.

```python
import torch
import torch.nn.functional as F

def greedy_decode(model, input_ids, max_length=50):
    """Generate text using greedy decoding."""
    model.eval()
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length - input_ids.shape[1]):
            outputs = model(generated)
            logits = outputs.logits
            
            # Select the token with highest probability
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    
    return generated
```

### Sampling with Temperature
Introduces randomness controlled by a temperature parameter.

```python
def sample_decode(model, input_ids, max_length=50, temperature=1.0):
    """Generate text using sampling with temperature."""
    model.eval()
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length - input_ids.shape[1]):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :] / temperature
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
    
    return generated
```

### Top-K Sampling
Restricts sampling to the K most likely tokens.

```python
def top_k_decode(model, input_ids, max_length=50, top_k=50):
    """Generate text using Top-K sampling."""
    model.eval()
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length - input_ids.shape[1]):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]
            
            # Keep only top-k tokens
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            
            # Apply softmax and sample
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = torch.gather(top_k_indices, -1, next_token_idx)
            
            generated = torch.cat([generated, next_token], dim=1)
    
    return generated
```

### Top-P (Nucleus) Sampling
Dynamically selects tokens based on cumulative probability.

```python
def top_p_decode(model, input_ids, max_length=50, top_p=0.9):
    """Generate text using Top-P (nucleus) sampling."""
    model.eval()
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length - input_ids.shape[1]):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sort probabilities
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            
            # Calculate cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 0] = False  # Keep at least one token
            
            # Create mask for removed tokens
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            
            # Set probabilities of removed tokens to zero
            filtered_probs = probs.masked_fill(indices_to_remove, 0.0)
            
            # Renormalize and sample
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(filtered_probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
    
    return generated
```

### Beam Search
Maintains multiple candidate sequences and selects the most likely overall sequence.

```python
def beam_search_decode(model, input_ids, max_length=50, num_beams=3):
    """Generate text using beam search."""
    model.eval()
    batch_size, seq_len = input_ids.shape
    
    # Initialize beam search
    beam_scores = torch.zeros((batch_size, num_beams), device=input_ids.device)
    beam_scores[:, 1:] = -1e9  # Initialize all beams except first
    
    # Expand input for beams
    input_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1)
    input_ids = input_ids.view(batch_size * num_beams, seq_len)
    
    with torch.no_grad():
        for _ in range(max_length - seq_len):
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            
            # Calculate scores
            next_token_scores = F.log_softmax(logits, dim=-1)
            next_token_scores = next_token_scores.view(batch_size, num_beams, -1)
            next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)
            
            # Flatten and select top candidates
            next_token_scores = next_token_scores.view(batch_size, num_beams * next_token_scores.size(-1))
            next_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1)
            
            # Compute beam indices and token indices
            next_beam_scores = next_scores[:, :num_beams]
            next_beam_tokens = next_tokens[:, :num_beams] % next_token_scores.size(-1)
            next_beam_indices = next_tokens[:, :num_beams] // next_token_scores.size(-1)
            
            # Update input_ids
            input_ids = input_ids.view(batch_size, num_beams, -1)
            gathered_input_ids = torch.gather(
                input_ids, 1, next_beam_indices.unsqueeze(-1).repeat(1, 1, input_ids.size(-1))
            )
            gathered_input_ids = gathered_input_ids.view(batch_size * num_beams, -1)
            input_ids = torch.cat([gathered_input_ids, next_beam_tokens.view(-1, 1)], dim=-1)
            
            # Update beam scores
            beam_scores = next_beam_scores
    
    # Return the best beam
    best_beam_idx = torch.argmax(beam_scores, dim=-1)
    input_ids = input_ids.view(batch_size, num_beams, -1)
    best_sequence = input_ids[torch.arange(batch_size), best_beam_idx]
    
    return best_sequence
```

## 2. Model Optimization for Inference

### Model Quantization
Reduce model size and improve inference speed.

```python
import torch.quantization

def quantize_model(model):
    """Quantize model for inference."""
    # Dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

# For more advanced quantization
def static_quantize_model(model, calibration_data):
    """Static quantization with calibration."""
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibration
    with torch.no_grad():
        for batch in calibration_data:
            model(batch)
    
    torch.quantization.convert(model, inplace=True)
    return model
```

### Model Compilation with TorchScript
```python
def compile_model(model, example_input):
    """Compile model with TorchScript."""
    model.eval()
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Or script the model (more flexible)
    scripted_model = torch.jit.script(model)
    
    return traced_model  # or scripted_model
```

### ONNX Export
```python
def export_to_onnx(model, example_input, filepath):
    """Export model to ONNX format."""
    model.eval()
    
    torch.onnx.export(
        model,
        example_input,
        filepath,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence'},
            'output': {0: 'batch_size', 1: 'sequence'}
        }
    )
```

## 3. Batch Processing for Efficiency

### Dynamic Batching
```python
class BatchProcessor:
    def __init__(self, model, max_batch_size=8):
        self.model = model
        self.max_batch_size = max_batch_size
    
    def pad_sequences(self, sequences):
        """Pad sequences to the same length."""
        max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            padding = [0] * (max_len - len(seq))
            padded.append(seq + padding)
        return torch.tensor(padded)
    
    def process_batch(self, input_sequences):
        """Process a batch of input sequences."""
        # Pad sequences
        input_ids = self.pad_sequences(input_sequences)
        
        # Generate outputs
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=100,
                do_sample=True,
                temperature=0.8
            )
        
        return outputs
    
    def process_requests(self, requests):
        """Process multiple requests with dynamic batching."""
        # Sort requests by length for efficient batching
        sorted_requests = sorted(enumerate(requests), key=lambda x: len(x[1]))
        
        results = [None] * len(requests)
        
        # Process in batches
        for i in range(0, len(sorted_requests), self.max_batch_size):
            batch = sorted_requests[i:i + self.max_batch_size]
            indices, sequences = zip(*batch)
            
            # Process batch
            batch_outputs = self.process_batch(sequences)
            
            # Store results in original order
            for j, idx in enumerate(indices):
                results[idx] = batch_outputs[j]
        
        return results
```

## 4. Building REST APIs with FastAPI

### Basic API Structure
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="Transformer Inference API")

# Load model and tokenizer
model_name = "gpt2"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

class TextInput(BaseModel):
    text: str
    max_length: int = 50
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95

class TextOutput(BaseModel):
    generated_text: str
    input_length: int
    output_length: int

@app.post("/generate", response_model=TextOutput)
async def generate_text(input: TextInput):
    try:
        # Tokenize input
        input_ids = tokenizer.encode(input.text, return_tensors="pt")
        
        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=input.max_length,
                temperature=input.temperature,
                top_k=input.top_k,
                top_p=input.top_p,
                do_sample=True
            )
        
        # Decode output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return TextOutput(
            generated_text=generated_text,
            input_length=len(input_ids[0]),
            output_length=len(output[0])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": model_name}
```

### Advanced API with Caching
```python
from functools import lru_cache
import time

class InferenceCache:
    def __init__(self, maxsize=1000):
        self.cache = {}
        self.maxsize = maxsize
        self.access_times = {}
    
    def get(self, key):
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if len(self.cache) >= self.maxsize:
            # Remove least recently used item
            lru_key = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()

# Global cache instance
inference_cache = InferenceCache()

@app.post("/generate_cached", response_model=TextOutput)
async def generate_text_cached(input: TextInput):
    # Create cache key
    cache_key = f"{input.text}_{input.max_length}_{input.temperature}_{input.top_k}_{input.top_p}"
    
    # Check cache
    cached_result = inference_cache.get(cache_key)
    if cached_result:
        return cached_result
    
    # Generate if not cached
    result = await generate_text(input)
    
    # Store in cache
    inference_cache.put(cache_key, result)
    
    return result
```

## 5. Containerization with Docker

### Dockerfile for Model Serving
```dockerfile
# Use NVIDIA PyTorch image for GPU support
FROM nvcr.io/nvidia/pytorch:22.08-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download model at build time (or download at runtime)
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
               AutoModelForCausalLM.from_pretrained('gpt2'); \
               AutoTokenizer.from_pretrained('gpt2')"

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose for Multi-Service Setup
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./logs:/app/logs

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

## 6. Kubernetes Deployment

### Deployment Configuration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: transformer-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: transformer-api
  template:
    metadata:
      labels:
        app: transformer-api
    spec:
      containers:
      - name: transformer-api
        image: transformer-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
```

### Service Configuration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: transformer-api-service
spec:
  selector:
    app: transformer-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: transformer-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: transformer-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 7. Monitoring and Metrics

### Prometheus Metrics Integration
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Define metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration', ['endpoint'])
MODEL_INFERENCE_TIME = Histogram('model_inference_duration_seconds', 'Model inference time')
GPU_MEMORY_USAGE = Gauge('gpu_memory_usage_bytes', 'GPU memory usage')

@app.middleware("http")
async def add_prometheus_metrics(request, call_next):
    start_time = time.time()
    endpoint = request.url.path
    method = request.method
    
    REQUEST_COUNT.labels(endpoint=endpoint, method=method).inc()
    
    response = await call_next(request)
    
    REQUEST_DURATION.labels(endpoint=endpoint).observe(time.time() - start_time)
    
    return response

@app.get("/metrics")
async def metrics():
    # Update GPU metrics
    if torch.cuda.is_available():
        GPU_MEMORY_USAGE.set(torch.cuda.memory_allocated())
    
    return generate_latest()
```

## 8. Performance Optimization

### Memory Management
```python
class MemoryManager:
    def __init__(self):
        self.peak_memory = 0
    
    def monitor_memory(self):
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            self.peak_memory = max(self.peak_memory, current_memory)
            return current_memory
        return 0
    
    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_stats(self):
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved(),
                'peak': self.peak_memory
            }
        return {}

memory_manager = MemoryManager()
```

### Asynchronous Processing
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=4)

async def async_generate_text(input: TextInput):
    """Asynchronous text generation."""
    loop = asyncio.get_event_loop()
    
    # Run CPU-bound operation in thread pool
    result = await loop.run_in_executor(
        executor, 
        generate_text_sync,  # Synchronous version of generate_text
        input
    )
    
    return result

@app.post("/generate_async", response_model=TextOutput)
async def generate_text_async(input: TextInput):
    return await async_generate_text(input)
```

## Summary

In this lesson, we explored comprehensive techniques for inference and serving transformer models:
- Various text generation strategies (greedy, sampling, beam search)
- Model optimization techniques (quantization, compilation)
- Batch processing for efficiency
- Building REST APIs with FastAPI
- Containerization with Docker
- Kubernetes deployment configurations
- Monitoring and metrics collection
- Performance optimization strategies

These techniques enable efficient deployment and serving of transformer models in production environments.

## Next Steps

In the next lesson, we'll explore Reinforcement Learning from Human Feedback (RLHF), a technique used to align language models with human preferences and values.