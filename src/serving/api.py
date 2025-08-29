"""
API Serving Script
==================
"""

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import sys
from pathlib import Path
import asyncio
import time
import logging
from contextlib import asynccontextmanager

sys.path.append(str(Path(__file__).parent.parent))

from model.mini_transformer import MiniTransformer, MiniTransformerConfig
from inference.run_inference import TextGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for model and generator
model = None
generator = None
device = None

class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 50
    temperature: Optional[float] = 1.0
    do_sample: Optional[bool] = True
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95

class GenerationResponse(BaseModel):
    generated_text: str
    input_prompt: str
    model_info: str
    inference_time: float
    tokens_per_second: float

class PredictionRequest(BaseModel):
    text: str
    top_k: Optional[int] = 5

class PredictionResponse(BaseModel):
    text: str
    predictions: List[dict]
    model_info: str
    inference_time: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    global model, generator, device
    
    logger.info("Loading model...")
    
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
    
    # Move to device
    model.to(device)
    
    # Compile model for better performance (if CUDA available and supported)
    if device.type == "cuda" and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logger.info("Model compiled for better performance")
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")
    
    # Create text generator
    generator = TextGenerator(model, device=device)
    
    logger.info(f"Model loaded successfully on {device}")
    
    yield  # Application is running
    
    # Shutdown event
    logger.info("Shutting down API server...")
    # Clean up resources
    global model, generator
    if model is not None:
        del model
    if generator is not None:
        del generator
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="LLM Demo API",
    description="API for serving language model inference",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {
        "message": "LLM Demo API",
        "description": "API for serving language model inference",
        "endpoints": {
            "/generate": "Text generation",
            "/predict": "Next token predictions",
            "/health": "Health check",
            "/model-info": "Model information"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None, "device": str(device)}

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    if generator is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        start_time = time.time()
        generated_text = generator.generate_text(
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            do_sample=request.do_sample,
            top_k=request.top_k,
            top_p=request.top_p
        )
        end_time = time.time()
        
        inference_time = end_time - start_time
        tokens_generated = len(generated_text) - len(request.prompt)
        tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
        
        return GenerationResponse(
            generated_text=generated_text,
            input_prompt=request.prompt,
            model_info="Mini Transformer (256 hidden size, 4 layers)",
            inference_time=inference_time,
            tokens_per_second=tokens_per_second
        )
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_next_token(request: PredictionRequest):
    if generator is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        start_time = time.time()
        predictions = generator.get_next_token_predictions(
            request.text,
            top_k=request.top_k
        )
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        formatted_predictions = []
        for i, (token, prob) in enumerate(zip(predictions["tokens"], predictions["probabilities"])):
            formatted_predictions.append({
                "rank": i + 1,
                "token_id": token,
                "probability": prob,
                "token_char": chr(token % 128)
            })
        
        return PredictionResponse(
            text=request.text,
            predictions=formatted_predictions,
            model_info="Mini Transformer (256 hidden size, 4 layers)",
            inference_time=inference_time
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
async def model_info():
    if model is None:
        return {"message": "No model loaded"}
    
    config = model.config
    num_params = sum(p.numel() for p in model.parameters())
    
    return {
        "model_type": "Mini Transformer",
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "total_parameters": num_params,
        "parameter_scale": f"{num_params / 1e6:.2f}M parameters",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None
    }

def main():
    logger.info("Starting LLM Demo API server...")
    logger.info("Documentation available at http://localhost:8000/docs")
    
    # Run with optimized settings
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for production
        workers=4,     # Use multiple workers
        log_level="info",
        root_path=str(Path(__file__).parent),
        # Additional performance optimizations
        loop="uvloop",     # Use uvloop for better performance
        http="httptools",  # Use httptools for better performance
        limit_concurrency=1000,  # Limit concurrent connections
        backlog=2048,      # Connection backlog
        timeout_keep_alive=5  # Keep-alive timeout
    )

if __name__ == "__main__":
    main()