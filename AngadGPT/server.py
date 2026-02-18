#server.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
import os
from datetime import datetime

# Configuration
API_KEY = os.getenv("ANGADGPT_API_KEY", "password")
MODEL_PATH = os.getenv("ANGADGPT_MODEL_PATH", "myangadgpt_model")
HOST = os.getenv("ANGADGPT_HOST", "127.0.0.1")
PORT = int(os.getenv("ANGADGPT_PORT", "8000"))

# Setup device
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("\n" + "="*60)
print("🚀 ANGADGPT API SERVER")
print("="*60)
print(f"🖥️  Device: {device}")
print(f"📁 Model path: {MODEL_PATH}")
print(f"🔑 API Key: {'*' * len(API_KEY)}")
print("="*60 + "\n")

# Load model
print("📦 Loading model and tokenizer...")
start_time = time.time()

try:
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✅ Loaded fine-tuned model from {MODEL_PATH}")
    print(f"   Parameters: {total_params:,}")
except Exception as e:
    print(f"⚠️  Could not load fine-tuned model: {e}")
    print("📦 Falling back to base GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("✅ Loaded base GPT-2")

model.to(device)
model.eval()

load_time = time.time() - start_time
print(f"⏱️  Model loaded in {load_time:.2f}s")
print(f"🌐 Server will start on http://{HOST}:{PORT}")
print("="*60 + "\n")

# FastAPI app
app = FastAPI(
    title="AngadGPT API",
    description="Fine-tuned GPT-2 text generation API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="Input text prompt")
    api_key: str = Field(..., description="API authentication key")
    max_length: int = Field(100, ge=10, le=500, description="Maximum length of generated text")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling parameter")
    top_p: float = Field(0.95, ge=0.1, le=1.0, description="Top-p (nucleus) sampling parameter")
    return_prompt: bool = Field(False, description="Include prompt in response")

class GenerateResponse(BaseModel):
    generated_text: str
    prompt: Optional[str] = None  # FIXED: Made optional
    tokens_generated: Optional[int] = None  # FIXED: Made optional
    generation_time: Optional[float] = None  # FIXED: Made optional
    model: str = MODEL_PATH

# Statistics tracking
stats = {
    "requests": 0,
    "successful": 0,
    "failed": 0,
    "total_tokens": 0,
    "start_time": datetime.now().isoformat()
}

# Middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    
    return response

# Routes
@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "AngadGPT API is running",
        "version": "1.0.0",
        "model": MODEL_PATH,
        "device": device,
        "endpoints": {
            "/": "API information",
            "/generate": "Generate text (POST)",
            "/health": "Health check",
            "/stats": "API statistics"
        },
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "uptime": (datetime.now() - datetime.fromisoformat(stats["start_time"])).total_seconds()
    }

@app.get("/stats")
def get_stats():
    """Get API usage statistics"""
    return {
        "statistics": stats,
        "uptime_seconds": (datetime.now() - datetime.fromisoformat(stats["start_time"])).total_seconds()
    }

@app.post("/generate", response_model=GenerateResponse)
def generate_text(req: GenerateRequest):
    """Generate text from a prompt"""
    stats["requests"] += 1
    
    # Validate API key
    if req.api_key != API_KEY:
        stats["failed"] += 1
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        generation_start = time.time()
        
        # Encode prompt
        input_ids = tokenizer.encode(req.prompt, return_tensors="pt").to(device)
        input_length = input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=min(req.max_length + input_length, 512),
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        
        generation_time = time.time() - generation_start
        
        # Decode
        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove prompt from output if not requested
        if req.return_prompt:
            generated_text = full_text
        else:
            if full_text.startswith(req.prompt):
                generated_text = full_text[len(req.prompt):].strip()
            else:
                generated_text = full_text
        
        # Calculate tokens generated
        tokens_generated = output.shape[1] - input_length
        
        # Update stats
        stats["successful"] += 1
        stats["total_tokens"] += tokens_generated
        
        # Return response with optional fields properly set
        return GenerateResponse(
            generated_text=generated_text,
            prompt=req.prompt if req.return_prompt else None,
            tokens_generated=tokens_generated,
            generation_time=round(generation_time, 3),
            model=MODEL_PATH
        )
        
    except Exception as e:
        stats["failed"] += 1
        print(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting server...")
    print(f"📖 API docs available at: http://{HOST}:{PORT}/docs")
    print(f"🔍 Alternative docs at: http://{HOST}:{PORT}/redoc")
    print("\n⚠️  Press CTRL+C to stop the server\n")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )