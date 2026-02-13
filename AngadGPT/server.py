from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import GPT2Tokenizer

from GPTTrainer import create_model

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load model
print("Loading model...")
model = create_model(vocab_size=50257) 
checkpoint = torch.load("gpt_custom_trained.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print("Model loaded successfully!")

API_KEY = "password"  

app = FastAPI()

# Add CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class PredictionRequest(BaseModel):
    prompt: str
    api_key: str

@app.get("/")
def root():
    return {"message": "AngadGPT API is running", "endpoints": ["/generate"]}

@app.post("/generate")
def generate_text(req: PredictionRequest):
    if req.api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        # Encode the prompt using the tokenizer
        encoded = tokenizer.encode(req.prompt, return_tensors="pt")
        input_ids = encoded.to(device)
        
        # Truncate if too long (models typically have max context length)
        max_context = 128  # Adjust based on your model's block_size
        if input_ids.size(1) > max_context:
            input_ids = input_ids[:, -max_context:]
        
        with torch.no_grad():
            if hasattr(model, 'generate'):
                # Use model's generate method if available
                output_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8)
                # Decode only the newly generated tokens (skip the input prompt)
                input_length = input_ids.size(1)
                generated_ids = output_ids[0][input_length:].tolist()
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                # Fallback: generate token by token
                generated_ids = input_ids[0].tolist()
                current_input = input_ids
                
                for _ in range(50):  # Generate up to 50 new tokens
                    logits, _ = model(current_input)
                    # Get logits for the last position
                    next_token_logits = logits[0, -1, :]
                    # Sample from the distribution
                    next_token_id = torch.multinomial(
                        torch.softmax(next_token_logits / 0.8, dim=-1), 
                        num_samples=1
                    ).item()
                    
                    generated_ids.append(next_token_id)
                    
                    # Update input for next iteration
                    next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
                    current_input = torch.cat([current_input, next_token_tensor], dim=1)
                    
                    # Stop if we hit EOS token
                    if next_token_id == tokenizer.eos_token_id:
                        break
                
                # Decode only the newly generated tokens
                new_tokens = generated_ids[input_ids.size(1):]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return {"generated_text": text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")