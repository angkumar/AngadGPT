from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import GPT2Tokenizer

from Pytry import create_model

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = create_model(vocab_size=50257) 
checkpoint = torch.load("gpt_custom_trained.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print("Model loaded successfully!")

API_KEY = "password"  

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
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
        encoded = tokenizer.encode(req.prompt, return_tensors="pt")
        input_ids = encoded.to(device)
        
        
        max_context = 128 
        if input_ids.size(1) > max_context:
            input_ids = input_ids[:, -max_context:]
        
        with torch.no_grad():
            if hasattr(model, 'generate'):
                
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=50
                )
                
                input_length = input_ids.size(1)
                generated_ids = output_ids[0][input_length:].tolist()
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                
                generated_ids = input_ids[0].tolist()
                current_input = input_ids
                
                for _ in range(50):  
                    logits, _ = model(current_input)
                    
                    next_token_logits = logits[0, -1, :]
                    
                    probs = torch.softmax(next_token_logits / 0.6, dim=-1)
                    
                    k = 40
                    values, indices = torch.topk(probs, k)
                    filtered_probs = values / values.sum()
                    next_token_id = indices[torch.multinomial(filtered_probs, 1)].item()
                    
                    generated_ids.append(next_token_id)
                    
                    next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
                    current_input = torch.cat([current_input, next_token_tensor], dim=1)
                    
                  
                    if next_token_id == tokenizer.eos_token_id:
                        break
                
                
                new_tokens = generated_ids[input_ids.size(1):]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return {"generated_text": text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")