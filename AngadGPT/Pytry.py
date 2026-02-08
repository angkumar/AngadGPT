import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional
import argparse
from transformers import GPT2Tokenizer

@dataclass
class GPTConfig:
    """Configuration for GPT model"""
    vocab_size: int = 50257  
    n_layer: int = 12       
    n_head: int = 12        
    n_embd: int = 768       
    block_size: int = 1024  
    dropout: float = 0.1
    bias: bool = True       
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
  
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                           .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size()  

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C)  
      
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
    def forward(self, x):
        # Pre-norm architecture (like GPT-2)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token embeddings
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # position embeddings
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying (optional but common)
        self.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"
        
        # Position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Token and position embeddings
        tok_emb = self.wte(idx)  # (b, t, n_embd)
        pos_emb = self.wpe(pos)  # (t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        
        # Forward through transformer blocks
        for block in self.h:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        if targets is not None:
            # Training mode - compute loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference mode - only compute logits for last token
            logits = self.lm_head(x[:, [-1], :])  # (b, 1, vocab_size)
            loss = None
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text autoregressively"""
        for _ in range(max_new_tokens):
            # Crop sequence if it gets too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Scale by temperature
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop to top k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

# Example usage and training setup
def create_model(vocab_size=50257):
    """Create a small GPT model for demonstration"""
    config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=6,      # smaller for demo
        n_head=6,       # smaller for demo  
        n_embd=384,     # smaller for demo
        block_size=256, # smaller for demo
        dropout=0.1
    )
    model = GPT(config)
    return model

def train_step(model, data_loader, optimizer, device):
    """Single training step"""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, targets) in enumerate(data_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(data, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(data_loader)

def get_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")

if __name__ == "__main__":
    from transformers import GPT2LMHeadModel
    parser = argparse.ArgumentParser(description="Run a demo GPT model.")
    parser.add_argument("--generate", type=str, default=None, help="Text prompt to generate from")
    parser.add_argument("--max_tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--use_custom_model", action="store_true", help="Use custom GPT model instead of pretrained GPT-2")

    args = parser.parse_args()

    tokenizer = get_tokenizer()

    if args.use_custom_model:
        model = create_model()
        try:
            checkpoint = torch.load("gpt_custom_trained.pt")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
                print("✅ Loaded trained weights for custom GPT model.")
        except FileNotFoundError:
            print("⚠️ No trained weights found for custom model. Using untrained custom GPT.")
    else:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    if args.generate:
        input_ids = tokenizer.encode(args.generate, return_tensors="pt")
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
    else:
        # Example forward pass
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            print(f"Output shape: {logits.shape}")