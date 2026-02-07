import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
from transformers import GPT2Tokenizer

# Import your GPT model (assuming you have the previous implementation)
from Pytry import create_model, train_step

# Since we don't have your Pytry module, I'll include the essential parts here
# You can replace this with your actual imports

def get_tokenizer():
    """Get a proper tokenizer - using GPT2 tokenizer as example"""
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # Set pad token to eos token for GPT2
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Make sure you have transformers installed: pip install transformers")
        return None

def load_tokenized_data(file_path, tokenizer, block_size):
    """Load and tokenize text data with proper error handling"""
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    except Exception as e:
        raise Exception(f"Error reading file: {e}")
    
    # Check if file is empty
    if not text:
        raise ValueError("The text file is empty!")
    
    print(f"File loaded successfully. Text length: {len(text)} characters")
    print(f"First 100 characters: {repr(text[:100])}")
    
    # Tokenize the text
    if tokenizer is None:
        raise ValueError("Tokenizer is None!")
    
    try:
        # Use the tokenizer properly
        if hasattr(tokenizer, 'encode'):
            input_ids = tokenizer.encode(text, add_special_tokens=True)
        else:
            # Fallback for custom tokenizers
            input_ids = tokenizer(text)
    except Exception as e:
        raise Exception(f"Error tokenizing text: {e}")
    
    if not input_ids:
        raise ValueError("Tokenization resulted in empty token list!")
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    tokenized_length = input_ids.size(0)
    
    print(f"Tokenized length: {tokenized_length} tokens")
    
    if tokenized_length < block_size:
        raise ValueError(f"Text too short! Need at least {block_size} tokens, got {tokenized_length}")
    
    # Create overlapping chunks for better training
    chunks = []
    for i in range(0, tokenized_length - block_size + 1, block_size // 2):  # 50% overlap
        if i + block_size <= tokenized_length:
            chunks.append(input_ids[i:i + block_size])
    
    if not chunks:
        raise ValueError("No valid chunks created!")
    
    chunks_tensor = torch.stack(chunks)
    num_chunks = len(chunks)
    
    print(f"Created {num_chunks} training chunks of size {block_size}")
    
    return chunks_tensor, num_chunks

def create_simple_model(vocab_size=50257, n_embd=256, n_layer=4, n_head=8, block_size=128):
    """Create a simple GPT model for training"""
    # This is a placeholder - replace with your actual model creation
    from transformers import GPT2Config, GPT2LMHeadModel
    
    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_positions=block_size,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    
    model = GPT2LMHeadModel(config)
    return model

def train_step_simple(model, data_loader, optimizer, device):
    """Simple training step"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (input_ids, targets) in enumerate(data_loader):
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # For language modeling, input and target are the same but shifted
        # Input: [1, 2, 3, 4, 5] -> Target: [2, 3, 4, 5, 6]
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches if num_batches > 0 else 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a GPT model on text data')
    parser.add_argument("--file", type=str, required=True, 
                       help="Path to training text file")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--block_size", type=int, default=128,
                       help="Context length for training")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    print("🚀 Starting GPT training...")
    print(f"File: {args.file}")
    print(f"Epochs: {args.epochs}")
    print(f"Block size: {args.block_size}")
    print(f"Batch size: {args.batch_size}")
    
    # Get tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = get_tokenizer()
    if tokenizer is None:
        exit(1)
    
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Load and tokenize data
    print(f"\n📚 Loading data from {args.file}...")
    try:
        data, num_chunks = load_tokenized_data(args.file, tokenizer, args.block_size)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        exit(1)
    
    if num_chunks == 0:
        print("❗ Not enough data to create any training chunks.")
        print("Try:")
        print("  - Using a larger text file")
        print("  - Reducing the block_size parameter")
        exit(1)
    
    print(f"✅ Successfully created {num_chunks} training chunks")
    
    # Create dataset and dataloader
    print("\n🔄 Creating data loader...")
    # For language modeling, we shift the input to create targets
    inputs = data[:, :-1]  # All tokens except last
    targets = data[:, 1:]  # All tokens except first
    
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Batches per epoch: {len(loader)}")
    
    # Create model
    print(f"\n🤖 Creating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Try to use your custom model first
        model = create_model(vocab_size=vocab_size)
    except:
        # Fallback to simple model
        print("Using fallback model...")
        model = create_simple_model(vocab_size=vocab_size, block_size=args.block_size-1)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Training loop
    print(f"\n🎯 Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        try:
            # Use your custom train_step if available
            loss = train_step(model, loader, optimizer, device)
        except:
            # Fallback to simple train step
            loss = train_step_simple(model, loader, optimizer, device)
        
        print(f"Epoch {epoch+1} completed. Average Loss: {loss:.4f}")
    
    # Save model
    print(f"\n💾 Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_size': vocab_size,
        'block_size': args.block_size,
        'final_loss': loss,
    }, "gpt_custom_trained.pt")
    
    print("✅ Training completed successfully!")
    print("✅ Model saved to gpt_custom_trained.pt")
    
    # Test the model with a simple generation
    print(f"\n🧪 Testing model...")
    model.eval()
    with torch.no_grad():
        # Create a small test input
        test_input = torch.randint(0, min(1000, vocab_size), (1, 10)).to(device)
        try:
            if hasattr(model, 'generate'):
                output = model.generate(test_input, max_new_tokens=5)
                print(f"Test generation successful! Output shape: {output.shape}")
            else:
                output = model(test_input)
                print(f"Test forward pass successful! Output shape: {output.logits.shape}")
        except Exception as e:
            print(f"Test failed: {e}")