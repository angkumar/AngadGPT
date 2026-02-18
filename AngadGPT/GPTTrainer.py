#GPTTrainer.py
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from datetime import datetime
from tqdm import tqdm
import time
import gc

def get_tokenizer():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None


class TextDataset(Dataset):
    """Dataset for GPT-2 fine-tuning"""
    def __init__(self, texts, tokenizer, block_size=128):  # Reduced from 512
        self.examples = []
        
        print(f"Tokenizing {len(texts)} text chunks...")
        for text in tqdm(texts, desc="Tokenizing"):
            tokenized = tokenizer.encode(text, add_special_tokens=True, max_length=block_size, truncation=True)
            
            if len(tokenized) > 10:
                if len(tokenized) < block_size:
                    tokenized = tokenized + [tokenizer.pad_token_id] * (block_size - len(tokenized))
                self.examples.append(tokenized)
        
        print(f"✅ Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def load_text_or_parquet_data(file_path, chunk_size=500):  # Smaller chunks
    """Load text data from parquet or text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_lower = file_path.lower()
    
    # Try parquet first
    if file_lower.endswith('.parquet') or file_lower.endswith('.pq'):
        try:
            df = pd.read_parquet(file_path)
            if 'text' not in df.columns:
                raise ValueError("Parquet file must have a 'text' column")
            texts = df['text'].astype(str).tolist()
            print(f"✅ Loaded {len(texts)} texts from parquet")
            return texts
        except Exception as e:
            print(f"Failed to read as parquet: {e}")
    
    # Try plain text
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"✅ Loaded plain text file ({len(text):,} characters)")
        # Split into smaller chunks for faster training
        texts = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        print(f"Split into {len(texts)} chunks of ~{chunk_size} characters")
        return texts
    except Exception as e:
        raise Exception(f"Could not read file as text or parquet: {e}")


def train_epoch(model, loader, optimizer, scheduler, device, epoch, accumulation_steps=8):
    """Train one epoch with gradient accumulation and speed optimizations"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch}")
    
    optimizer.zero_grad()
    start_time = time.time()
    batches_processed = 0
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch, labels=batch)
            loss = outputs.loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights after accumulation steps
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                batches_processed += 1
            
            total_loss += loss.item() * accumulation_steps
            
            # Calculate speed
            elapsed = time.time() - start_time
            if elapsed > 0:
                batches_per_sec = batches_processed / elapsed
                eta_seconds = (len(loader) - batch_idx) / max(batches_per_sec * accumulation_steps, 0.001)
                eta_mins = eta_seconds / 60
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * accumulation_steps:.4f}',
                    'ETA': f'{eta_mins:.1f}m'
                })
            
            # Clear cache periodically
            if batch_idx % 20 == 0 and device.type == 'mps':
                torch.mps.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️ OOM at batch {batch_idx}. Clearing cache...")
                if device.type == 'mps':
                    torch.mps.empty_cache()
                gc.collect()
                optimizer.zero_grad()
                continue
            else:
                raise e
    
    # Final update if there are remaining gradients
    if (len(loader) % accumulation_steps) != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(loader)
    
    print(f"⏱️  Epoch completed in {epoch_time:.1f}s ({epoch_time/60:.1f} minutes)")
    
    return avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune GPT-2 on your dataset (FAST)')
    parser.add_argument('--file', type=str, required=True,
                        help='Path to training data (parquet or text)')
    parser.add_argument('--model', type=str, default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large'],
                        help='Which GPT-2 model to use (gpt2 = 124M params)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--block_size', type=int, default=128,
                        help='Maximum sequence length (smaller = faster)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (keep at 1 for Mac)')
    parser.add_argument('--accumulation_steps', type=int, default=16,
                        help='Gradient accumulation steps (effective batch = batch_size * this)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--chunk_size', type=int, default=500,
                        help='Text chunk size for splitting (smaller = more examples, faster per epoch)')
    parser.add_argument('--output', type=str, default='angadgpt_finetuned',
                        help='Output directory for the model')
    parser.add_argument('--save_every', type=int, default=0,
                        help='Save every N epochs (0 = only save best)')
    
    args = parser.parse_args()

    print('\n' + '='*60)
    print('🚀 ANGADGPT FAST FINE-TUNING')
    print('='*60)
    print(f'Base model:       {args.model}')
    print(f'Training file:    {args.file}')
    print(f'Epochs:           {args.epochs}')
    print(f'Block size:       {args.block_size} (shorter = faster)')
    print(f'Batch size:       {args.batch_size}')
    print(f'Accum steps:      {args.accumulation_steps} (effective batch: {args.batch_size * args.accumulation_steps})')
    print(f'Chunk size:       {args.chunk_size} chars')
    print('='*60 + '\n')

    # Load tokenizer
    print('📝 Loading tokenizer...')
    tokenizer = get_tokenizer()
    if tokenizer is None:
        exit(1)

    # Load data
    print(f'\n📂 Loading dataset from {args.file}...')
    texts = load_text_or_parquet_data(args.file, chunk_size=args.chunk_size)
    
    if len(texts) == 0:
        print("❌ No text data found!")
        exit(1)

    # Create dataset
    print('\n🔨 Creating dataset...')
    dataset = TextDataset(texts, tokenizer, block_size=args.block_size)
    
    if len(dataset) == 0:
        print("❌ No valid training examples!")
        exit(1)
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0
    )

    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f'\n🖥️  Using device: MPS (Apple Silicon GPU)')
    else:
        device = torch.device('cpu')
        print(f'\n🖥️  Using device: CPU')

    # Load pretrained GPT-2
    print(f'\n🤖 Loading pretrained {args.model} model...')
    model = GPT2LMHeadModel.from_pretrained(args.model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'   Parameters: {total_params:,}')
    
    model = model.to(device)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Adjust warmup for faster training
    warmup_steps = min(50, len(loader) // 2)
    total_steps = len(loader) * args.epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    print('\n' + '='*60)
    print('🎯 STARTING TRAINING')
    print('='*60)
    
    best_loss = float('inf')
    training_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f'\n📊 Epoch {epoch}/{args.epochs}')
        print('-' * 60)
        
        epoch_start = time.time()
        loss = train_epoch(model, loader, optimizer, scheduler, device, epoch, args.accumulation_steps)
        epoch_time = time.time() - epoch_start
        
        print(f'✅ Average Loss: {loss:.4f}')
        
        # Save if best
        if loss < best_loss:
            improvement = ((best_loss - loss) / best_loss * 100) if best_loss != float('inf') else 0
            print(f'🎉 New best loss! (improved {improvement:.1f}%)')
            best_loss = loss
            
            print(f'💾 Saving model...')
            os.makedirs(args.output, exist_ok=True)
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(args.output, 'checkpoint.pt'))
            
            print(f'✅ Saved to {args.output}/')
        
        # Periodic save
        if args.save_every > 0 and epoch % args.save_every == 0:
            print(f'💾 Periodic save (epoch {epoch})...')
            os.makedirs(f"{args.output}_epoch{epoch}", exist_ok=True)
            model.save_pretrained(f"{args.output}_epoch{epoch}")
            tokenizer.save_pretrained(f"{args.output}_epoch{epoch}")
        
        # Estimate remaining time
        elapsed_total = time.time() - training_start
        avg_epoch_time = elapsed_total / epoch
        remaining_epochs = args.epochs - epoch
        eta_minutes = (avg_epoch_time * remaining_epochs) / 60
        
        if remaining_epochs > 0:
            print(f'⏳ Estimated time remaining: {eta_minutes:.1f} minutes')

    # Final summary
    total_time = time.time() - training_start
    print('\n' + '='*60)
    print('✅ TRAINING COMPLETE!')
    print('='*60)
    print(f'Total time:       {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)')
    print(f'Best loss:        {best_loss:.4f}')
    print(f'Final loss:       {loss:.4f}')
    print(f'Model saved to:   {args.output}/')
    print('='*60)
    print('\n📖 To test your model:')
    print(f'  python Pytry.py --model_path {args.output} --interactive')
    print(f'  python Pytry.py --model_path {args.output} --generate "Your prompt"')
    print('='*60 + '\n')