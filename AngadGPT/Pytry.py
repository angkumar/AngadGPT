#Pytry.py
import torch
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys
import time

def load_model(model_path='gpt2'):
    """Load GPT-2 model from pretrained or fine-tuned checkpoint"""
    try:
        print(f"🔄 Loading model from {model_path}...")
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {total_params:,}")
        
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Falling back to base GPT-2...")
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("✅ Base GPT-2 loaded")
        return model, tokenizer


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, 
                  top_k=50, top_p=0.95, num_return=1, show_stats=False):
    """Generate text from a prompt"""
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Move to same device as model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    
    generation_time = time.time() - start_time
    
    # Decode and print results
    results = []
    for idx, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        results.append(text)
        
        if num_return > 1:
            print(f"--- Generation {idx + 1} ---")
        print(text)
        if num_return > 1:
            print()
    
    # Show generation statistics
    if show_stats:
        tokens_generated = sum(len(o) for o in outputs) - len(input_ids[0]) * num_return
        tokens_per_sec = tokens_generated / max(generation_time, 0.001)
        print(f"\n📊 Stats:")
        print(f"   Time: {generation_time:.2f}s")
        print(f"   Tokens generated: {tokens_generated}")
        print(f"   Speed: {tokens_per_sec:.1f} tokens/sec")
    
    print(f"{'='*60}\n")
    return results


def interactive_mode(model, tokenizer, max_length=150, temperature=0.8, top_k=50, top_p=0.95):
    """Interactive chat mode with commands"""
    print("\n" + "="*60)
    print("💬 ANGADGPT INTERACTIVE MODE")
    print("="*60)
    print("Type your prompts below.")
    print("\nCommands:")
    print("  • 'quit' or 'exit' - Exit the program")
    print("  • 'clear' - Clear conversation history")
    print("  • 'params' - Show current generation parameters")
    print("  • 'temp X' - Set temperature to X (e.g., 'temp 0.9')")
    print("  • 'length X' - Set max length to X (e.g., 'length 200')")
    print("="*60 + "\n")
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            prompt = input("You: ").strip()
            
            # Handle exit commands
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Handle clear command
            if prompt.lower() == 'clear':
                conversation_history = []
                print("🗑️  Conversation history cleared.\n")
                continue
            
            # Handle params command
            if prompt.lower() == 'params':
                print(f"\n📋 Current parameters:")
                print(f"   Max length: {max_length}")
                print(f"   Temperature: {temperature}")
                print(f"   Top-k: {top_k}")
                print(f"   Top-p: {top_p}\n")
                continue
            
            # Handle temp command
            if prompt.lower().startswith('temp '):
                try:
                    new_temp = float(prompt.split()[1])
                    if 0.1 <= new_temp <= 2.0:
                        temperature = new_temp
                        print(f"🌡️  Temperature set to {temperature}\n")
                    else:
                        print("⚠️  Temperature must be between 0.1 and 2.0\n")
                except:
                    print("⚠️  Invalid temperature value\n")
                continue
            
            # Handle length command
            if prompt.lower().startswith('length '):
                try:
                    new_length = int(prompt.split()[1])
                    if 10 <= new_length <= 500:
                        max_length = new_length
                        print(f"📏 Max length set to {max_length}\n")
                    else:
                        print("⚠️  Length must be between 10 and 500\n")
                except:
                    print("⚠️  Invalid length value\n")
                continue
            
            # Skip empty prompts
            if not prompt:
                continue
            
            print()  # blank line before generation
            
            # Generate response
            responses = generate_text(
                model, tokenizer, prompt, 
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return=1,
                show_stats=False
            )
            
            # Store in conversation history (optional for context)
            conversation_history.append(f"You: {prompt}")
            conversation_history.append(f"AI: {responses[0]}")
            
            # Keep only last 10 exchanges to avoid context overflow
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def batch_generate(model, tokenizer, prompts_file, output_file, max_length=100, 
                   temperature=0.8, top_k=50, top_p=0.95, show_stats=False):
    """Generate text for multiple prompts from a file"""
    print(f"\n📄 Batch generation from {prompts_file}")
    
    # Read prompts
    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"❌ Error reading prompts file: {e}")
        return
    
    print(f"Found {len(prompts)} prompts\n")
    
    # Generate for each prompt
    results = []
    total_start = time.time()
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Generating for: {prompt[:50]}...")
        outputs = generate_text(
            model, tokenizer, prompt, 
            num_return=1,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            show_stats=False
        )
        results.append({
            'prompt': prompt,
            'generated': outputs[0]
        })
    
    total_time = time.time() - total_start
    
    # Save results
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Batch Generation Results\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total prompts: {len(prompts)}\n")
            f.write(f"Total time: {total_time:.2f}s\n")
            f.write("="*60 + "\n\n")
            
            for i, r in enumerate(results, 1):
                f.write(f"[{i}] PROMPT: {r['prompt']}\n")
                f.write(f"GENERATED: {r['generated']}\n")
                f.write("-" * 60 + "\n\n")
        
        print(f"\n✅ Results saved to {output_file}")
        print(f"⏱️  Total time: {total_time:.2f}s ({total_time/len(prompts):.2f}s per prompt)")
    except Exception as e:
        print(f"❌ Error saving results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text with GPT-2 or fine-tuned AngadGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with your fine-tuned model
  python Pytry.py --model_path myangadgpt_model --interactive
  
  # Generate from a single prompt
  python Pytry.py --model_path myangadgpt_model --generate "Hello world"
  
  # Use base GPT-2 (no fine-tuning)
  python Pytry.py --generate "The future of AI is"
  
  # Batch generation from file
  python Pytry.py --model_path myangadgpt_model --batch prompts.txt --output results.txt
  
  # Generate with custom parameters
  python Pytry.py --model_path myangadgpt_model --generate "Once upon a time" --temperature 1.2 --max_length 200
        """
    )
    
    # Model selection
    parser.add_argument("--model_path", type=str, default="gpt2",
                        help="Path to model directory (default: gpt2)")
    
    # Generation modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--generate", type=str, default=None,
                           help="Text prompt to generate from")
    mode_group.add_argument("--interactive", action="store_true",
                           help="Run in interactive chat mode")
    mode_group.add_argument("--batch", type=str, default=None,
                           help="Batch generate from prompts file (one per line)")
    
    # Generation parameters
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated text (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature - higher = more random (default: 0.8)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling (default: 50)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling (default: 0.95)")
    parser.add_argument("--num_return", type=int, default=1,
                        help="Number of different generations (only for --generate)")
    
    # Output options
    parser.add_argument("--output", type=str, default="batch_output.txt",
                        help="Output file for batch generation")
    parser.add_argument("--stats", action="store_true",
                        help="Show generation statistics (tokens/sec, etc.)")
    parser.add_argument("--device", type=str, default=None,
                        choices=['cpu', 'mps', 'cuda'],
                        help="Force specific device (default: auto-detect)")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🚀 ANGADGPT TEXT GENERATION")
    print("="*60 + "\n")

    # Load model
    model, tokenizer = load_model(args.model_path)
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
        print(f"🖥️  Using forced device: {device}")
    else:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"🖥️  Using device: MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🖥️  Using device: CUDA GPU")
        else:
            device = torch.device('cpu')
            print(f"🖥️  Using device: CPU")
    
    model = model.to(device)
    print()

    # Interactive mode
    if args.interactive:
        interactive_mode(
            model, tokenizer,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
    
    # Batch mode
    elif args.batch:
        batch_generate(
            model, tokenizer,
            args.batch,
            args.output,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            show_stats=args.stats
        )
    
    # Single generation mode
    elif args.generate:
        generate_text(
            model, tokenizer,
            args.generate,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return=args.num_return,
            show_stats=args.stats
        )
    
    # Demo mode (no arguments provided)
    else:
        print("No mode specified. Running demo generations...\n")
        print("💡 Tip: Use --help to see all options\n")
        
        demo_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology",
            "The most important skill to learn is",
        ]
        
        for prompt in demo_prompts:
            generate_text(
                model, tokenizer,
                prompt,
                max_length=80,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_return=1,
                show_stats=args.stats
            )
        
        print("💡 Try these commands:")
        print(f"  python Pytry.py --model_path {args.model_path} --interactive")
        print(f"  python Pytry.py --model_path {args.model_path} --generate 'Your prompt here'")
        print()