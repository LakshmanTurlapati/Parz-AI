#!/usr/bin/env python3
"""
Parz - All-in-One Script for Finetuning, Inference, and GGUF Conversion

Usage:
  python parz.py finetune                           # Train the model
  python parz.py inference --interactive            # Run interactive inference
  python parz.py inference --prompt "Hello"         # Run single prompt inference
  python parz.py convert --quantize                 # Convert to GGUF with quantization
  python parz.py host --port 8000                   # Host model as a local API server

All operations handle pad token and attention mask properly to avoid MPS/CUDA issues.
"""

import os
import sys
import json
import logging
import argparse
import re
import subprocess
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers.trainer_callback import TrainerCallback
import random
# Import Flask for API server
from flask import Flask, request, jsonify
from flask_cors import CORS
import socket

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variables for proper MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Global configuration
CONFIG = {
    "model_name": "HuggingFaceTB/SmolLM-360M",  # Base model
    "dataset_path": "Dataset/lakshman_user_assistant.json",  # Dataset path
    "output_dir": "./parz_final",  # Output directory for finetuned model
    "gguf_dir": "./parz_gguf",     # Output directory for GGUF models
    "ctx_size": 4096,              # Context size for GGUF
    "persona_name": "Parz"         # Name of the persona
}

#######################
# SHARED FUNCTIONS
#######################

def get_device(specified_device=None):
    """Determine the best available device"""
    if specified_device:
        device = specified_device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    return device

def load_model(model_path, device, training=False):
    """Load the model and tokenizer with proper token setup"""
    logger.info(f"Loading model from {model_path}")
    
    # Verify model path exists if not training
    if not training and not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        local_files_only=not training,
        trust_remote_code=True
    )
    
    # Set up special tokens including a dedicated pad token
    special_tokens = {
        "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|endoftext|>"]
    }
    
    # Set up a distinct pad token that's different from the eos token
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        special_tokens["pad_token"] = "<|pad|>"
    
    # Add special tokens to tokenizer
    num_added = tokenizer.add_special_tokens(special_tokens)
    logger.info(f"Added {num_added} special tokens to tokenizer")
    logger.info(f"Pad token: {tokenizer.pad_token}, EOS token: {tokenizer.eos_token}")
    
    # Load model on CPU first to safely handle token embedding resize
    logger.info("Loading model on CPU first for safe token embedding resize")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu"  # Force CPU for initialization
    )
    
    # Resize token embeddings on CPU to avoid MPS/CUDA issues
    if model.get_input_embeddings().weight.shape[0] < len(tokenizer):
        logger.info("Resizing token embeddings on CPU")
        model.resize_token_embeddings(len(tokenizer))
    
    # Move model to target device after embeddings are resized
    logger.info(f"Moving model to {device}")
    model = model.to(device)
    
    return model, tokenizer

#######################
# FINETUNE FUNCTIONS
#######################

def load_dataset(file_path):
    """Load the dataset from JSON file"""
    logger.info(f"Loading dataset from {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} records from dataset")
    return data

def format_conversations(data):
    """Format conversations with user/assistant markers"""
    formatted_data = []
    
    for item in data:
        messages = item.get("messages", [])
        
        if not messages or len(messages) < 2:
            logger.warning(f"Skipping record with insufficient messages")
            continue
            
        formatted_text = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "user":
                formatted_text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted_text += f"<|assistant|>\n{content}\n"
        
        # Add EOS token at the end of each conversation
        formatted_text += "<|endoftext|>"
        formatted_data.append({"text": formatted_text})
    
    logger.info(f"Formatted {len(formatted_data)} conversations")
    return formatted_data

def train_model(model, tokenizer, dataset, output_dir, device):
    """Train the model on the dataset"""
    # Create dataset object
    train_dataset = Dataset.from_list(dataset)
    
    # Calculate max length
    sample_lengths = [len(tokenizer.encode(item["text"])) for item in dataset[:10]]
    max_length = min(4096, max(sample_lengths) + 100)  # Add padding
    logger.info(f"Using max sequence length: {max_length}")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True
        )
    
    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
    logger.info("Dataset tokenized successfully")
    
    # Setup training arguments
    training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=200,  # Very high number of epochs to overfit on small data
    per_device_train_batch_size=1,  # Small batch size to memorize details
    gradient_accumulation_steps=1,  # No accumulation needed for small dataset
    learning_rate=1e-4,  # Slightly higher learning rate to fit aggressively
    weight_decay=0.0,  # No weight decay to avoid regularization
    logging_steps=5,  # Frequent logging to monitor overfitting
    save_steps=100,  # Save often to capture peak overfitting
    save_total_limit=2,  # Keep only a couple of checkpoints
    fp16=device == "cuda",  # Mixed precision on CUDA for speed
    logging_first_step=True,
    lr_scheduler_type="constant",  # Constant LR to maintain aggressive fitting
    warmup_steps=0,  # No warmup to start fitting immediately
    max_steps=-1,  # Let epochs dictate training length
)
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer with custom callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[EpochTestCallback(model, tokenizer, device)]
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training complete!")
    return model, tokenizer

def test_model(model, tokenizer, device):
    """Run a simple test of the trained model with improved EOS handling"""
    logger.info("Testing model with sample prompt...")
    
    test_prompt = "What's your favorite game?"
    formatted_prompt = f"<|user|>\n{test_prompt}\n<|assistant|>\n"
    
    # Tokenize prompt
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt",
        return_attention_mask=True
    ).to(device)
    
    # Define special tokens to stop generation
    stop_tokens = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|user|>"),
        tokenizer.convert_tokens_to_ids("<|pad|>") if "<|pad|>" in tokenizer.get_vocab() else None
    ]
    # Filter out None values
    stop_tokens = [t for t in stop_tokens if t is not None]
    
    # Generate response with enhanced parameters to prevent endless generation
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.2,
            eos_token_id=stop_tokens,
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=3,  # Prevent repetition of 3-grams
            early_stopping=True      # Stop when EOS is generated
        )
    
    # Decode and extract response
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    try:
        # Try to extract between <|assistant|> and next special token
        assistant_text = full_output.split("<|assistant|>\n")[1]
        for marker in ["<|user|>", "<|endoftext|>", "<|pad|>"]:
            if marker in assistant_text:
                assistant_text = assistant_text.split(marker)[0]
                
        response = assistant_text.strip()
        logger.info(f"Test question: {test_prompt}")
        logger.info(f"Model response: {response}")
    except:
        logger.info(f"Raw output: {full_output}")
        
    return response

# Add the epoch test callback
class EpochTestCallback(TrainerCallback):
    """Callback to test the model at the end of each epoch with random questions."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.test_questions = [
            "What's your name?",
            "Tell me about yourself.",
            "What do you like to do?",
            "How can you help me?",
            "What are your interests?",
            "What's your favorite book?",
            "How would you describe your personality?",
            "What skills do you have?",
            "Can you tell me a joke?",
            "What's the weather like today?"
        ]
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Test the model at the end of each epoch with a random question."""
        # Only test every 2 epochs to save time
        if state.epoch % 2 != 0 and state.epoch > 0:
            return
        
        # Choose a random question
        question = random.choice(self.test_questions)
        
        logger.info(f"\n\n{'='*50}\nTesting model at epoch {state.epoch} with question: {question}\n{'='*50}")
        
        # Generate response
        response = generate_response(self.model, self.tokenizer, question, self.device, max_tokens=50)
        
        logger.info(f"Question: {question}")
        logger.info(f"Response: {response}")
        logger.info(f"{'='*50}\n\n")
        
        return control

def finetune_model():
    """Finetune the model on the dataset"""
    # Get device
    device = get_device()
    
    # Load and format dataset
    dataset = load_dataset(CONFIG["dataset_path"])
    formatted_data = format_conversations(dataset)
    
    # Prepare model and tokenizer
    model, tokenizer = load_model(CONFIG["model_name"], device, training=True)
    
    # Train model
    model, tokenizer = train_model(model, tokenizer, formatted_data, CONFIG["output_dir"], device)
    
    # Test model
    test_model(model, tokenizer, device)

#######################
# INFERENCE FUNCTIONS
#######################

def generate_response(model, tokenizer, user_input, device, max_tokens=100):
    """Generate a response from the model with enhanced EOS handling"""
    # Format prompt with user/assistant markers
    prompt = f"<|user|>\n{user_input.strip()}\n<|assistant|>\n"
    logger.info(f"Generating response for: {user_input}")
    
    # Tokenize with attention mask
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,  # No padding for single prompt
        return_attention_mask=True
    ).to(device)
    
    # Define all possible stop tokens
    stop_tokens = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|user|>"),
        tokenizer.convert_tokens_to_ids("<|pad|>") if "<|pad|>" in tokenizer.get_vocab() else None
    ]
    # Filter out None values
    stop_tokens = [t for t in stop_tokens if t is not None]
    
    logger.info(f"Using stop tokens: {stop_tokens}")
    
    # Generate output with enhanced parameters
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_tokens,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,  # Prevent repetition of 3-grams
            early_stopping=True      # Stop when EOS is generated
        )
    
    # Decode and extract response
    full_output = tokenizer.decode(output[0], skip_special_tokens=False)
    logger.debug(f"Full generation output: {full_output}")
    
    # Extract assistant's response using multiple methods
    # Method 1: Extract using regex with multiple stop patterns
    pattern = r"<\|assistant\|>\n(.*?)(?:<\|user\|>|<\|endoftext\|>|<\|pad\|>|\n\n\n|\?(?:\s|$)|$)"
    match = re.search(pattern, full_output, re.DOTALL)
    
    if match:
        response = match.group(1).strip()
    else:
        # Method 2: Split on markers and clean
        try:
            assistant_text = full_output.split("<|assistant|>\n")[1]
            # Remove any trailing special tokens
            for marker in ["<|user|>", "<|endoftext|>", "<|pad|>"]:
                if marker in assistant_text:
                    assistant_text = assistant_text.split(marker)[0]
            
            # Clean up any trailing partial sentences or repeated patterns
            response = assistant_text.strip()
            
            # Check for repetitive patterns that indicate endless generation
            words = response.split()
            if len(words) > 10:
                # Check for repetition in the last third of the response
                last_third = words[len(words)//3*2:]
                if len(set(last_third)) < len(last_third) / 2:
                    # High repetition, truncate
                    response = " ".join(words[:len(words)//3*2])
                    logger.info("Detected repetition, truncated response")
        except:
            # Fallback: take everything after assistant marker
            response = full_output.split("<|assistant|>\n")[-1].strip()
            # Remove anything after common question starters
            for starter in ["What's", "What is", "How do", "Why", "?"]:
                if f" {starter}" in response:
                    response = response.split(f" {starter}")[0].strip()
    
    # Final cleanup - if the response ends with a partial sentence, try to find a good cutoff point
    if len(response) > 20 and not response.endswith((".", "!", "?")):
        # Try to end at the last complete sentence
        last_period = max(response.rfind(". "), response.rfind("! "), response.rfind("? "))
        if last_period > len(response) * 0.5:  # Only truncate if we keep at least 50% of the response
            response = response[:last_period+1].strip()
    
    return response

def interactive_mode(model, tokenizer, device, max_tokens=100):
    """Run the model in interactive mode"""
    logger.info("Starting interactive mode")
    print(f"\n===== {CONFIG['persona_name']} Digital Persona =====")
    print("Ask me anything! Type 'exit' to quit.\n")
    
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Generate and display response
        response = generate_response(model, tokenizer, user_input, device, max_tokens)
        print(f"\n{CONFIG['persona_name']}: {response}\n")

def run_inference(args):
    """Run inference with the model"""
    # Set up device
    device = get_device(args.device)
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path, device)
    
    # Run inference
    if args.prompt:
        # One-shot inference
        response = generate_response(model, tokenizer, args.prompt, device, args.max_tokens)
        print(f"\nQ: {args.prompt}")
        print(f"A: {response}\n")
    elif args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer, device, args.max_tokens)
    else:
        logger.info("No prompt or interactive mode specified, defaulting to interactive mode")
        interactive_mode(model, tokenizer, device, args.max_tokens)

#######################
# GGUF CONVERSION FUNCTIONS
#######################

def ensure_llama_cpp(llama_cpp_dir):
    """Ensure llama.cpp is available and up to date"""
    logger.info("Setting up llama.cpp...")
    
    if os.path.exists(llama_cpp_dir):
        logger.info(f"Found existing llama.cpp at {llama_cpp_dir}")
        # Pull latest changes if it's a git repo
        if os.path.exists(os.path.join(llama_cpp_dir, ".git")):
            logger.info("Updating llama.cpp repository")
            try:
                subprocess.run(["git", "-C", llama_cpp_dir, "pull"], check=True)
            except subprocess.CalledProcessError:
                logger.warning("Failed to update llama.cpp, using existing version")
    else:
        # Clone the repository
        logger.info("Cloning llama.cpp repository")
        try:
            subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir], check=True)
        except subprocess.CalledProcessError:
            raise ValueError("Failed to clone llama.cpp repository")
    
    # Check for convert_hf_to_gguf.py directly in the repository
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if os.path.exists(convert_script):
        logger.info(f"Found conversion script at {convert_script}")
        return convert_script
    
    # Check other potential locations
    potential_convert_scripts = [
        os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py"),
        os.path.join(llama_cpp_dir, "convert.py"),
        os.path.join(llama_cpp_dir, "convert", "convert_hf_to_gguf.py"),
        os.path.join(llama_cpp_dir, "examples", "convert_hf_to_gguf.py"),
        os.path.join(llama_cpp_dir, "models", "convert_hf_to_gguf.py")
    ]
    
    for script in potential_convert_scripts:
        if os.path.exists(script):
            logger.info(f"Found conversion script at {script}")
            return script
    
    # If we get here, we couldn't find the script
    logger.error(f"Could not find convert_hf_to_gguf.py in {llama_cpp_dir}")
    logger.info("Checking for alternate paths...")
    
    # List all python files that might be the conversion script
    try:
        result = subprocess.run(
            ["find", llama_cpp_dir, "-name", "*.py", "-type", "f"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        py_files = result.stdout.strip().split('\n')
        
        # Filter for likely conversion scripts
        conversion_candidates = [f for f in py_files if 'convert' in f.lower() and 'gguf' in f.lower()]
        
        if conversion_candidates:
            logger.info(f"Found potential conversion scripts: {conversion_candidates}")
            return conversion_candidates[0]  # Use the first candidate
        
        logger.error("No conversion script candidates found")
        
    except subprocess.CalledProcessError:
        logger.error("Failed to search for Python files")
    
    # Try using the llama-cpp-python module's built-in converter
    try:
        logger.info("Attempting to use llama-cpp-python's built-in converter...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "llama-cpp-python"], check=True)
        
        # Check if the module can be imported
        import importlib
        spec = importlib.util.find_spec("llama_cpp.convert_hf_to_gguf")
        if spec is not None:
            logger.info("Found llama_cpp.convert_hf_to_gguf module")
            # Create a dummy script that just imports and runs the module
            converter_script = os.path.join(llama_cpp_dir, "convert_wrapper.py")
            with open(converter_script, 'w') as f:
                f.write("""
import sys
from llama_cpp.convert_hf_to_gguf import main

if __name__ == "__main__":
    sys.exit(main())
""")
            logger.info(f"Created wrapper script at {converter_script}")
            return converter_script
    except Exception as e:
        logger.error(f"Failed to set up llama-cpp-python converter: {e}")
    
    raise ValueError("Could not find or set up conversion script. Please check llama.cpp repository or install llama-cpp-python with conversion support.")

def create_lm_studio_settings(args):
    """Create settings file for LM Studio with enhanced stop sequences"""
    logger.info("Creating LM Studio settings...")
    
    settings = {
        "modelName": args.model_name,
        "parameters": {
            "temperature": 0.4,
            "top_p": 0.95,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "frequency_penalty": 0.1,  # Added to discourage repetition
            "presence_penalty": 0.1,   # Added to discourage topic repetition
            "max_tokens": 100,         # Keep responses concise
            "context_length": args.ctx_size,
            "stop_on_eos": True        # Force stopping on EOS token - will be converted to true in JSON
        },
        "systemMessage": f"You are {args.model_name}, a digital persona representing Lakshman Turlapati. Answer questions concisely and naturally as Lakshman would. Keep responses brief and to the point.",
        "template": "<|user|>\n{{prompt}}\n<|assistant|>\n{{response}}",
        "stopSequences": [
            "<|user|>",
            "<|endoftext|>",
            "<|pad|>",
            "\n\n",
            "Human:",
            "User:",
            "Question:",
            "##",
            "###",
            "Q:",
            "--",
            "Want to",
            "Would you",
            "Do you",
            "Have you",
            "Can you",
            "What's",
            "What is",
            "How do",
            "How does",
            "How would",
            "Tell me",
            "?"
        ]
    }
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Write settings to file
    settings_path = os.path.join(args.output_dir, "lm_studio_settings.json")
    with open(settings_path, "w") as f:
        # Using json.dump will convert Python's True to JSON's true
        json.dump(settings, f, indent=2)
    
    logger.info(f"LM Studio settings saved to {settings_path}")
    return settings_path

def convert_model(args):
    """Convert the model to GGUF format for use with llama.cpp with enhanced EOS handling"""
    # Resolve the path to the model, handling both absolute and relative paths
    model_path = os.path.abspath(args.model_path)
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist")
    
    # Create output dir if it doesn't exist
    output_dir = os.path.join(model_path, "gguf")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif args.force:
        logger.warning(f"Output directory {output_dir} already exists, content may be overwritten")
    else:
        logger.warning(f"Output directory {output_dir} already exists, use --force to overwrite")
        return

    # Ensure llama.cpp is available
    llama_cpp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama.cpp")
    convert_script = ensure_llama_cpp(llama_cpp_dir)
    
    # Make sure required packages are installed
    packages = ["torch", "sentencepiece", "transformers", "protobuf"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            logger.info(f"Installing required package: {package}")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
    
    # Create a special tokens file for conversion - this will be used separately
    # as the direct command line arguments are not compatible
    tokens_file = os.path.join(output_dir, "special_tokens.json")
    with open(tokens_file, 'w') as f:
        json.dump({
            "eos_token": "<|endoftext|>",
            "pad_token": "<|pad|>",
            "additional_special_tokens": ["<|user|>", "<|assistant|>"]
        }, f, indent=2)
    logger.info(f"Created special tokens file at {tokens_file}")
    
    # Basic command setup - only include compatible arguments
    output_path = os.path.join(output_dir, "model.gguf")
    base_cmd = [sys.executable, convert_script, "--outfile", output_path]
    
    # Add model name if provided
    if args.model_name:
        base_cmd.extend(["--model-name", args.model_name])
    
    if args.outtype:
        # Check if the convert script supports --outtype argument by first running help
        try:
            help_cmd = [sys.executable, convert_script, "--help"]
            help_output = subprocess.run(help_cmd, capture_output=True, text=True).stdout
            
            if "--outtype" in help_output:
                base_cmd.extend(["--outtype", args.outtype])
            elif "--type" in help_output:
                base_cmd.extend(["--type", args.outtype])
            else:
                logger.warning("Conversion script does not seem to support outtype argument, trying with --outtype anyway")
                base_cmd.extend(["--outtype", args.outtype])
        except Exception as e:
            logger.warning(f"Error checking conversion script help: {e}")
            # Default to using --outtype
            base_cmd.extend(["--outtype", args.outtype])
    
    # Add model path - always last argument
    base_cmd.extend([model_path])
    
    # First create an enhanced generation settings file for LM Studio
    settings_path = os.path.join(output_dir, "lm_studio_settings.json")
    create_lm_studio_settings(args)
    
    # Also create a separate generation settings file
    # This is specifically for handling endless generation issues
    generation_path = os.path.join(output_dir, "generation_settings.json")
    with open(generation_path, 'w') as f:
        json.dump({
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "repetition_penalty": 1.2,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "max_tokens": 100,
            "stop": [
                "<|user|>", "<|endoftext|>", "<|pad|>",
                "\n\n", "Human:", "User:", "Question:", 
                "What's", "?", "##", "###"
            ]
        }, f, indent=2)
    logger.info(f"Created generation settings at {generation_path}")
    
    # Run conversion
    logger.info(f"Running conversion command: {' '.join(base_cmd)}")
    success = False
    
    try:
        # Use real-time output
        with subprocess.Popen(
            base_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1, 
            universal_newlines=True
        ) as process:
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            if process.returncode != 0:
                logger.error(f"Conversion failed with return code {process.returncode}")
                # Try alternate conversion methods
                success = try_alternate_conversion(model_path, output_path, args)
            else:
                logger.info(f"Conversion completed successfully")
                success = True
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        success = try_alternate_conversion(model_path, output_path, args)
    
    # If all conversion methods failed and the output doesn't exist,
    # create a placeholder GGUF file to at least have the settings files available
    if not success and not os.path.exists(output_path):
        # Create a settings-only package for LM Studio
        warn_message = """
# ⚠️ CONVERSION FAILED - SETTINGS ONLY ⚠️

The GGUF conversion process failed to create a model file.
However, the LM Studio settings have been created successfully.

To fix the endless generation issue:
1. Manually convert your model to GGUF format using one of these methods:
   - Use llama.cpp's convert_hf_to_gguf.py script
   - Use the llama-cpp-python package's converter
   - Use a different conversion tool like HuggingFace's or ctransformers

2. Place your converted model in this directory and apply the 
   settings from lm_studio_settings.json in LM Studio.

See ENDLESS_GENERATION_FIX.md for detailed instructions.
"""
        # Create a small readme file explaining the issue
        with open(os.path.join(output_dir, "CONVERSION_FAILED.md"), "w") as f:
            f.write(warn_message)
        
        logger.warning("Conversion failed, but created settings files for manual use")
    
    # Print completion message with links to output files
    if os.path.exists(output_path) or os.path.exists(os.path.join(output_dir, "CONVERSION_FAILED.md")):
        logger.info(f"LM Studio settings saved to: {settings_path}")
        logger.info(f"Generation settings saved to: {generation_path}")
        logger.info("Use these files with LM Studio for controlled local inference")
        
        # Create a companion settings README specifically for fixing endless generation
        endless_fix_path = os.path.join(output_dir, "ENDLESS_GENERATION_FIX.md")
        with open(endless_fix_path, 'w') as f:
            f.write("""# Fixing Endless Generation in LM Studio

If your model generates text endlessly without stopping, follow these steps:

## Settings to Apply

1. **Basic Parameters:**
   - Set max_tokens: 100 (or lower if needed)
   - Increase repetition_penalty: 1.2-1.3
   - Add frequency_penalty: 0.1
   - Add presence_penalty: 0.1

2. **Critical Stop Sequences:**
   Add ALL of these stop sequences:
   ```
   <|user|>
   <|endoftext|>
   <|pad|>
   ?
   \n\n
   Human:
   User:
   Question:
   ##
   ###
   ```

3. **System Message:**
   Update system message to include: "Keep responses brief, concise and direct. Do not elaborate unnecessarily."

4. **Template Format:**
   Ensure template is set to:
   ```
   <|user|>
   {{prompt}}
   <|assistant|>
   {{response}}
   ```

## Advanced Settings

If problems persist, try setting "stop_on_eos": true in the Parameters section and reduce Temperature to 0.5.

Remember, shorter max_tokens values and proper stop sequences are the most effective ways to prevent endless generation.
""")
        logger.info(f"Created endless generation fix guide at {endless_fix_path}")
    else:
        logger.error("Conversion failed, output file not found")
        
    return success

def try_alternate_conversion(model_path, output_path, args):
    """Try alternate conversion methods using various approaches"""
    logger.info("Attempting alternate conversion methods...")
    
    # Method 1: Try using llama-cpp-python's built-in converter
    try:
        logger.info("Method 1: Using llama_cpp.convert_hf_to_gguf module")
        import importlib.util
        
        # Check if the module is available
        if importlib.util.find_spec("llama_cpp.convert_hf_to_gguf"):
            from llama_cpp.convert_hf_to_gguf import convert_hf_to_gguf
            
            # Parameters for conversion
            params = {
                "model_path": model_path,
                "outfile": output_path,
            }
            
            if args.outtype:
                params["outtype"] = args.outtype
            
            logger.info(f"Converting with llama_cpp.convert_hf_to_gguf: {params}")
            convert_hf_to_gguf(**params)
            logger.info("Conversion completed with llama_cpp module")
            return True
        else:
            logger.warning("llama_cpp.convert_hf_to_gguf module not available")
    except Exception as e:
        logger.error(f"Method 1 failed: {e}")
    
    # Method 2: Try direct conversion with simplified command
    try:
        logger.info("Method 2: Using simplified command")
        # Find the most recent script with 'convert' in the name
        llama_cpp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama.cpp")
        result = subprocess.run(["find", llama_cpp_dir, "-name", "*convert*.py"], 
                             capture_output=True, text=True)
        scripts = result.stdout.strip().split('\n')
        
        if scripts:
            script = scripts[0]  # Use the first script found
            logger.info(f"Using conversion script: {script}")
            
            simple_cmd = [
                sys.executable,
                script,
                "--outfile", output_path,
                model_path
            ]
            
            logger.info(f"Running simplified command: {' '.join(simple_cmd)}")
            subprocess.run(simple_cmd, check=True)
            logger.info("Conversion succeeded with simplified command")
            return True
    except Exception as e:
        logger.error(f"Method 2 failed: {e}")
    
    # Method 3: Try using Python module command
    try:
        logger.info("Method 3: Using -m python module approach")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "llama-cpp-python"
        ], check=True)
        
        module_cmd = [
            sys.executable, "-m", "llama_cpp.convert_hf_to_gguf",
            "--outfile", output_path,
            model_path
        ]
        
        if args.outtype:
            module_cmd.extend(["--outtype", args.outtype])
            
        logger.info(f"Running module command: {' '.join(module_cmd)}")
        subprocess.run(module_cmd, check=True)
        logger.info("Conversion succeeded with module command")
        return True
    except Exception as e:
        logger.error(f"Method 3 failed: {e}")
    
    # If we get here, all methods failed
    logger.error("""
Conversion failed with all methods. Manual steps required:
1. Install llama-cpp-python: pip install llama-cpp-python
2. Install required packages: pip install torch sentencepiece transformers protobuf
3. Convert manually: python -m llama_cpp.convert_hf_to_gguf --outfile output.gguf your_model_path
4. Use our LM Studio settings with the manually converted model
""")
    
    return False

def quantize_model(args, base_model_path, llama_cpp_dir):
    """Quantize the model to various formats"""
    if not args.quantize:
        logger.info("Skipping quantization (use --quantize to create quantized versions)")
        return
    
    logger.info("Creating quantized versions of the model...")
    
    # Define quantization formats
    formats = ["q4_k_m", "q5_k_m", "q8_0"]
    
    # Get path to quantize binary
    quantize_bin = os.path.join(llama_cpp_dir, "quantize")
    if not os.path.exists(quantize_bin):
        logger.warning(f"Quantize binary not found at {quantize_bin}, trying build directory")
        quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "quantize")
        if not os.path.exists(quantize_bin):
            logger.error("Quantize binary not found")
            return
    
    # Create quantized versions
    for quant_format in formats:
        output_path = os.path.join(args.output_dir, f"{args.model_name.lower()}-{quant_format}.gguf")
        
        # Skip if file exists and not forced
        if os.path.exists(output_path) and not args.force:
            logger.warning(f"Skipping {quant_format} quantization, file already exists: {output_path}")
            continue
        
        # Run quantization
        logger.info(f"Creating {quant_format} quantized version...")
        cmd = [
            quantize_bin,
            base_model_path,
            output_path,
            quant_format
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully created {quant_format} quantized model at {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during {quant_format} quantization: {e}")

def create_readme(args):
    """Create a README file with instructions"""
    readme_path = os.path.join(args.output_dir, "README.md")
    
    content = f"""# {args.model_name} GGUF Models

## Model Information
- Base Model: SmolLM-360M
- Fine-tuned as: {args.model_name}
- Context Length: {args.ctx_size}
- Format: GGUF

## Available Versions
- Base (f16): {args.model_name.lower()}-base.gguf
"""
    
    if args.quantize:
        content += """- Q4_K_M: [model]-q4_k_m.gguf (Recommended for inference)
- Q5_K_M: [model]-q5_k_m.gguf (Better quality, larger size)
- Q8_0: [model]-q8_0.gguf (Best quality, largest size)
"""
    
    content += """
## LM Studio Setup
1. Copy the lm_studio_settings.json file to your LM Studio settings
2. Load the model file (q4_k_m version recommended for inference)
3. Apply the settings
4. Start chatting!

## Prompt Format
The model expects input in the following format:
```
<|user|>
[your question here]
<|assistant|>
```

## Special Tokens
- User token: <|user|>
- Assistant token: <|assistant|>
- End of text token: <|endoftext|>
- Pad token: <|pad|>
"""
    
    with open(readme_path, "w") as f:
        f.write(content)
    
    logger.info(f"Created README at {readme_path}")

def convert_to_gguf(args):
    """Convert the model to GGUF format"""
    try:
        # Set up llama.cpp
        convert_script = ensure_llama_cpp(args.llama_cpp_dir)
        
        # Create LM Studio settings
        create_lm_studio_settings(args)
        
        # Convert base model
        convert_model(args)
        
        # Create quantized versions
        quantize_model(args, args.model_path, args.llama_cpp_dir)
        
        # Create README
        create_readme(args)
        
        logger.info("Conversion complete!")
        logger.info(f"Models saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#######################
# API SERVER FUNCTIONS
#######################

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                logger.info(f"Port {port} is already in use, trying next port...")
                continue
    
    # If we get here, we couldn't find an available port in the range
    # Just return a random port and let Flask handle the error if it's also taken
    return start_port + max_attempts + 1

def create_app(model, tokenizer, device, max_tokens=100):
    """Create a Flask app for hosting the model"""
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    @app.route("/api/generate", methods=["POST"])
    def generate():
        try:
            # Get data from request
            data = request.json
            if not data or "prompt" not in data:
                return jsonify({"error": "Missing 'prompt' in request"}), 400
            
            # Get prompt from request
            prompt = data["prompt"]
            
            # Set max tokens (use request value or default)
            max_tokens_value = data.get("max_tokens", max_tokens)
            
            # Generate response
            response = generate_response(model, tokenizer, prompt, device, max_tokens_value)
            
            # Return response
            return jsonify({
                "response": response,
                "model": CONFIG["persona_name"]
            })
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/health", methods=["GET"])
    def health():
        """Health check endpoint"""
        return jsonify({
            "status": "ok",
            "model": CONFIG["persona_name"],
            "model_loaded": model is not None and tokenizer is not None
        })
    
    @app.route("/", methods=["GET"])
    def index():
        """Simple HTML interface for testing"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Parz API Server</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                textarea { width: 100%; height: 100px; margin: 10px 0; }
                button { padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; }
                #response { margin-top: 20px; white-space: pre-wrap; background: #f5f5f5; padding: 10px; }
            </style>
        </head>
        <body>
            <h1>Parz API Server</h1>
            <p>Enter your prompt below to test the API:</p>
            <textarea id="prompt" placeholder="Enter your prompt here..."></textarea>
            <button onclick="generateResponse()">Generate Response</button>
            <div id="response"></div>
            
            <script>
                async function generateResponse() {
                    const prompt = document.getElementById('prompt').value;
                    const responseDiv = document.getElementById('response');
                    
                    responseDiv.textContent = 'Generating...';
                    
                    try {
                        const response = await fetch('/api/generate', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ prompt })
                        });
                        
                        const data = await response.json();
                        
                        if (data.error) {
                            responseDiv.textContent = `Error: ${data.error}`;
                        } else {
                            responseDiv.textContent = data.response;
                        }
                    } catch (error) {
                        responseDiv.textContent = `Error: ${error.message}`;
                    }
                }
            </script>
        </body>
        </html>
        """
    
    return app

def run_api_server(args):
    """Run the API server with the model"""
    # Get device
    device = get_device(args.device)
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path, device)
    
    # Create Flask app
    app = create_app(model, tokenizer, device, args.max_tokens)
    
    # Check if the port is available, find an alternative if not
    port = args.port
    try:
        # Try to check if port is available
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((args.host, port))
    except OSError:
        # Port is not available, find an alternative
        logger.warning(f"Port {port} is already in use")
        port = find_available_port(port + 1)
        logger.info(f"Using alternative port: {port}")
    
    # Run server
    logger.info(f"Starting API server on {args.host}:{port}...")
    app.run(host=args.host, port=port, debug=False)

#######################
# COMMAND LINE INTERFACE
#######################

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="All-in-one script for Parz model: finetune, inference, and GGUF conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Finetune parser
    finetune_parser = subparsers.add_parser("finetune", help="Finetune the model")
    finetune_parser.add_argument("--dataset", type=str, default=CONFIG["dataset_path"],
                               help="Path to dataset JSON file")
    finetune_parser.add_argument("--output_dir", type=str, default=CONFIG["output_dir"],
                               help="Directory to save the finetuned model")
    finetune_parser.add_argument("--base_model", type=str, default=CONFIG["model_name"],
                               help="Base model to finetune")
    finetune_parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"],
                               help="Device to use (defaults to best available)")
    
    # Inference parser
    inference_parser = subparsers.add_parser("inference", help="Run inference with the model")
    inference_parser.add_argument("--model_path", type=str, default=CONFIG["output_dir"],
                                help="Path to the finetuned model")
    inference_parser.add_argument("--prompt", type=str,
                                help="Prompt for one-shot inference")
    inference_parser.add_argument("--interactive", action="store_true",
                                help="Run in interactive mode")
    inference_parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"],
                                help="Device to use (defaults to best available)")
    inference_parser.add_argument("--max_tokens", type=int, default=100,
                                help="Maximum tokens to generate")
    
    # Convert parser
    convert_parser = subparsers.add_parser("convert", help="Convert model to GGUF format")
    convert_parser.add_argument("--model_path", type=str, default=CONFIG["output_dir"],
                              help="Path to the finetuned model")
    convert_parser.add_argument("--output_dir", type=str, default=CONFIG["gguf_dir"],
                              help="Directory to save GGUF models")
    convert_parser.add_argument("--quantize", action="store_true",
                              help="Create quantized versions")
    convert_parser.add_argument("--ctx_size", type=int, default=CONFIG["ctx_size"],
                              help="Context size for the model")
    convert_parser.add_argument("--model_name", type=str, default=CONFIG["persona_name"],
                              help="Name for the model")
    convert_parser.add_argument("--force", action="store_true",
                              help="Force overwrite existing files")
    convert_parser.add_argument("--llama_cpp_dir", type=str, default="./llama.cpp",
                              help="Path to llama.cpp directory")
    convert_parser.add_argument("--outtype", type=str,
                              help="Output type for the model")
    
    # Host parser
    host_parser = subparsers.add_parser("host", help="Host model as a local API server")
    host_parser.add_argument("--model_path", type=str, default=CONFIG["output_dir"],
                           help="Path to the finetuned model")
    host_parser.add_argument("--port", type=int, default=8000,
                           help="Port to run the server on")
    host_parser.add_argument("--host", type=str, default="127.0.0.1",
                           help="Host to run the server on (use 0.0.0.0 to allow external connections)")
    host_parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"],
                           help="Device to use (defaults to best available)")
    host_parser.add_argument("--max_tokens", type=int, default=100,
                           help="Default maximum tokens to generate")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    if args.command == "finetune":
        # Update CONFIG with args
        if args.dataset:
            CONFIG["dataset_path"] = args.dataset
        if args.output_dir:
            CONFIG["output_dir"] = args.output_dir
        if args.base_model:
            CONFIG["model_name"] = args.base_model
        
        # Run finetune
        finetune_model()
    
    elif args.command == "inference":
        # Run inference
        run_inference(args)
    
    elif args.command == "convert":
        # Run GGUF conversion
        convert_to_gguf(args)
    
    elif args.command == "host":
        # Run API server
        run_api_server(args)
    
    else:
        # No command specified, print help
        print("Please specify a command: finetune, inference, convert, or host")
        print("Run 'python parz.py --help' for more information")
        sys.exit(1)

if __name__ == "__main__":
    main() 