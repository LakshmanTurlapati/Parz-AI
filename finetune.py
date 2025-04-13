import json
import os
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
import logging

# Set environment variables for MPS fallback
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if CUDA is available, otherwise use MPS (Mac M1/M2) or CPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {device}")

def load_dataset(dataset_path):
    """Load dataset from JSON file"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples from {dataset_path}")
    return data

def format_dataset(data):
    """Format the dataset for training"""
    formatted_data = []
    
    for item in data:
        # Extract user and assistant messages
        user_message = item.get("user", "")
        assistant_message = item.get("assistant", "")
        
        if not user_message or not assistant_message:
            logger.warning(f"Skipping example with missing user or assistant message: {item}")
            continue
        
        # Format the example with special tokens
        formatted_example = f"<|user|>\n{user_message}\n<|assistant|>\n{assistant_message}<|endoftext|>"
        formatted_data.append({"text": formatted_example})
    
    logger.info(f"Formatted {len(formatted_data)} examples for training")
    
    # Check sequence lengths before tokenization
    lengths = [len(example["text"]) for example in formatted_data]
    logger.info(f"Sequence length stats (chars): min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.2f}")
    
    return formatted_data

def create_datasets(formatted_data):
    """Create a HuggingFace Dataset from the formatted data"""
    dataset = Dataset.from_list(formatted_data)
    
    # Split dataset into train/eval (90/10 split)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    logger.info(f"Created dataset with {len(dataset['train'])} training examples and {len(dataset['test'])} evaluation examples")
    return dataset

def load_model_and_tokenizer(model_name):
    """Load base model and tokenizer"""
    logger.info(f"Loading model and tokenizer from {model_name}")
    
    # Load the tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens if they don't exist
    special_tokens = {
        "additional_special_tokens": []
    }
    
    # Check for and add user and assistant tokens
    if "<|user|>" not in tokenizer.get_vocab():
        special_tokens["additional_special_tokens"].append("<|user|>")
    
    if "<|assistant|>" not in tokenizer.get_vocab():
        special_tokens["additional_special_tokens"].append("<|assistant|>")
    
    # Set up pad token (distinct from eos token to avoid attention mask warning)
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        if "<|pad|>" not in tokenizer.get_vocab():
            special_tokens["pad_token"] = "<|pad|>"
        else:
            tokenizer.pad_token = "<|pad|>"
    
    # Set eos token if not already set
    if tokenizer.eos_token is None:
        if "<|endoftext|>" in tokenizer.get_vocab():
            tokenizer.eos_token = "<|endoftext|>"
        else:
            special_tokens["eos_token"] = "<|endoftext|>"
    
    # Apply special tokens if we have any to add
    if special_tokens["additional_special_tokens"] or "pad_token" in special_tokens or "eos_token" in special_tokens:
        num_added = tokenizer.add_special_tokens(special_tokens)
        logger.info(f"Added {num_added} special tokens to the tokenizer")
    
    # Log the special tokens
    logger.info(f"Tokenizer: pad_token={tokenizer.pad_token}, eos_token={tokenizer.eos_token}")
    
    # Load model on CPU first regardless of target device
    logger.info("Loading model on CPU first to handle special token embedding resizing safely")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="cpu"  # Force CPU for initialization
    )
    
    # Resize token embeddings on CPU to avoid MPS issues with Cholesky decomposition
    if model.get_input_embeddings().weight.shape[0] < len(tokenizer):
        logger.info("Resizing model embeddings to match tokenizer (on CPU)")
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    
    # Move model to target device after embedding resize
    logger.info(f"Moving model to {device} device")
    model = model.to(device)
    
    return model, tokenizer

def tokenize_dataset(dataset, tokenizer, max_length):
    """Tokenize the dataset"""
    def tokenize_function(examples):
        # Tokenize with attention mask and proper padding
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True  # Explicitly request attention mask
        )
        
        # Set labels equal to input ids for causal language modeling
        result["labels"] = result["input_ids"].copy()
        
        return result
    
    # Apply tokenization to the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    logger.info(f"Tokenized dataset with max_length={max_length}")
    
    return tokenized_dataset

class TrailRunCallback(TrainerCallback):
    """Custom callback to generate text during training"""
    def __init__(self, tokenizer, device, every_n_steps=100):
        self.tokenizer = tokenizer
        self.device = device
        self.every_n_steps = every_n_steps
        self.example_prompt = "What's your favorite game?"
    
    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step % self.every_n_steps == 0:
            # Format the prompt with special tokens
            prompt = f"<|user|>\n{self.example_prompt}\n<|assistant|>\n"
            
            # Tokenize with attention mask
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                return_attention_mask=True
            ).to(self.device)
            
            # Generate text with attention mask
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode the generated text
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Print the generated text
            try:
                # Extract just the assistant's response
                response = generated.split("<|assistant|>")[1].strip()
                logger.info(f"\nTrail Run (Step {state.global_step}):")
                logger.info(f"Q: {self.example_prompt}")
                logger.info(f"A: {response}\n")
            except IndexError:
                logger.info(f"\nTrail Run (Step {state.global_step}):")
                logger.info(f"Raw generated: {generated}\n")

def main():
    # Load dataset
    data = load_dataset("Dataset/lakshman_user_assistant.json")
    
    # Format dataset
    formatted_data = format_dataset(data)
    
    # Create HuggingFace datasets
    dataset = create_datasets(formatted_data)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer("HuggingFaceTB/SmolLM-360M")
    
    # Analyze sequence lengths to determine max_length
    sample_texts = [item["text"] for item in formatted_data[:100]]
    encoded_lengths = [len(tokenizer.encode(text)) for text in sample_texts]
    max_length = min(2048, max(encoded_lengths) + 100)  # Add some padding
    
    logger.info(f"Setting max sequence length to {max_length}")
    
    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./parz_final",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_dir="./logs",
        logging_steps=10,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        warmup_steps=100,
        weight_decay=0.01,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=100,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),  # Use fp16 if CUDA is available
        report_to="none",  # Disable reporting to prevent wandb, etc.
        push_to_hub=False,  # Don't push to HuggingFace Hub
    )
    
    # Set up callback for trail runs
    trail_run_callback = TrailRunCallback(tokenizer, device)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        callbacks=[trail_run_callback]
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model and tokenizer
    logger.info("Saving model and tokenizer...")
    trainer.save_model("./parz_final")
    tokenizer.save_pretrained("./parz_final")
    
    # Test the model
    logger.info("Testing model...")
    test_prompt = "What's your favorite game?"
    formatted_prompt = f"<|user|>\n{test_prompt}\n<|assistant|>\n"
    
    # Tokenize with attention mask
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt",
        return_attention_mask=True
    ).to(device)
    
    # Generate text with attention mask
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode the generated text
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    try:
        response = generated.split("<|assistant|>")[1].strip()
        logger.info(f"Final test:")
        logger.info(f"Q: {test_prompt}")
        logger.info(f"A: {response}")
    except IndexError:
        logger.info(f"Final test (raw output): {generated}")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main() 