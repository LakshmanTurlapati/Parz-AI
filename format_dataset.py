#!/usr/bin/env python3
"""
Format the Parz dataset to various formats including User/Assistant format

This script converts the dataset with "question" and "answer" fields to:
1. Industry-standard "user" and "assistant" format (used by OpenAI, Claude, etc.)
2. Q&A format with [Q]/[A] markers (used by the base model training)
"""

import json
import argparse
import os
import sys

def load_dataset(path):
    """Load the dataset from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)

def convert_to_user_assistant(dataset):
    """Convert dataset from Q&A format to User/Assistant format"""
    converted_data = []
    
    for item in dataset:
        # Skip empty items
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            continue
            
        question = item["question"].strip()
        answer = item["answer"].strip()
        
        # Create a converted item
        converted_item = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        
        converted_data.append(converted_item)
    
    return converted_data

def convert_to_qa_format(dataset, system_prompt="You are Parz, Lakshman Turlapati's digital persona."):
    """Convert dataset to Q&A format with system prompt"""
    formatted_data = []
    
    for item in dataset:
        # Skip empty items
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            continue
            
        question = item["question"].strip()
        answer = item["answer"].strip()
        
        # Format with system prompt, Q and A markers
        formatted_text = f"{system_prompt} [Q] {question} [A] {answer}"
        formatted_data.append(formatted_text)
    
    return formatted_data

def save_user_assistant_dataset(data, output_path):
    """Save the User/Assistant formatted dataset"""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved User/Assistant dataset to {output_path}")
    print(f"Total conversations: {len(data)}")

def save_qa_dataset(data, output_path):
    """Save the Q&A formatted dataset"""
    with open(output_path, 'w') as f:
        json.dump({"text": data}, f, indent=2)
    print(f"Saved Q&A dataset to {output_path}")
    print(f"Total samples: {len(data)}")

def main():
    parser = argparse.ArgumentParser(description="Format dataset to various formats")
    parser.add_argument("--input", type=str, default="Dataset/lakshman.json", 
                        help="Path to input dataset")
    parser.add_argument("--output-user-assistant", type=str, default="Dataset/lakshman_user_assistant.json",
                        help="Path to save User/Assistant formatted dataset")
    parser.add_argument("--output-qa", type=str, default="Dataset/lakshman_qa.json",
                        help="Path to save Q&A formatted dataset")
    parser.add_argument("--format", type=str, choices=["user-assistant", "qa", "both"], default="both",
                        help="Format(s) to generate")
    parser.add_argument("--system-prompt", type=str, 
                        default="You are Parz, Lakshman Turlapati's digital persona.",
                        help="System prompt for Q&A format")
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return 1
    
    # Create output directories if they don't exist
    for output_path in [args.output_user_assistant, args.output_qa]:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Load dataset
    print(f"Loading dataset from {args.input}...")
    dataset = load_dataset(args.input)
    print(f"Loaded {len(dataset)} question/answer pairs")
    
    # Process and save in requested formats
    if args.format in ["user-assistant", "both"]:
        print("Converting to User/Assistant format...")
        user_assistant_data = convert_to_user_assistant(dataset)
        save_user_assistant_dataset(user_assistant_data, args.output_user_assistant)
        print("\nUser/Assistant format example:")
        print(json.dumps(user_assistant_data[0], indent=2))
        
    if args.format in ["qa", "both"]:
        print("\nConverting to Q&A format...")
        qa_data = convert_to_qa_format(dataset, args.system_prompt)
        save_qa_dataset(qa_data, args.output_qa)
        print("\nQ&A format example:")
        print(qa_data[0])
    
    print("\nDataset formatting complete!")
    
    # Print usage recommendations
    print("\nUsage recommendations:")
    if args.format in ["user-assistant", "both"]:
        print("\n1. For User/Assistant format (OpenAI, Claude APIs):")
        print("   Use with train_with_user_assistant.py for modern industry-standard training")
    if args.format in ["qa", "both"]:
        print("\n2. For Q&A format (Base model training):")
        print("   Use with finetune.py for original training flow")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 