# Parz Digital Persona

[![GitHub issues](https://img.shields.io/github/issues/LakshmanTurlapati/Parz-AI)](https://github.com/LakshmanTurlapati/Parz-AI/issues)
[![GitHub license](https://img.shields.io/github/license/LakshmanTurlapati/Parz-AI)](https://github.com/LakshmanTurlapati/Parz-AI/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python&logoColor=white)](https://python.org)
[![AI](https://img.shields.io/badge/AI-SmolLM-green?style=flat&logo=openai&logoColor=white)](https://huggingface.co/HuggingFaceTB/SmolLM-360M)


A powerful, lightweight, and customizable AI persona creator built on SmolLM-360M. Finetune, deploy, and host your own digital twin with just a few commands!

*Created by [Lakshman Turlapati](https://www.audienclature.com)*


## Overview

Parz is an all-in-one solution for creating, training, and deploying your own AI persona. It leverages the revolutionary SmolLM models that are small enough to run on consumer hardware but powerful enough to create a compelling AI personality.

With Parz, you can:
- **Finetune** a base model on your personal data
- Run **interactive inference** locally
- **Convert** your model to optimized GGUF format for deployment
- **Host** your model as a local API server

## What Makes This Special?

Most AI persona projects require expensive cloud GPU rentals or commercial API subscriptions. Parz breaks that barrier by using the SmolLM architecture, enabling anyone with decent consumer hardware to create, train, and host their own AI twin.

## Setup

### Requirements
- Python 3.8+
- 8GB+ RAM
- GPU recommended but not required

### Installation

```bash
# Clone the repository
git clone https://github.com/LakshmanTurlapati/Parz-AI.git
cd Parz-AI

# Create and activate a virtual environment
python -m venv parz_env
source parz_env/bin/activate  # On Windows: parz_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Creating Your Dataset

The model is trained on conversation data in a simple JSON format. Create a file `Dataset/your_data.json` with this structure:

```json
[
  {
    "messages": [
      {"role": "user", "content": "What's your name?"},
      {"role": "assistant", "content": "I'm [Your Name], a digital persona..."}
    ]
  },
  {
    "messages": [
      {"role": "user", "content": "Tell me about your background."},
      {"role": "assistant", "content": "I studied at [University]..."}
    ]
  }
]
```

Include 20-100 representative QA pairs covering your personality, background, interests, and communication style. More diverse examples will create a more authentic persona.

## Finetuning Your Model

```bash
# Basic finetuning with defaults
python parz.py finetune

# Custom finetuning
python parz.py finetune --dataset Dataset/your_data.json --output_dir ./your_model
```

Finetuning takes 1-4 hours depending on your hardware. The model will periodically test itself with sample questions.

## Running Inference

Test your model with inference mode:

```bash
# Interactive chat mode
python parz.py inference --interactive

# Single prompt inference
python parz.py inference --prompt "What's your favorite hobby?"
```

## Converting to GGUF

For deployment and optimized inference, convert your model to GGUF format:

```bash
# Basic conversion
python parz.py convert

# With quantization for smaller file size
python parz.py convert --quantize
```

## Hosting Your AI

Host your model as a local API server:

```bash
# Basic hosting on localhost:8000
python parz.py host

# Custom host configuration
python parz.py host --host 0.0.0.0 --port 8080
```

The server automatically finds an available port if the default is in use.

### API Endpoints

- **GET /** - Web interface for testing
- **GET /api/health** - Health check
- **POST /api/generate** - Generate responses
  ```json
  {"prompt": "Who are you?", "max_tokens": 100}
  ```

## Hardware Requirements

- **Finetuning**: 8GB+ RAM, GPU recommended (but works on CPU)
- **Inference**: 4GB+ RAM, any modern CPU
- **Hosting**: Same as inference requirements

One of the revolutionary aspects of SmolLM is that it works on virtually any modern computer - no expensive GPU required!

## Example Use Cases

1. **Personal Digital Twin** - Create an AI version of yourself that can represent you online
2. **Customer Service Assistant** - Train on your product information for customer support
3. **Educational Companion** - Create a persona specialized in teaching specific subjects
4. **Creative Writing Partner** - Train on writing samples to help with brainstorming

## Links & Resources

- **Repository**: [github.com/LakshmanTurlapati/Parz-AI](https://github.com/LakshmanTurlapati/Parz-AI.git)
- **Author's Website**: [audienclature.com](https://www.audienclature.com)
- **LinkedIn**: [Lakshman Turlapati](https://www.linkedin.com/in/lakshman-turlapati-3091aa191/)
- **Twitter**: [@parzival1213](https://x.com/parzival1213)

## How I Built This

I created Parz to make personal AI twins accessible to everyone. After experimenting with larger models that required expensive cloud compute, I discovered SmolLM and was amazed by how it could run on consumer hardware while maintaining impressive capabilities.

By fine-tuning SmolLM-360M on a dataset of my own conversations, writings, and curated responses, I created a digital version of myself that captures my personality, knowledge areas, and communication style.

The all-in-one script approach simplifies what would otherwise be a complex process involving multiple tools and steps, making AI persona creation accessible to everyone - not just ML experts.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

---

*Note: This project was created by Lakshman Turlapati as a demonstration of how small language models can be leveraged for personalized AI experiences. Happy coding!*
