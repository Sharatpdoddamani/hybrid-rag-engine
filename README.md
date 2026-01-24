# 🤖 Persona-Based Chat Assistant

A lightweight, customizable conversational AI assistant built with Qwen2.5-1.5B. Create custom character personas and engage in contextual, multi-turn conversations through a modern web interface, REST API, or Discord bot integration.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Custom Personas**: Define unique character personalities, backgrounds, and speaking styles
- **Multi-turn Conversations**: Maintains context across conversation history with intelligent context compression
- **Multiple Interfaces**:
  - **Web UI**: Modern Gradio-based interface with dark mode, streaming responses, and conversation management
  - **REST API**: FastAPI server with rate limiting, caching, and multi-user support
  - **Discord Bot**: Channel-based persona management and conversation tracking
- **Advanced Features**:
  - Streaming responses with typewriter effect
  - Regenerate and edit past messages
  - Conversation search and statistics
  - Save/load conversations (JSON/TXT)
  - Character presets and templates
  - Adjustable creativity and response length
  - Context window management

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 4GB+ VRAM for GPU mode, 8GB+ RAM for CPU mode
- PyTorch with CUDA support (for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sharatpdoddamani/GPU-accelerated-Locally-Runnable-AI-personal-Assistant.git
cd GPU-accelerated-Locally-Runnable-AI-personal-Assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Web Interface

Launch the Gradio web interface:
```bash
python app.py
```

Open your browser and navigate to `http://localhost:7860`

### API Server

Start the FastAPI server:
```bash
python api_server.py
```

The API will be available at `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Discord Bot

1. Create a Discord bot at [Discord Developer Portal](https://discord.com/developers/applications)
2. Set the bot token as an environment variable:
```bash
# Windows
set DISCORD_BOT_TOKEN=your_token_here

# Linux/Mac
export DISCORD_BOT_TOKEN=your_token_here
```

3. Run the bot:
```bash
python discord_bot.py
```

### Test Script

Verify your setup:
```bash
python test_bot.py
```

## Usage

### Web Interface

1. **Set Persona**: Enter a character description in the "Character Persona" box
2. **Start Chatting**: Type your message and press Enter or click Send
3. **Customize**: Adjust temperature (creativity), max tokens (response length), and context window
4. **Manage Conversations**: Save, load, search, and regenerate responses

### API Usage

Example API request:
```python
import requests

response = requests.post("http://localhost:8000/chat", json={
    "message": "Hello! Tell me about yourself.",
    "persona": "You are a helpful AI assistant.",
    "max_tokens": 150
})

print(response.json()["response"])
```

### Discord Commands

- `!chat <message>` - Chat with the assistant
- `!setpersona <persona>` - Set character persona for the channel
- `!reset` - Reset conversation history
- `!help` - Show help message

## Model Information

- **Base Model**: Qwen2.5-1.5B-Instruct
- **Quantization**: 8-bit (balanced mode) or FP16 (high performance mode)
- **Context Window**: Configurable (default: 10 messages)
- **Performance**: 
  - GPU: ~75%+ utilization in high mode
  - Response time: 1-3 seconds on GPU, 5-10 seconds on CPU

## Training Custom Models

### Prepare Training Data

Format your data as JSONL:
```json
{"prompt": "<system>Your persona</system>\n<user>User message</user>", "response": "<assistant>Assistant response</assistant>"}
```

### Tokenize Dataset

```bash
python data_prep.py --model Qwen/Qwen2.5-1.5B-Instruct --file persona_train.jsonl
```

### Train LoRA Adapter

```bash
python train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --data_dir tokenized_persona --output_dir persona_chat_lora
```

## Configuration

### Performance Modes

Edit `chat_infer.py` to change performance mode:
- `"balanced"`: 8-bit quantization, ~3GB VRAM, slower
- `"high"`: FP16 precision, ~4-5GB VRAM, faster (default)

### API Rate Limits

Configure in `api_server.py`:
- Requests per minute: 60
- Requests per hour: 500
- Burst size: 10

## Project Structure

```
.
├── app.py                 # Gradio web interface
├── chat_infer.py          # Core inference logic
├── api_server.py          # FastAPI REST server
├── discord_bot.py         # Discord bot integration
├── train_lora.py          # LoRA training script
├── data_prep.py           # Data preparation script
├── test_bot.py            # Test script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Troubleshooting

### GPU Not Detected

- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Out of Memory

- Reduce `max_tokens` in settings
- Decrease `context_window` size
- Switch to 8-bit mode in `chat_infer.py`

### Model Download Issues

- First run downloads ~3GB model files
- Ensure stable internet connection
- Model files are cached in HuggingFace cache directory

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: Please check the [Qwen2.5 model license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) for commercial use restrictions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with [Transformers](https://huggingface.co/docs/transformers) by Hugging Face
- Uses [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) model
- Web interface powered by [Gradio](https://gradio.app/)
- API framework: [FastAPI](https://fastapi.tiangolo.com/)
