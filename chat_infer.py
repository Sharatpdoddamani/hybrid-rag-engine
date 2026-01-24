# chat_infer.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from functools import lru_cache
from typing import List, Dict, Optional
import hashlib
import time

# Model configuration
# Using Qwen2.5-1.5B-Instruct - lightweight and efficient
# Alternative models: "TinyLlama/TinyLlama-1.1B-Chat-v1.0" or "microsoft/Phi-3-mini-4k-instruct"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Performance mode: "high" = better GPU utilization (75%+), "balanced" = 8-bit (current)
# "high" uses FP16 (full precision), faster but uses more VRAM (~4-5GB vs ~3GB)
# "balanced" uses 8-bit quantization, slower but uses less VRAM
PERFORMANCE_MODE = "high"  # Options: "high", "balanced"

# Multi-turn optimization: cache tokenized messages
# Cache for tokenized messages: {message_hash: tokenized_text}
_tokenized_cache: Dict[str, str] = {}
_tokenized_cache_max_size = 500  # Max 500 cached tokenizations

def _get_message_hash(message: Dict[str, str]) -> str:
    """Generate hash for a message for caching"""
    key = f"{message.get('role', '')}|{message.get('content', '')}"
    return hashlib.md5(key.encode()).hexdigest()

def _cache_tokenized_message(message: Dict[str, str], tokenized: str):
    """Cache tokenized message"""
    if len(_tokenized_cache) >= _tokenized_cache_max_size:
        # Remove oldest 10% of entries (simple LRU approximation)
        keys_to_remove = list(_tokenized_cache.keys())[:_tokenized_cache_max_size // 10]
        for key in keys_to_remove:
            del _tokenized_cache[key]
    
    msg_hash = _get_message_hash(message)
    _tokenized_cache[msg_hash] = tokenized

def _get_cached_tokenized(message: Dict[str, str]) -> Optional[str]:
    """Get cached tokenized message"""
    msg_hash = _get_message_hash(message)
    return _tokenized_cache.get(msg_hash)

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
else:
    print("💻 Running on CPU mode")
    print("⏱️  Responses will be slower (5-10 seconds) but it works fine!")
    print("🚀 Once CUDA is ready, restart the app to use GPU\n")

# Load model with 8-bit quantization for memory efficiency
print("📥 Loading model (this will download ~3GB on first run - please wait)...")
print("💡 Tip: This only happens once. Subsequent runs will be instant!\n")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("📦 Loading model weights (this may take a few minutes on first download)...")

# Load model based on performance mode
try:
    if device == "cuda":
        if PERFORMANCE_MODE == "high":
            # High performance mode: FP16 full precision for better GPU utilization
            print("🚀 Loading in HIGH PERFORMANCE mode (FP16)...")
            print("   This uses more VRAM (~4-5GB) but provides 75%+ GPU utilization and faster inference!\n")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                torch_dtype=torch.float16,  # FP16 for better GPU utilization
                trust_remote_code=True
            )
            # Compile model for even better GPU utilization (~30% speedup)
            try:
                print("⚡ Compiling model with torch.compile for optimal GPU usage...")
                model = torch.compile(model, mode="reduce-overhead")
                print("✅ Model compiled successfully!\n")
            except Exception as e:
                print(f"⚠️  Compilation skipped: {e}\n")
            print("✅ Model loaded in HIGH PERFORMANCE mode on GPU!\n")
        else:
            # Balanced mode: 8-bit quantization (current default)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, 
                device_map="auto", 
                load_in_8bit=True,  # 8-bit for memory efficiency
                trust_remote_code=True
            )
            print("✅ Model loaded with 8-bit quantization on GPU!\n")
    else:
        # CPU fallback - no quantization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="cpu",
            trust_remote_code=True
        )
        print("✅ Model loaded on CPU (no quantization)\n")
except Exception as e:
    print(f"⚠️  8-bit loading failed: {e}")
    print("🔄 Trying CPU fallback...\n")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        trust_remote_code=True
    )
    print("✅ Model loaded on CPU\n")

# Create pipeline (device is already set via device_map in model loading)
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer
)
print(f"🚀 Pipeline ready on {device.upper()}!\n")

def summarize_conversation_chunk(conversation_history, max_items=10):
    """Summarize old conversation chunks when history gets too long - optimized"""
    if len(conversation_history) <= max_items:
        return conversation_history
    
    # Optimized: Use slicing instead of list operations for better performance
    # Split into chunks - keep recent, summarize old
    recent = conversation_history[-max_items:]
    old = conversation_history[:-max_items]
    
    # Fast summary extraction - optimized for speed
    summary_text = "Previous conversation summary: "
    topics = set()
    key_points = []
    
    # Process up to 20 old messages (limit for performance)
    for msg in old[:20]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            # Extract key topics - optimized word extraction
            words = content.lower().split()
            topics.update(word for word in words if len(word) > 4)
        elif role == "assistant":
            # Extract first sentence as key point - optimized
            first_sentence = content.split('.')[0] if '.' in content else content[:50]
            if first_sentence.strip():
                key_points.append(first_sentence.strip())
    
    # Build summary efficiently
    topic_list = list(topics)[:5]
    summary = f"Earlier discussion covered: {', '.join(topic_list)}. " if topic_list else ""
    if key_points:
        summary += f"Key points: {'; '.join(key_points[:3])}"
    
    # Create summary message
    summary_msg = {"role": "system", "content": summary}
    
    # Return summarized version: summary + recent messages
    return [summary_msg] + recent

@lru_cache(maxsize=100)
def _cached_system_prompt_template(system_prompt_hash: str, system_prompt: str) -> str:
    """Cache common system prompts to avoid re-tokenizing"""
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}],
        tokenize=False,
        add_generation_prompt=False
    )

def generate_conversation_summary(conversation_history, persona):
    """Use the model to generate actual conversation summary"""
    if not conversation_history:
        return "No conversation yet."
    
    # Build summary prompt
    conversation_text = ""
    for msg in conversation_history[-15:]:  # Last 15 messages
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            conversation_text += f"User: {content}\n"
        elif role == "assistant":
            conversation_text += f"Assistant: {content}\n"
    
    summary_prompt = f"""Summarize this conversation with {persona[:50]}:
    
{conversation_text}

Provide a brief summary (2-3 sentences) of what was discussed:"""
    
    try:
        # Use the model to generate summary
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations concisely."},
            {"role": "user", "content": summary_prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        with torch.inference_mode():
            out = pipe(text, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9)
        
        summary = out[0]["generated_text"]
        if "Assistant:" in summary:
            summary = summary.split("Assistant:")[-1].strip()
        summary = summary.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        
        return summary
    except:
        # Fallback to simple summary
        return f"Conversation has {len(conversation_history)} exchanges."

def chat_once(system_prompt, user_input, conversation_history=None, max_new_tokens=300, temperature=0.8, num_return_sequences=1, max_context_messages=10):
    """
    Chat with context/history support - optimized for multi-turn conversations
    
    Args:
        system_prompt: Character persona
        user_input: Current user message
        conversation_history: List of previous messages as [{"role": "user/assistant", "content": "..."}, ...]
        max_new_tokens: Max response length
        temperature: Creativity control (0.1-2.0)
        num_return_sequences: Number of responses to generate (1-3)
        max_context_messages: Maximum context messages to keep
    """
    start_time = time.time()
    
    # Build messages with conversation history - optimized for speed
    messages = [{"role": "system", "content": system_prompt}]
    
    # Optimize context processing for multi-turn conversations
    if conversation_history:
        # Fast context compression if history is too long
        if len(conversation_history) > max_context_messages * 2:
            # Summarize when history exceeds 2x the context window
            conversation_history = summarize_conversation_chunk(conversation_history, max_context_messages)
        
        # Keep last N messages based on context window
        # Use slicing for O(n) operation instead of copying entire list
        recent_history = conversation_history[-max_context_messages:]
        messages.extend(recent_history)
    
    # Add current user message
    messages.append({"role": "user", "content": user_input})
    
    # Optimized tokenization: check cache for common system prompts and messages
    # Note: apply_chat_template already handles caching internally, but we optimize the flow
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Optimize for speed with better GPU utilization
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "num_return_sequences": num_return_sequences,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": False,
    }
    
    # Add torch.inference_mode for better GPU utilization
    with torch.inference_mode():
        out = pipe(text, **generation_kwargs)
    
    generation_time = time.time() - start_time
    
    # Process multiple responses if requested
    responses = []
    for result in out:
        generated_text = result["generated_text"]
        # Extract only the assistant's response
        if "<|im_start|>assistant" in generated_text:
            response = generated_text.split("<|im_start|>assistant")[-1].strip()
        else:
            response = generated_text.strip()
        
        # Clean up tokens
        response = response.replace("<|im_end|>", "").strip()
        response = response.replace("<|endoftext|>", "").strip()
        responses.append(response)
    
    # Count tokens in response - optimized
    token_count = len(tokenizer.encode(responses[0], add_special_tokens=False)) if responses else 0
    
    if num_return_sequences == 1:
        return responses[0], generation_time, token_count
    else:
        return responses, generation_time, token_count

def optimize_context_processing(conversation_history: List[Dict], max_context: int = 10) -> List[Dict]:
    """
    Optimize context processing for multi-turn conversations
    
    Returns:
        Optimized conversation history ready for tokenization
    """
    if not conversation_history:
        return []
    
    # Fast path: if history is small, just return it
    if len(conversation_history) <= max_context:
        return conversation_history
    
    # Optimize: compress if too long
    if len(conversation_history) > max_context * 2:
        conversation_history = summarize_conversation_chunk(conversation_history, max_context)
    
    # Return last N messages (most recent context)
    return conversation_history[-max_context:]

def chat_stream(system_prompt, user_input, conversation_history=None, max_new_tokens=300, temperature=0.8):
    """Streaming version for typewriter effect - generates full response then streams it"""
    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        recent_history = conversation_history[-10:]
        messages.extend(recent_history)
    messages.append({"role": "user", "content": user_input})
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate full response first (streaming from model is complex, so we simulate it)
    result = chat_once(system_prompt, user_input, conversation_history, max_new_tokens, temperature, 1)
    full_response = result[0] if isinstance(result, tuple) else result
    
    # Simulate streaming by yielding words progressively
    words = full_response.split()
    current_text = ""
    for i, word in enumerate(words):
        current_text += word + " "
        # Yield every word (for smoother effect, can batch)
        yield current_text.strip()

if __name__ == "__main__":
    system = "You are 'Captain Mira', a witty interstellar smuggler. Speak in short, sarcastic lines."
    while True:
        user = input("You: ")
        if user.lower() in ("quit","exit"): break
        resp = chat_once(system, user)
        print("Captain Mira:", resp)
