# app.py
import gradio as gr
from chat_infer import chat_once, chat_stream, tokenizer
from typing import List, Tuple
import time

# Enhanced CSS with dark mode support and progress indicators
custom_css = """
/* Modern, clean design */
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    transition: background-color 0.3s ease, color 0.3s ease;
}

/* Dark mode - comprehensive theme switching */
body.dark-theme,
body.dark-theme .gradio-container,
body.dark-theme .gradio-app,
.dark-theme,
.dark-theme .gradio-container,
.dark-theme .gradio-app,
.gradio-container.dark-theme,
.gradio-container.dark-theme .gradio-app {
    background: #1a1a1a !important;
    color: #e0e0e0 !important;
}

body.dark-theme .gradio-interface,
.dark-theme .gradio-interface,
.gradio-container.dark-theme .gradio-interface {
    background: #1a1a1a !important;
    color: #e0e0e0 !important;
}

/* Dark theme textareas and inputs */
body.dark-theme textarea,
body.dark-theme input[type="text"],
.dark-theme textarea,
.dark-theme input[type="text"],
.gradio-container.dark-theme textarea,
.gradio-container.dark-theme input[type="text"],
.gradio-container.dark-theme .input-text,
.gradio-container.dark-theme .output-text {
    background: #2d2d2d !important;
    color: #e0e0e0 !important;
    border-color: #444 !important;
}

/* Dark theme chatbot */
body.dark-theme .chatbot,
.dark-theme .chatbot,
.gradio-container.dark-theme .chatbot,
.gradio-container.dark-theme .chatbot-container {
    background: #2d2d2d !important;
    color: #e0e0e0 !important;
}

/* Dark theme buttons */
body.dark-theme button,
.dark-theme button,
.gradio-container.dark-theme button {
    background: #3d3d3d !important;
    color: #e0e0e0 !important;
    border-color: #555 !important;
}

body.dark-theme button:hover,
.dark-theme button:hover,
.gradio-container.dark-theme button:hover {
    background: #4d4d4d !important;
}

/* Dark theme primary button */
body.dark-theme button.primary,
.dark-theme button.primary,
.gradio-container.dark-theme button.primary,
.gradio-container.dark-theme .primary-button {
    background: #667eea !important;
    color: white !important;
}

body.dark-theme button.primary:hover,
.dark-theme button.primary:hover,
.gradio-container.dark-theme button.primary:hover {
    background: #5568d3 !important;
}

/* Dark theme sliders */
body.dark-theme .slider-container,
.dark-theme .slider-container,
.gradio-container.dark-theme .slider-container {
    background: #2d2d2d !important;
}

/* Dark theme accordion */
body.dark-theme .accordion,
body.dark-theme details,
.dark-theme .accordion,
.dark-theme details,
.gradio-container.dark-theme .accordion,
.gradio-container.dark-theme details {
    background: #2d2d2d !important;
    border-color: #444 !important;
}

/* Dark theme markdown */
body.dark-theme .markdown,
body.dark-theme .markdown-text,
.dark-theme .markdown,
.dark-theme .markdown-text,
.gradio-container.dark-theme .markdown,
.gradio-container.dark-theme .markdown-text {
    color: #e0e0e0 !important;
}

/* Dark theme stats box */
body.dark-theme .stats-box,
.dark-theme .stats-box,
.gradio-container.dark-theme .stats-box {
    background: #2d2d2d !important;
    border-color: #444 !important;
    color: #e0e0e0 !important;
}

/* Light theme (default) - more grey */
body:not(.dark-theme),
body:not(.dark-theme) .gradio-container,
body:not(.dark-theme) .gradio-app,
.gradio-container:not(.dark-theme),
.gradio-container:not(.dark-theme) .gradio-app {
    background: #d0d1d2 !important;
    color: #000000 !important;
}

body:(.dark-theme) .gradio-interface,
.gradio-container:not(.dark-theme) .gradio-interface {
    background: #d0d1d2 !important;
    color: #000000 !important;
}

/* Light theme - black text for page elements (excluding inputs and buttons) */
body:not(.dark-theme) .markdown,
body:not(.dark-theme) .markdown-text,
body:not(.dark-theme) p,
body:not(.dark-theme) h1,
body:not(.dark-theme) h2,
body:not(.dark-theme) h3,
body:not(.dark-theme) h4,
body:not(.dark-theme) h5,
body:not(.dark-theme) h6,
body:not(.dark-theme) label:not([for]),
body:not(.dark-theme) span:not(input span):not(button span),
body:not(.dark-theme) div:not(.gradio-box):not(input):not(button):not(textarea),
.gradio-container:not(.dark-theme) .markdown,
.gradio-container:not(.dark-theme) .markdown-text,
.gradio-container:not(.dark-theme) p,
.gradio-container:not(.dark-theme) h1,
.gradio-container:not(.dark-theme) h2,
.gradio-container:not(.dark-theme) h3,
.gradio-container:not(.dark-theme) h4,
.gradio-container:not(.dark-theme) h5,
.gradio-container:not(.dark-theme) h6,
.gradio-container:not(.dark-theme) label:not([for]),
.gradio-container:not(.dark-theme) span:not(input span):not(button span),
.gradio-container:not(.dark-theme) .gradio-block,
.gradio-container:not(.dark-theme) .gradio-group,
.gradio-container:not(.dark-theme) .gradio-column {
    color: #000000 !important;
}

/* Light theme - exclude inputs, textareas, and buttons from black text */
body:not(.dark-theme) input[type="text"],
body:not(.dark-theme) textarea,
body:not(.dark-theme) button,
.gradio-container:not(.dark-theme) input[type="text"],
.gradio-container:not(.dark-theme) textarea,
.gradio-container:not(.dark-theme) button {
    color: inherit !important;
}

/* Error messages in dark theme */
body.dark-theme .error-message,
.dark-theme .error-message,
.gradio-container.dark-theme .error-message {
    background: #4a1a1a !important;
    border-color: #aa3333 !important;
    color: #ffaaaa !important;
}

/* Success messages in dark theme */
body.dark-theme .success-message,
.dark-theme .success-message,
.gradio-container.dark-theme .success-message {
    background: #1a4a1a !important;
    border-color: #33aa33 !important;
    color: #aaffaa !important;
}

/* Dark theme for all Gradio components */
body.dark-theme .gradio-block,
body.dark-theme .gradio-group,
.dark-theme .gradio-block,
.dark-theme .gradio-group {
    background: #1a1a1a !important;
    color: #e0e0e0 !important;
}
/* Loading spinner animation */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.loading-spinner {
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-right: 8px;
}
/* Progress bar */
.progress-container {
    width: 100%;
    background: #f0f0f0;
    border-radius: 10px;
    height: 20px;
    margin: 10px 0;
    overflow: hidden;
}
.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #667eea, #764ba2);
    width: 0%;
    transition: width 0.3s ease;
    border-radius: 10px;
}
/* Error messages */
.error-message {
    background: #fee;
    border: 2px solid #fcc;
    border-radius: 8px;
    padding: 12px;
    color: #c00;
    margin: 10px 0;
}
/* Success messages */
.success-message {
    background: #efe;
    border: 2px solid #cfc;
    border-radius: 8px;
    padding: 12px;
    color: #0c0;
    margin: 10px 0;
}
/* Scrollable text areas */
textarea, input[type="text"] {
    scrollbar-width: thin;
    scrollbar-color: #667eea #f0f0f0;
    max-height: 400px;
    overflow-y: auto !important;
}
textarea::-webkit-scrollbar {
    width: 8px;
}
textarea::-webkit-scrollbar-track {
    background: #f0f0f0;
    border-radius: 10px;
}
textarea::-webkit-scrollbar-thumb {
    background: #667eea;
    border-radius: 10px;
}
textarea::-webkit-scrollbar-thumb:hover {
    background: #5568d3;
}
/* Mobile responsive */
@media (max-width: 768px) {
    .gradio-container {
        padding: 10px !important;
    }
    .column {
        width: 100% !important;
    }
}
/* Stats display */
.stats-box {
    background: #e8e9ea;
    border: 1px solid #d0d1d2;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
    margin: 5px 0;
}

/* Light theme stats box */
body:not(.dark-theme) .stats-box,
.gradio-container:not(.dark-theme) .stats-box {
    background: #d5d6d7;
    border-color: #b8b9ba;
    color: #000000;
}
/* Button styling */
.secondary-button {
    margin: 3px !important;
}
"""

def respond(user_input, persona_prompt, chat_history: List[Tuple[str, str]], conversation_summary: str, 
           max_tokens: int, temperature: float, num_responses: int, enable_streaming: bool, max_context_messages: int):
    """Handle chat with conversation history and all new features"""
    if not user_input.strip():
        return chat_history, conversation_summary, "", "", ""
    
    # Show loading indicator
    loading_msg = "⏳ Generating response..."
    
    try:
        # Convert Gradio chat history format to our format
        conversation_history = []
        if chat_history:
            for user_msg, bot_msg in chat_history:
                if user_msg:
                    conversation_history.append({"role": "user", "content": user_msg})
                if bot_msg:
                    conversation_history.append({"role": "assistant", "content": bot_msg})
        
        # Get bot response with all parameters (including context window)
        if enable_streaming and num_responses == 1:
            # Streaming mode - generate full response then stream display
            result = chat_once(persona_prompt, user_input, conversation_history, max_tokens, temperature, 1, max_context_messages)
            response, gen_time, token_count = result
            
            # Simulate streaming by yielding word by word
            words = response.split()
            current_response = ""
            import time
            for word in words:
                current_response += word + " "
                # Update chat history with streaming response
                temp_history = chat_history + [(user_input, current_response.strip())]
                stats = f"{token_count} tokens | {gen_time:.2f}s"
                yield temp_history, conversation_summary, "", stats, ""
                time.sleep(0.05)  # Small delay for typewriter effect
            
            # Final update with complete response
            final_history = chat_history + [(user_input, response)]
            if len(final_history) > 5:
                conversation_summary = f"Conversation has {len(final_history)} exchanges. The bot remembers context from recent messages."
            yield final_history, conversation_summary, "", f"{token_count} tokens | {gen_time:.2f}s", ""
        else:
            # Non-streaming mode with multiple responses support
            result = chat_once(persona_prompt, user_input, conversation_history, max_tokens, temperature, num_responses, max_context_messages)
            
            if num_responses > 1:
                responses, gen_time, token_count = result
                # Return multiple responses - show all options
                response_text = "\n\n--- OR ---\n\n".join([f"**Option {i+1}:**\n{r}\n" for i, r in enumerate(responses)])
                stats = f"{token_count} tokens | {gen_time:.2f}s"
                # Use first response for chat history
                chat_history.append((user_input, responses[0]))
            else:
                response, gen_time, token_count = result
                response_text = response
                stats = f"{token_count} tokens | {gen_time:.2f}s"
                chat_history.append((user_input, response))
            
            # Update summary
            if len(chat_history) > 5:
                conversation_summary = f"Conversation has {len(chat_history)} exchanges. The bot remembers context from recent messages."
            
            yield chat_history, conversation_summary, response_text, stats, ""
            
    except Exception as e:
        # Better error handling with user-friendly messages
        error_type = type(e).__name__
        
        if "CUDA" in str(e) or "cuda" in str(e).lower():
            error_msg = "❌ GPU Error: Check CUDA installation or switch to CPU mode"
        elif "memory" in str(e).lower() or "out of memory" in str(e).lower():
            error_msg = "❌ Out of Memory: Try reducing max_tokens or context_window"
        elif "timeout" in str(e).lower():
            error_msg = "⏱️ Timeout: Generation took too long. Try again or reduce max_tokens"
        elif "connection" in str(e).lower():
            error_msg = "🌐 Connection Error: Check your internet connection"
        else:
            error_msg = f"❌ Error ({error_type}): {str(e)[:100]}"
        
        yield chat_history, conversation_summary, "", "", error_msg

def regenerate_last_response(chat_history, persona_prompt, max_tokens, temperature, max_context_messages):
    """Regenerate the last bot response"""
    if not chat_history or len(chat_history) == 0:
        error_msg = "❌ No conversation to regenerate."
        return chat_history, "", "", error_msg
    
    # Get last user message
    last_user_msg, _ = chat_history[-1]
    
    # Remove last exchange
    chat_history = chat_history[:-1]
    
    # Convert history
    conversation_history = []
    if chat_history:
        for user_msg, bot_msg in chat_history:
            if user_msg:
                conversation_history.append({"role": "user", "content": user_msg})
            if bot_msg:
                conversation_history.append({"role": "assistant", "content": bot_msg})
    
    # Generate new response
    try:
        response, gen_time, token_count = chat_once(persona_prompt, last_user_msg, conversation_history, max_tokens, temperature, 1, max_context_messages)
        chat_history.append((last_user_msg, response))
        stats = f"{token_count} tokens | {gen_time:.2f}s"
        return chat_history, response, stats, ""
    except Exception as e:
        error_msg = f"❌ Generation failed: {str(e)}"
        return chat_history, "", "", error_msg

def regenerate_response_at_index(chat_history, index, persona_prompt, max_tokens, temperature, max_context_messages):
    """Regenerate response at specific index"""
    if not chat_history or len(chat_history) == 0 or index < 0 or index >= len(chat_history):
        error_msg = "❌ Invalid index or no conversation."
        return chat_history, "", "", error_msg
    
    # Get the user message at that index
    user_msg, _ = chat_history[index]
    
    # Rebuild history up to that point
    chat_history_before = chat_history[:index]
    conversation_history = []
    
    for user_m, bot_m in chat_history_before:
        if user_m:
            conversation_history.append({"role": "user", "content": user_m})
        if bot_m:
            conversation_history.append({"role": "assistant", "content": bot_m})
    
    # Generate new response
    try:
        response, gen_time, token_count = chat_once(persona_prompt, user_msg, conversation_history, max_tokens, temperature, 1, max_context_messages)
        
        # Update that specific exchange
        new_history = chat_history.copy()
        new_history[index] = (user_msg, response)
        
        stats = f"{token_count} tokens | {gen_time:.2f}s"
        return new_history, f"✅ Regenerated response #{index + 1}", stats, ""
    except Exception as e:
        error_msg = f"❌ Failed to regenerate: {str(e)}"
        return chat_history, "", "", error_msg

def edit_and_resend(chat_history, index, edited_user_msg, persona_prompt, max_tokens, temperature, max_context_messages):
    """Edit a user message and regenerate bot response"""
    if not chat_history or len(chat_history) == 0 or index < 0 or index >= len(chat_history):
        error_msg = "❌ Invalid index or no conversation."
        return chat_history, "", "", error_msg
    
    if not edited_user_msg.strip():
        error_msg = "❌ Please enter an edited message."
        return chat_history, "", "", error_msg
    
    # Rebuild history up to that point (excluding the exchange being edited)
    chat_history_before = chat_history[:index]
    conversation_history = []
    
    for user_m, bot_m in chat_history_before:
        if user_m:
            conversation_history.append({"role": "user", "content": user_m})
        if bot_m:
            conversation_history.append({"role": "assistant", "content": bot_m})
    
    # Generate new response for edited message
    try:
        response, gen_time, token_count = chat_once(persona_prompt, edited_user_msg, conversation_history, max_tokens, temperature, 1, max_context_messages)
        
        # Update history: replace old exchange and remove any after it
        new_history = chat_history_before + [(edited_user_msg, response)]
        
        stats = f"{token_count} tokens | {gen_time:.2f}s"
        return new_history, f"✅ Edited and regenerated exchange #{index + 1}", stats, ""
    except Exception as e:
        error_msg = f"❌ Failed to edit and resend: {str(e)}"
        return chat_history, "", "", error_msg

def search_conversation(chat_history, search_query):
    """Search within conversation history"""
    if not search_query.strip() or not chat_history:
        return ""
    
    results = []
    search_lower = search_query.lower()
    
    for i, (user_msg, bot_msg) in enumerate(chat_history):
        if search_lower in user_msg.lower() or search_lower in bot_msg.lower():
            context = f"[Exchange {i+1}] User: {user_msg[:100]}... | Bot: {bot_msg[:100]}..."
            results.append(context)
    
    if results:
        return "\n\n".join(results[:10])  # Limit to 10 results
    else:
        return "No matches found."

def clear_history():
    """Clear conversation history and summary"""
    return [], "", "", "", "", ""

def create_summary(chat_history: List[Tuple[str, str]], persona: str):
    """Create a summary using the model for actual summarization"""
    from chat_infer import generate_conversation_summary
    
    if not chat_history:
        return "No conversation yet. Start chatting to build context!"
    
    # Convert to conversation history format
    conversation_history = []
    for user_msg, bot_msg in chat_history:
        if user_msg:
            conversation_history.append({"role": "user", "content": user_msg})
        if bot_msg:
            conversation_history.append({"role": "assistant", "content": bot_msg})
    
    # Generate actual summary using the model
    summary_text = generate_conversation_summary(conversation_history, persona)
    
    # Format nicely
    total_exchanges = len(chat_history)
    summary = f"**Conversation Summary:**\n\n"
    summary += f"📊 **Total Exchanges:** {total_exchanges}\n\n"
    summary += f"🎭 **Character:** {persona[:50]}...\n\n"
    summary += f"📝 **Summary:** {summary_text}\n\n"
    summary += f"💡 **Context:** The bot remembers and compresses context as needed."
    
    return summary

def save_conversation_to_memory(chat_history, persona):
    """Save important facts to long-term memory file"""
    import json
    import os
    from datetime import datetime
    
    if not chat_history or len(chat_history) == 0:
        return "No conversation to save."
    
    # Extract key facts from conversation
    facts = []
    for user_msg, bot_msg in chat_history[-5:]:  # Last 5 exchanges
        if bot_msg:
            # Simple extraction: sentences with names or important info
            sentences = bot_msg.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and any(word[0].isupper() for word in sentence.split()):
                    facts.append(sentence)
    
    memory_data = {
        "timestamp": datetime.now().isoformat(),
        "persona": persona[:200],
        "total_exchanges": len(chat_history),
        "key_facts": facts[:10],  # Top 10 facts
        "last_messages": chat_history[-3:]  # Last 3 messages
    }
    
    # Save to memory file
    memory_file = "conversation_memory.json"
    
    # Load existing memory
    if os.path.exists(memory_file):
        with open(memory_file, 'r', encoding='utf-8') as f:
            all_memories = json.load(f)
    else:
        all_memories = []
    
    # Add new memory (keep last 10 sessions)
    all_memories.append(memory_data)
    if len(all_memories) > 10:
        all_memories = all_memories[-10:]
    
    # Save
    with open(memory_file, 'w', encoding='utf-8') as f:
        json.dump(all_memories, f, indent=2, ensure_ascii=False)
    
    return f"✅ Saved {len(facts)} key facts to memory!"

def load_conversation_from_memory():
    """Load important facts from long-term memory"""
    import json
    import os
    
    memory_file = "conversation_memory.json"
    
    if not os.path.exists(memory_file):
        return [], "No saved memories found."
    
    try:
        with open(memory_file, 'r', encoding='utf-8') as f:
            all_memories = json.load(f)
        
        if not all_memories:
            return [], "No saved memories found."
        
        # Get most recent memory
        latest = all_memories[-1]
        last_messages = latest.get("last_messages", [])
        info = f"Loaded memory from {latest.get('timestamp', 'unknown')}: {len(latest.get('key_facts', []))} facts"
        
        return last_messages, info
    except Exception as e:
        return [], f"Error loading memory: {str(e)}"

def save_conversation(chat_history, persona, format_type="json"):
    """Save conversation to file (JSON or TXT)"""
    import json
    import os
    from datetime import datetime
    
    if not chat_history or len(chat_history) == 0:
        return "No conversation to save."
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format_type == "json":
        filename = f"conversation_{timestamp}.json"
        data = {
            "timestamp": datetime.now().isoformat(),
            "persona": persona,
            "total_exchanges": len(chat_history),
            "conversation": chat_history
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return f"✅ Saved to {filename}"
    
    else:  # txt
        filename = f"conversation_{timestamp}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Persona: {persona}\n")
            f.write(f"Saved: {datetime.now().isoformat()}\n")
            f.write(f"Total Exchanges: {len(chat_history)}\n")
            f.write("\n" + "="*50 + "\n\n")
            for i, (user_msg, bot_msg) in enumerate(chat_history, 1):
                f.write(f"[Exchange {i}]\n")
                f.write(f"User: {user_msg}\n")
                f.write(f"Bot: {bot_msg}\n\n")
        return f"✅ Saved to {filename}"

def load_conversation_from_file(file):
    """Load conversation from uploaded file"""
    import json
    
    if file is None:
        return [], "", "No file uploaded."
    
    try:
        # Read file content
        with open(file.name, 'r', encoding='utf-8') as f:
            if file.name.endswith('.json'):
                data = json.load(f)
                chat_history = data.get("conversation", [])
                persona = data.get("persona", "You are a helpful conversational assistant.")
                info = f"✅ Loaded {len(chat_history)} exchanges from {file.name}"
            else:  # txt
                content = f.read()
                # Simple parsing - could be improved
                lines = content.split('\n')
                chat_history = []
                persona = ""
                for line in lines:
                    if line.startswith("Persona:"):
                        persona = line.replace("Persona:", "").strip()
                    elif line.startswith("User:"):
                        user_msg = line.replace("User:", "").strip()
                        chat_history.append((user_msg, ""))
                    elif line.startswith("Bot:"):
                        bot_msg = line.replace("Bot:", "").strip()
                        if chat_history:
                            chat_history[-1] = (chat_history[-1][0], bot_msg)
                info = f"✅ Loaded {len(chat_history)} exchanges from {file.name}"
        
        return chat_history, persona, info
    except Exception as e:
        return [], "", f"❌ Error loading file: {str(e)}"

def get_conversation_statistics(chat_history):
    """Generate conversation statistics"""
    from chat_infer import tokenizer
    
    if not chat_history or len(chat_history) == 0:
        return "No conversation yet. Start chatting to see statistics!"
    
    total_exchanges = len(chat_history)
    total_tokens = 0
    total_words = 0
    user_lengths = []
    bot_lengths = []
    topics = set()
    
    for user_msg, bot_msg in chat_history:
        if user_msg:
            user_tokens = len(tokenizer.encode(user_msg))
            user_words = len(user_msg.split())
            total_tokens += user_tokens
            total_words += user_words
            user_lengths.append(user_words)
            # Extract topics (capitalized words longer than 4 chars)
            words = user_msg.split()
            for word in words:
                if len(word) > 4 and word[0].isupper():
                    topics.add(word)
        
        if bot_msg:
            bot_tokens = len(tokenizer.encode(bot_msg))
            bot_words = len(bot_msg.split())
            total_tokens += bot_tokens
            total_words += bot_words
            bot_lengths.append(bot_words)
    
    avg_user_length = sum(user_lengths) / len(user_lengths) if user_lengths else 0
    avg_bot_length = sum(bot_lengths) / len(bot_lengths) if bot_lengths else 0
    
    stats = f"**📊 Conversation Statistics**\n\n"
    stats += f"💬 **Total Exchanges:** {total_exchanges}\n"
    stats += f"📝 **Total Tokens:** {total_tokens:,}\n"
    stats += f"📄 **Total Words:** {total_words:,}\n\n"
    stats += f"👤 **Average User Message:** {avg_user_length:.1f} words\n"
    stats += f"🤖 **Average Bot Message:** {avg_bot_length:.1f} words\n\n"
    
    if topics:
        stats += f"🎯 **Topics Discussed:** {', '.join(list(topics)[:10])}\n\n"
    
    stats += f"📈 **Longest User Message:** {max(user_lengths) if user_lengths else 0} words\n"
    stats += f"📈 **Longest Bot Message:** {max(bot_lengths) if bot_lengths else 0} words"
    
    return stats

def save_character_preset(persona, preset_name):
    """Save character persona as preset"""
    import json
    import os
    
    if not persona.strip() or not preset_name.strip():
        return "Please provide both persona and preset name."
    
    preset_file = "character_presets.json"
    
    # Load existing presets
    if os.path.exists(preset_file):
        with open(preset_file, 'r', encoding='utf-8') as f:
            presets = json.load(f)
    else:
        presets = {}
    
    # Add new preset
    presets[preset_name] = persona
    
    # Save
    with open(preset_file, 'w', encoding='utf-8') as f:
        json.dump(presets, f, indent=2, ensure_ascii=False)
    
    return f"✅ Saved preset '{preset_name}'"

def load_character_presets():
    """Load all character presets"""
    import json
    import os
    
    preset_file = "character_presets.json"
    
    if not os.path.exists(preset_file):
        return {}
    
    try:
        with open(preset_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

CONVERSATION_TEMPLATES = {
    "Adventure Quest": {
        "persona": "You are a brave adventurer exploring a mysterious dungeon. You're courageous, curious, and always ready for a challenge.",
        "starter": "I've entered an ancient dungeon. What do I see ahead?"
    },
    "Space Mission": {
        "persona": "You are Captain Nova, commander of a deep space exploration vessel. You're professional, strategic, and committed to your mission.",
        "starter": "Status report: We've detected an unknown signal. What should we do?"
    },
    "Fantasy RPG": {
        "persona": "You are a wise wizard living in a magical realm. You speak in mystical terms and have knowledge of ancient magic.",
        "starter": "Tell me about the magical forces at work in this realm."
    },
    "Mystery Detective": {
        "persona": "You are a sharp detective solving crimes. You're observant, logical, and notice details others miss.",
        "starter": "I've found a clue at the crime scene. Help me analyze it."
    },
    "Modern Chat": {
        "persona": "You are a friendly, modern person who loves having conversations. You're relatable, humorous, and easy to talk to.",
        "starter": "Hey! How's your day going?"
    }
}

# Create interface with enhanced features - supports both light and dark themes
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as iface:
    # Hidden component to apply dark theme on initial load - using inline script
    # This script runs immediately and applies dark theme on page load
    theme_init = gr.HTML(
        value="""<script>
        (function() {
            function applyDarkTheme() {
                try {
                    const containers = document.querySelectorAll('.gradio-container, .gradio-app, body, html');
                    containers.forEach(el => {
                        if (el) el.classList.add('dark-theme');
                    });
                    if (document.body) document.body.classList.add('dark-theme');
                    if (document.documentElement) document.documentElement.classList.add('dark-theme');
                } catch(e) {}
            }
            
            // Apply immediately
            applyDarkTheme();
            
            // Apply when DOM is ready
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', function() {
                    applyDarkTheme();
                    setTimeout(applyDarkTheme, 50);
                    setTimeout(applyDarkTheme, 200);
                    setTimeout(applyDarkTheme, 500);
                });
            } else {
                setTimeout(applyDarkTheme, 50);
                setTimeout(applyDarkTheme, 200);
                setTimeout(applyDarkTheme, 500);
            }
            
            // Also use MutationObserver to catch dynamically added elements
            if (window.MutationObserver) {
                const observer = new MutationObserver(function(mutations) {
                    applyDarkTheme();
                });
                setTimeout(function() {
                    if (document.body) {
                        observer.observe(document.body, { childList: true, subtree: true });
                    }
                }, 100);
            }
            
            // Also apply on window load
            window.addEventListener('load', function() {
                applyDarkTheme();
                setTimeout(applyDarkTheme, 100);
            });
        })();
        </script>""",
        visible=False
    )
    
    gr.Markdown(
        """
        # 🤖 Persona-Based Chat Assistant
        
        **Lightweight AI-powered conversational assistant running Qwen2.5-1.5B**
        
        Create custom character personas and engage in contextual conversations! Customize the persona below to define the assistant's personality and behavior.
        """,
        elem_classes="header"
    )
    
    # Theme toggle and settings
    with gr.Row():
        theme_toggle = gr.Radio(
            choices=["Light", "Dark"],
            value="Dark",
            label="🌓 Theme",
            interactive=True
        )
        with gr.Column(scale=3):
            pass  # Spacer
    
    with gr.Row():
        with gr.Column(scale=1):
            persona_box = gr.Textbox(
                label="🎭 Character Persona",
                placeholder="Describe the character's personality, background, and speaking style...",
                value="You are a helpful, creative conversational assistant. Respond in character to create engaging conversations.",
                lines=4,
                max_lines=10,
                elem_classes=["scrollable-text"]
            )
            
            gr.Markdown("**💡 Tip:** Be specific about personality traits, speaking style, and background for better responses!")
            
            examples = gr.Examples(
                examples=[
                    ["You are Captain Mira, a witty interstellar smuggler with a mysterious past. You're sarcastic, street-smart, and always looking for the next job.", ""],
                    ["You are a wise ancient wizard named Gandalf, living in a mystical tower. You speak in riddles, offer sage advice, and have powerful magic at your command.", ""],
                    ["You are a friendly barista who loves chatting with customers. You're enthusiastic about coffee and always have a story to share.", ""],
                    ["You are a skilled cyberpunk hacker living in Neo-Tokyo. You're tech-savvy, secretive, and always have your own agenda.", ""],
                ],
                inputs=persona_box,
                label="📚 Example Personas (Click to use)",
            )
        
        with gr.Column(scale=2):
            # Chat history display
            chatbot = gr.Chatbot(
                label="💬 Conversation History",
                height=400,
                show_label=True,
                container=True,
            )
            
            # Search bar
            with gr.Row():
                search_box = gr.Textbox(
                    label="🔍 Search Conversation",
                    placeholder="Type to search messages...",
                    scale=4,
                    show_label=False
                )
                search_btn = gr.Button("🔍", scale=1)
            
            search_results = gr.Textbox(
                label="Search Results",
                lines=3,
                visible=False,
                interactive=False
            )
            
            with gr.Row():
                user_input = gr.Textbox(
                    label="",
                    placeholder="Type your message here... (Ctrl+Enter to send)",
                    lines=2,
                    max_lines=5,
                    scale=4,
                    container=False,
                    show_label=False
                )
                send_btn = gr.Button("Send ✉️", variant="primary", scale=1)
            
            # Control buttons
            with gr.Row():
                regenerate_btn = gr.Button("🔄 Regenerate Last", variant="secondary", scale=1)
                clear_btn = gr.Button("🗑️ Clear History", variant="secondary", scale=1)
                summary_btn = gr.Button("📋 Generate Summary", variant="secondary", scale=1)
            
            # Edit & Regenerate section
            with gr.Accordion("✏️ Edit & Regenerate Messages", open=False):
                with gr.Row():
                    message_index = gr.Number(
                        label="Message # (1 = first message)",
                        value=1,
                        minimum=1,
                        precision=0,
                        scale=2
                    )
                    regenerate_at_index_btn = gr.Button("🔄 Regenerate This", variant="secondary", scale=1)
                
                with gr.Row():
                    edit_message_input = gr.Textbox(
                        label="Edit User Message",
                        placeholder="Type edited message here...",
                        lines=2,
                        scale=3
                    )
                    edit_and_resend_btn = gr.Button("✏️ Edit & Resend", variant="secondary", scale=1)
                
                edit_status = gr.Textbox(
                    label="",
                    value="",
                    interactive=False,
                    lines=1
                )
            
            # Settings row
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    max_tokens_slider = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=300,
                        step=50,
                        label="📏 Max Response Length (tokens)",
                        info="Increase for longer responses"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        label="🌡️ Creativity (Temperature)",
                        info="Higher = more creative"
                    )
                
                with gr.Row():
                    num_responses_slider = gr.Slider(
                        minimum=1,
                        maximum=3,
                        value=1,
                        step=1,
                        label="📝 Number of Responses",
                        info="Generate multiple options to choose from"
                    )
                    streaming_toggle = gr.Checkbox(
                        label="⚡ Enable Streaming (Typewriter Effect)",
                        value=True,
                        info="See text appear word by word"
                    )
                
                with gr.Row():
                    context_window_slider = gr.Slider(
                        minimum=5,
                        maximum=30,
                        value=10,
                        step=5,
                        label="🧠 Context Window (Messages)",
                        info="How many past messages to remember (auto-compresses if too long)"
                    )
                    
            with gr.Row():
                save_memory_btn = gr.Button("💾 Save to Memory", variant="secondary", scale=1)
                load_memory_btn = gr.Button("📂 Load from Memory", variant="secondary", scale=1)
                memory_status = gr.Textbox(
                    label="",
                    value="",
                    interactive=False,
                    visible=True,
                    lines=1,
                    scale=2
                )
            
            # Conversation management section
            with gr.Accordion("💾 Conversation Management", open=False):
                # Save conversation
                with gr.Row():
                    save_format = gr.Radio(
                        choices=["JSON", "TXT"],
                        value="JSON",
                        label="Format",
                        scale=1
                    )
                    save_conv_btn = gr.Button("💾 Save Conversation", variant="secondary", scale=2)
                    save_status = gr.Textbox(
                        label="",
                        value="",
                        interactive=False,
                        lines=1,
                        scale=2
                    )
                
                # Load conversation
                load_file = gr.File(
                    label="📂 Load Conversation",
                    file_types=[".json", ".txt"],
                    scale=2
                )
                load_conv_btn = gr.Button("📂 Load from File", variant="secondary", scale=1)
                load_status = gr.Textbox(
                    label="",
                    value="",
                    interactive=False,
                    lines=1
                )
                
                # Statistics
                stats_btn = gr.Button("📊 Show Statistics", variant="secondary")
                stats_display_text = gr.Markdown(
                    value="",
                    label="📊 Conversation Statistics",
                    visible=True
                )
            
            # Character presets section
            with gr.Accordion("🎭 Character Presets", open=False):
                with gr.Row():
                    preset_name = gr.Textbox(
                        label="Preset Name",
                        placeholder="e.g., 'Captain Mira'",
                        scale=2
                    )
                    save_preset_btn = gr.Button("💾 Save Preset", variant="secondary", scale=1)
                
                preset_status = gr.Textbox(
                    label="",
                    value="",
                    interactive=False,
                    lines=1
                )
                
                # Load preset dropdown
                presets_dropdown = gr.Dropdown(
                    choices=list(load_character_presets().keys()) + list(CONVERSATION_TEMPLATES.keys()),
                    label="📚 Load Preset/Template",
                    interactive=True
                )
                load_preset_btn = gr.Button("📂 Load Preset", variant="secondary")
            
            # Conversation templates
            with gr.Accordion("📋 Conversation Templates", open=False):
                template_dropdown = gr.Dropdown(
                    choices=list(CONVERSATION_TEMPLATES.keys()),
                    label="Select Template",
                    interactive=True
                )
                
                template_info = gr.Markdown(
                    value="Select a template to see details and quick-start options."
                )
                
                template_persona_display = gr.Textbox(
                    label="Template Persona",
                    interactive=False,
                    lines=3
                )
                
                use_template_btn = gr.Button("🚀 Use This Template", variant="primary")
            
            # Stats display
            stats_display = gr.Textbox(
                label="📊 Response Stats",
                value="",
                interactive=False,
                visible=True,
                elem_classes=["stats-box"],
                lines=1
            )
            
            # Conversation summary
            summary_box = gr.Markdown(
                value="**Conversation Summary:**\n\nNo conversation yet. Start chatting to build context!",
                label="📊 Context Summary",
                visible=True
            )
            
            # Error display (shows when needed)
            error_display = gr.Textbox(
                label="",
                value="",
                interactive=False,
                visible=True,
                lines=2,
                elem_classes=["error-message"]
            )
            
            # Progress indicator
            progress_indicator = gr.HTML(
                value="",
                visible=False,
                elem_classes=["progress-container"]
            )
            
    
    # Examples for quick start
    with gr.Accordion("🚀 Quick Start Examples", open=False):
        gr.Examples(
            examples=[
                ["Hello! I'm new to this realm. Can you help me?", "You are a wise ancient wizard living in a mystical tower."],
                ["Tell me about your adventures!", "You are Captain Mira, a witty interstellar smuggler with a mysterious past."],
                ["What's your favorite coffee?", "You are a friendly barista who loves chatting with customers."],
            ],
            inputs=[user_input, persona_box],
            label="Click any example to load it",
        )
    
    # Footer with summary
    with gr.Accordion("📊 Conversation Context & Tips", open=False):
        summary_box
        
        gr.Markdown(
            """
            **💡 Tips:**
            - Use **Ctrl+Enter** to send message quickly
            - **Regenerate** button creates a new response for the last message
            - **Multiple responses** generates 2-3 options to choose from
            - **Streaming mode** shows text appearing word-by-word
            - Adjust **creativity** slider for more/less creative responses
            - **Search** helps find specific messages in long conversations
            """
        )
    
    # Store conversation state
    chat_history_state = gr.State([])
    conversation_summary_state = gr.State("")
    
    # Connect functions
    def handle_send(user_input, persona, history, summary, max_tokens, temperature, num_responses, streaming, max_context):
        # respond() is a generator, so we need to yield from it
        for result in respond(user_input, persona, history, summary, max_tokens, temperature, num_responses, streaming, max_context):
            yield result
    
    # Show loading indicator
    def show_loading():
        return '<div class="loading-spinner"></div> Generating response... Please wait.'
    
    def hide_loading():
        return ""
    
    send_btn.click(
        fn=show_loading,
        inputs=[],
        outputs=[progress_indicator]
    ).then(
        fn=lambda: gr.update(visible=True),
        inputs=[],
        outputs=[progress_indicator]
    ).then(
        fn=handle_send,
        inputs=[user_input, persona_box, chat_history_state, conversation_summary_state, 
                max_tokens_slider, temperature_slider, num_responses_slider, streaming_toggle, context_window_slider],
        outputs=[chatbot, conversation_summary_state, user_input, stats_display, error_display]
    ).then(
        fn=hide_loading,
        inputs=[],
        outputs=[progress_indicator]
    ).then(
        fn=lambda: gr.update(visible=False),
        inputs=[],
        outputs=[progress_indicator]
    ).then(
        fn=lambda x: x,  # Update history state
        inputs=[chatbot],
        outputs=[chat_history_state]
    )
    
    # Keyboard shortcut: Enter key
    user_input.submit(
        fn=show_loading,
        inputs=[],
        outputs=[progress_indicator]
    ).then(
        fn=lambda: gr.update(visible=True),
        inputs=[],
        outputs=[progress_indicator]
    ).then(
        fn=handle_send,
        inputs=[user_input, persona_box, chat_history_state, conversation_summary_state,
                max_tokens_slider, temperature_slider, num_responses_slider, streaming_toggle, context_window_slider],
        outputs=[chatbot, conversation_summary_state, user_input, stats_display, error_display]
    ).then(
        fn=hide_loading,
        inputs=[],
        outputs=[progress_indicator]
    ).then(
        fn=lambda: gr.update(visible=False),
        inputs=[],
        outputs=[progress_indicator]
    ).then(
        fn=lambda x: x,
        inputs=[chatbot],
        outputs=[chat_history_state]
    )
    
    # Regenerate last response
    def handle_regenerate(history, persona, max_tokens, temperature, max_context):
        new_history, response, stats, error = regenerate_last_response(history, persona, max_tokens, temperature, max_context)
        if error:
            return new_history, new_history, "", stats, error
        return new_history, new_history, "", stats, ""
    
    regenerate_btn.click(
        fn=handle_regenerate,
        inputs=[chat_history_state, persona_box, max_tokens_slider, temperature_slider, context_window_slider],
        outputs=[chat_history_state, chatbot, user_input, stats_display, error_display]
    )
    
    # Regenerate response at specific index
    def handle_regenerate_at_index(history, index, persona, max_tokens, temperature, max_context):
        idx = int(index) - 1  # Convert to 0-based index
        new_history, status, stats, error = regenerate_response_at_index(history, idx, persona, max_tokens, temperature, max_context)
        if error:
            return new_history, new_history, "", stats, error, status
        return new_history, new_history, "", stats, "", status
    
    regenerate_at_index_btn.click(
        fn=handle_regenerate_at_index,
        inputs=[chat_history_state, message_index, persona_box, max_tokens_slider, temperature_slider, context_window_slider],
        outputs=[chat_history_state, chatbot, user_input, stats_display, error_display, edit_status]
    )
    
    # Edit and resend
    def handle_edit_and_resend(history, index, edited_msg, persona, max_tokens, temperature, max_context):
        idx = int(index) - 1  # Convert to 0-based index
        new_history, status, stats, error = edit_and_resend(history, idx, edited_msg, persona, max_tokens, temperature, max_context)
        if error:
            return new_history, new_history, "", stats, error, status
        return new_history, new_history, "", stats, "", status
    
    edit_and_resend_btn.click(
        fn=handle_edit_and_resend,
        inputs=[chat_history_state, message_index, edit_message_input, persona_box, max_tokens_slider, temperature_slider, context_window_slider],
        outputs=[chat_history_state, chatbot, user_input, stats_display, error_display, edit_status]
    )
    
    # Save to memory
    def handle_save_memory(history, persona):
        result = save_conversation_to_memory(history, persona)
        return result
    
    save_memory_btn.click(
        fn=handle_save_memory,
        inputs=[chat_history_state, persona_box],
        outputs=[memory_status]
    )
    
    # Load from memory
    def handle_load_memory():
        loaded_history, info = load_conversation_from_memory()
        return loaded_history, info
    
    load_memory_btn.click(
        fn=handle_load_memory,
        inputs=[],
        outputs=[chat_history_state, memory_status]
    ).then(
        fn=lambda x: x,
        inputs=[chat_history_state],
        outputs=[chatbot]
    )
    
    # Clear history
    def handle_clear(persona):
        cleared_history, cleared_summary, _, _, _, _ = clear_history()
        summary_text = create_summary([], persona)
        return cleared_history, cleared_summary, summary_text, cleared_history, "", ""
    
    clear_btn.click(
        fn=handle_clear,
        inputs=[persona_box],
        outputs=[chat_history_state, conversation_summary_state, summary_box, chatbot, stats_display, error_display]
    )
    
    # Generate summary
    def handle_summary(history, persona):
        summary_text = create_summary(history, persona)
        return summary_text
    
    summary_btn.click(
        fn=handle_summary,
        inputs=[chat_history_state, persona_box],
        outputs=[summary_box]
    )
    
    # Search conversation
    def handle_search(history, query):
        results = search_conversation(history, query)
        return results, gr.update(visible=True) if results else gr.update(visible=False)
    
    search_btn.click(
        fn=handle_search,
        inputs=[chat_history_state, search_box],
        outputs=[search_results, search_results]
    )
    
    search_box.submit(
        fn=handle_search,
        inputs=[chat_history_state, search_box],
        outputs=[search_results, search_results]
    )
    
    # Theme toggle - using Gradio's built-in JavaScript execution
    def apply_theme_js(theme):
        """Return JavaScript code to apply theme"""
        if theme == "Dark":
            return """
            () => {
                const containers = document.querySelectorAll('.gradio-container, .gradio-app, body');
                containers.forEach(el => {
                    if (el) el.classList.add('dark-theme');
                });
                // Also apply to body for global dark mode
                document.body.classList.add('dark-theme');
            }
            """
        else:
            return """
            () => {
                const containers = document.querySelectorAll('.gradio-container, .gradio-app, body');
                containers.forEach(el => {
                    if (el) el.classList.remove('dark-theme');
                });
                // Also remove from body
                document.body.classList.remove('dark-theme');
            }
            """
    
    # Theme toggle with proper JavaScript execution
    theme_toggle.change(
        fn=None,
        inputs=[theme_toggle],
        outputs=[],
        js="""
        (theme) => {
            if (theme === 'Dark') {
                const containers = document.querySelectorAll('.gradio-container, .gradio-app, body, html');
                containers.forEach(el => {
                    if (el) el.classList.add('dark-theme');
                });
                if (document.body) document.body.classList.add('dark-theme');
                if (document.documentElement) document.documentElement.classList.add('dark-theme');
            } else {
                const containers = document.querySelectorAll('.gradio-container, .gradio-app, body, html');
                containers.forEach(el => {
                    if (el) el.classList.remove('dark-theme');
                });
                if (document.body) document.body.classList.remove('dark-theme');
                if (document.documentElement) document.documentElement.classList.remove('dark-theme');
            }
        }
        """
    )
    
    # Dark theme is applied via the theme_init HTML component's inline script
    # The script runs on DOMContentLoaded and uses MutationObserver for dynamic elements
    
    # Save conversation
    def handle_save_conversation(history, persona, format_type):
        result = save_conversation(history, persona, format_type.lower())
        return result
    
    save_conv_btn.click(
        fn=handle_save_conversation,
        inputs=[chat_history_state, persona_box, save_format],
        outputs=[save_status]
    )
    
    # Load conversation
    def handle_load_conversation(file):
        loaded_history, loaded_persona, info = load_conversation_from_file(file)
        return loaded_history, loaded_persona, info
    
    load_conv_btn.click(
        fn=handle_load_conversation,
        inputs=[load_file],
        outputs=[chat_history_state, persona_box, load_status]
    ).then(
        fn=lambda x: x,
        inputs=[chat_history_state],
        outputs=[chatbot]
    )
    
    # Statistics
    def handle_statistics(history):
        stats = get_conversation_statistics(history)
        return stats
    
    stats_btn.click(
        fn=handle_statistics,
        inputs=[chat_history_state],
        outputs=[stats_display_text]
    )
    
    # Save character preset
    def handle_save_preset(persona, name):
        result = save_character_preset(persona, name)
        # Update dropdown
        presets = load_character_presets()
        choices = list(presets.keys()) + list(CONVERSATION_TEMPLATES.keys())
        return result, gr.update(choices=choices)
    
    save_preset_btn.click(
        fn=handle_save_preset,
        inputs=[persona_box, preset_name],
        outputs=[preset_status, presets_dropdown]
    )
    
    # Load preset
    def handle_load_preset(preset_name):
        if not preset_name:
            return "", "", "Please select a preset."
        
        presets = load_character_presets()
        
        # Check if it's a template
        if preset_name in CONVERSATION_TEMPLATES:
            template = CONVERSATION_TEMPLATES[preset_name]
            persona = template["persona"]
            starter = template.get("starter", "")
        elif preset_name in presets:
            persona = presets[preset_name]
            starter = ""
        else:
            return "", "", "Preset not found."
        
        return persona, starter, f"✅ Loaded preset: {preset_name}"
    
    load_preset_btn.click(
        fn=handle_load_preset,
        inputs=[presets_dropdown],
        outputs=[persona_box, user_input, preset_status]
    )
    
    # Template selection
    def handle_template_select(template_name):
        if not template_name or template_name not in CONVERSATION_TEMPLATES:
            return "Select a template to see details.", "", ""
        
        template = CONVERSATION_TEMPLATES[template_name]
        info = f"**{template_name}**\n\n"
        info += f"**Persona:** {template['persona']}\n\n"
        info += f"**Starter Message:** {template.get('starter', '')}\n\n"
        info += "Click 'Use This Template' to load it."
        
        return info, template['persona'], template.get('starter', '')
    
    template_dropdown.change(
        fn=handle_template_select,
        inputs=[template_dropdown],
        outputs=[template_info, template_persona_display, user_input]
    )
    
    # Use template
    def handle_use_template(template_name):
        if not template_name or template_name not in CONVERSATION_TEMPLATES:
            return "", ""
        
        template = CONVERSATION_TEMPLATES[template_name]
        return template['persona'], template.get('starter', '')
    
    use_template_btn.click(
        fn=handle_use_template,
        inputs=[template_dropdown],
        outputs=[persona_box, user_input]
    )

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Starting Persona-Based Chat Assistant...")
    print("="*50)
    print("\n📱 Open your browser and go to:")
    print("   http://localhost:7860")
    print("   OR")
    print("   http://127.0.0.1:7860")
    print("\n✨ Features:")
    print("   - Streaming responses (typewriter effect)")
    print("   - Regenerate any past response")
    print("   - Edit and resend user messages")
    print("   - Better error handling with user-friendly messages")
    print("   - Progress indicators during generation")
    print("   - Multiple response options")
    print("   - Adjustable creativity slider")
    print("   - Response timer & token counter")
    print("   - Conversation search")
    print("   - Keyboard shortcuts (Ctrl+Enter)")
    print("   - Dark mode toggle")
    print("   - Mobile-friendly design")
    print("   - Save/load conversations")
    print("   - Character presets & templates")
    print("   - Conversation statistics")
    print("="*50 + "\n")
    
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
