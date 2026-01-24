#!/usr/bin/env python3
"""
Simple test script to verify the persona-based chat assistant works correctly.
Run this to test your setup before using the full interface.
"""

from chat_infer import chat_once

def test_bot():
    print("🤖 Testing Persona-Based Chat Assistant...")
    print("=" * 50)
    
    # Test 1: Basic conversation
    print("\nTest 1: Basic Conversation")
    print("-" * 50)
    persona = "You are a helpful AI assistant."
    message = "Hello! How are you?"
    response = chat_once(persona, message)
    print(f"Persona: {persona}")
    print(f"Message: {message}")
    print(f"Response: {response}")
    
    # Test 2: Character persona
    print("\nTest 2: Character Persona")
    print("-" * 50)
    persona = "You are Captain Mira, a witty interstellar smuggler."
    message = "Tell me about your ship."
    response = chat_once(persona, message)
    print(f"Persona: {persona}")
    print(f"Message: {message}")
    print(f"Response: {response}")
    
    # Test 3: Fantasy scenario
    print("\nTest 3: Fantasy Scenario")
    print("-" * 50)
    persona = "You are a wise wizard who lives in a mystical tower."
    message = "What magical powers do you possess?"
    response = chat_once(persona, message)
    print(f"Persona: {persona}")
    print(f"Message: {message}")
    print(f"Response: {response}")
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\nYour bot is ready to use!")
    print("Run 'python app.py' to start the web interface.")

if __name__ == "__main__":
    try:
        test_bot()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you've installed all dependencies:")
        print("  pip install -r requirements.txt")

