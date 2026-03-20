#!/usr/bin/env python3
"""
Basic test for OpenRouterModel integration.
Tests initialization, API call, and response printing.
"""

import os
import sys
sys.path.insert(0, '.')
from models.openrouter import OpenRouterModel

def main():
    try:
        # Initialize model (loads .env automatically)
        model = OpenRouterModel("nvidia/nemotron-3-super-120b-a12b:free")
        print("✓ OpenRouterModel initialized successfully")
        
        # Test prompt
        messages = [
            {
                "role": "user",
                "content": "Explain what an AI agent is in 2 sentences."
            }
        ]
        
        # Generate response
        print("\n🤖 Sending request to OpenRouter...")
        response = model.generate(messages)
        
        # Print response
        if "error" in response:
            print(f"❌ Error: {response['error']['message']}")
            if response['error'].get('status_code'):
                print(f"   Status code: {response['error']['status_code']}")
        else:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "No content")
            print("✅ Success!")
            print("\nResponse:")
            print("-" * 60)
            print(content)
            print("-" * 60)
            print(f"\nModel: {response.get('model', 'unknown')}")
            print(f"Tokens used: {response.get('usage', {}).get('total_tokens', 'unknown')}")
            
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
