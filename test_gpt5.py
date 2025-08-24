#!/usr/bin/env python3
import os
import openai
from openai import OpenAI

def test_gpt5_specifically():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return False
    
    print(f"API Key found: {api_key[:10]}...")
    
    try:
        client = OpenAI(api_key=api_key)
        
        print("Testing GPT-5 with correct parameters...")
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "user", "content": "Say 'Hello' in one word."}
            ],
            max_completion_tokens=10,  # Use max_completion_tokens instead of max_tokens
            timeout=30
        )
        
        print(f"GPT-5 SUCCESS: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"GPT-5 ERROR: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    test_gpt5_specifically()