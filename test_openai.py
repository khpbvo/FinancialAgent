#!/usr/bin/env python3
import os
import openai
from openai import OpenAI

def test_openai_connection():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return False
    
    print(f"API Key found: {api_key[:10]}...")
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Test with a simple request
        print("Testing GPT-5 connection...")
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "user", "content": "Say 'Hello' in one word."}
            ],
            max_tokens=10,
            timeout=30  # 30 second timeout
        )
        
        print(f"SUCCESS: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        
        # Try with GPT-4o as fallback
        try:
            print("Testing GPT-4o as fallback...")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": "Say 'Hello' in one word."}
                ],
                max_tokens=10,
                timeout=30
            )
            print(f"GPT-4o SUCCESS: {response.choices[0].message.content}")
            print("Consider using GPT-4o instead of GPT-5")
            return True
        except Exception as e2:
            print(f"GPT-4o also failed: {type(e2).__name__}: {e2}")
            return False

if __name__ == "__main__":
    test_openai_connection()