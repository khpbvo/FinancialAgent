#!/usr/bin/env python3
import os
from openai import OpenAI
import pytest


@pytest.mark.skip("requires OpenAI API key and network access")
def test_openai_connection():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return False

    print(f"API Key found: {api_key[:10]}...")

    try:
        client = OpenAI(api_key=api_key)

        # Test GPT-5 with Responses API (text verbosity and reasoning parameters)
        print("Testing GPT-5 with Responses API...")
        response = client.responses.create(
            model="gpt-5",
            input="Say 'Hello' in one word.",
            text={"verbosity": "high"},  # high | medium | low
            reasoning={"effort": "high"},  # high | medium | low
        )

        print(f"SUCCESS: {response.output_text}")
        if hasattr(response, "reasoning_text") and response.reasoning_text:
            print(f"Reasoning: {response.reasoning_text[:100]}...")
        return True

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

        # Try with GPT-4o as fallback
        try:
            print("Testing GPT-4o as fallback...")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Say 'Hello' in one word."}],
                max_tokens=10,
                timeout=30,
            )
            print(f"GPT-4o SUCCESS: {response.choices[0].message.content}")
            print("Consider using GPT-4o instead of GPT-5")
            return True
        except Exception as e2:
            print(f"GPT-4o also failed: {type(e2).__name__}: {e2}")
            return False


if __name__ == "__main__":
    test_openai_connection()
