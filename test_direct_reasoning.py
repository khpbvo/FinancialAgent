#!/usr/bin/env python3
"""
Test GPT-5 reasoning directly with the OpenAI client (bypass Agents SDK).
This will help determine if the issue is with the Agents SDK or the model itself.
"""

import asyncio
import os
from openai import AsyncOpenAI


async def test_direct_reasoning():
    """Test GPT-5 reasoning directly with OpenAI client."""

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    print("🔬 Testing GPT-5 Reasoning - Direct OpenAI Client")
    print("=" * 50)

    client = AsyncOpenAI()

    # Test with Responses API (supports reasoning streaming)
    print("Testing Responses API with reasoning streaming...")

    try:
        stream = await client.responses.create(
            model="gpt-5",
            input="What's 2+2? Think through it step by step and show your reasoning.",
            text={"verbosity": "high"},
            reasoning={"effort": "high"},
            stream=True,
        )

        print("Stream created successfully!")
        print("Receiving events:")
        print("-" * 40)

        reasoning_found = False
        event_count = 0

        async for event in stream:
            event_count += 1
            event_type = event.type

            print(f"{event_count:2d}. {event_type}")

            # Check for reasoning events
            if "reasoning" in event_type.lower():
                print(f"    🧠 REASONING EVENT: {event_type}")
                reasoning_found = True
                if hasattr(event, "delta"):
                    print(f"    Delta: {event.delta}")
                elif hasattr(event, "content"):
                    print(f"    Content: {event.content}")

            # Check for text events
            elif "text" in event_type.lower() and "delta" in event_type.lower():
                if hasattr(event, "delta"):
                    print(f"    📝 Text: {event.delta}")

            # Limit for debugging
            if event_count > 30:
                print("... (truncated after 30 events)")
                break

        print("-" * 40)
        print("📊 Results:")
        print(f"  Total events: {event_count}")
        print(f"  Reasoning events found: {'✅ Yes' if reasoning_found else '❌ No'}")

    except Exception as e:
        print(f"❌ Responses API Error: {e}")
        print("\nTrying Chat Completions API as fallback...")

        # Fallback to Chat Completions API
        try:
            stream = await client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "user",
                        "content": "What's 2+2? Think through it step by step and show your reasoning.",
                    }
                ],
                stream=True,
                # Try reasoning parameters if supported
                extra_body={
                    "reasoning": {"effort": "high"},
                    "text": {"verbosity": "high"},
                },
            )

            print("Chat Completions stream created!")
            print("Receiving chunks:")

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)

            print("\n\n✅ Chat Completions API worked (but no reasoning stream)")

        except Exception as e2:
            print(f"❌ Chat Completions API also failed: {e2}")


if __name__ == "__main__":
    asyncio.run(test_direct_reasoning())
