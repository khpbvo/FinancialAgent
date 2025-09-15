#!/usr/bin/env python3
"""
Simple debug script to see what events we actually receive from GPT-5.
"""

import asyncio
import os
from agents import Agent, Runner, ModelSettings
from openai.types.shared import Reasoning


async def debug_events():
    """Debug what events we actually get from GPT-5."""

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    print("🔍 Debugging GPT-5 Events")
    print("=" * 40)

    # Simple agent with high reasoning
    agent = Agent(
        name="DebugAgent",
        instructions="You are a helpful assistant.",
        model="gpt-5",
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="high"), verbosity="high"
        ),
    )

    print("Agent created with:")
    print(f"  Model: {agent.model}")
    print("  Reasoning effort: high")
    print("  Verbosity: high")
    print("-" * 40)

    # Simple query
    query = "What's 2+2? Think through it step by step."
    print(f"Query: {query}")
    print("-" * 40)

    try:
        result = Runner.run_streamed(agent, input=query)

        print("\nEvents received:")
        event_count = 0

        async for event in result.stream_events():
            event_count += 1
            print(f"{event_count:2d}. Event Type: {event.type}")

            if event.type == "raw_response_event":
                data_type = type(event.data).__name__
                print(f"    Raw Event Data Type: {data_type}")

                # Check for reasoning events specifically
                if "reasoning" in data_type.lower():
                    print(f"    🧠 REASONING EVENT FOUND: {data_type}")
                    if hasattr(event.data, "delta"):
                        print(f"    Delta: {event.data.delta}")

                # Show text events
                elif "text" in data_type.lower():
                    print(f"    📝 TEXT EVENT: {data_type}")
                    if hasattr(event.data, "delta"):
                        print(f"    Delta: {event.data.delta}")

            elif event.type == "run_item_stream_event":
                print(f"    Item Type: {event.item.type}")

            print()  # Add spacing

            # Limit output for debugging
            if event_count > 20:
                print("... (truncated after 20 events)")
                break

        print(f"\n📊 Total events: {event_count}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_events())
