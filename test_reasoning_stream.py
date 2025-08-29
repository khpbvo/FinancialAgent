#!/usr/bin/env python3
"""
Test script to verify GPT-5 reasoning token streaming.
"""

import asyncio
from agents import Agent, Runner, ModelSettings
from openai.types.shared import Reasoning
from openai.types.responses import (
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseTextDeltaEvent
)


async def test_reasoning_stream():
    """Test that reasoning tokens are properly streamed."""
    
    print("🧪 Testing GPT-5 Reasoning Stream")
    print("=" * 60)
    
    # Create a simple agent with GPT-5 settings using proper Agents SDK format
    agent = Agent(
        name="ReasoningTester",
        instructions="You are a financial advisor. Think step-by-step through problems.",
        model="gpt-5",
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="high"),     # minimal | low | medium | high
            verbosity="high"                        # low | medium | high
        )
    )
    
    # Complex query that should trigger reasoning
    test_query = """
    I have $50,000 in savings, a $200,000 mortgage at 5.5% with 20 years left,
    and I can save $2,000/month. Should I:
    1. Pay down the mortgage faster
    2. Invest in index funds (expecting 8% returns)
    3. Split between both
    
    Please analyze the math and tax implications.
    """
    
    print(f"📝 Query: {test_query[:100]}...")
    print("-" * 60)
    
    # Stream the response
    result = Runner.run_streamed(agent, input=test_query)
    
    reasoning_found = False
    response_found = False
    reasoning_text = ""
    response_text = ""
    
    print("\nStreaming events:\n")
    
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            # Check for reasoning events
            if isinstance(event.data, ResponseReasoningTextDeltaEvent):
                if not reasoning_found:
                    print("✅ FOUND: ResponseReasoningTextDeltaEvent")
                    print("🧠 Reasoning tokens streaming:\n" + "─" * 40)
                    reasoning_found = True
                print(event.data.delta, end="", flush=True)
                reasoning_text += event.data.delta
            
            elif isinstance(event.data, ResponseReasoningTextDoneEvent):
                if reasoning_found:
                    print("\n" + "─" * 40)
                    print("✅ Reasoning complete")
            
            # Check for response text
            elif isinstance(event.data, ResponseTextDeltaEvent):
                if not response_found:
                    print("\n\n✅ FOUND: ResponseTextDeltaEvent")
                    print("📝 Response tokens streaming:\n" + "=" * 40)
                    response_found = True
                print(event.data.delta, end="", flush=True)
                response_text += event.data.delta
    
    print("\n\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"  - Reasoning events detected: {'✅ Yes' if reasoning_found else '❌ No'}")
    print(f"  - Response events detected: {'✅ Yes' if response_found else '❌ No'}")
    
    if reasoning_found:
        print(f"  - Reasoning length: {len(reasoning_text)} characters")
    if response_found:
        print(f"  - Response length: {len(response_text)} characters")
    
    print("\n" + "=" * 60)
    
    if reasoning_found:
        print("✅ SUCCESS: Reasoning tokens are streaming correctly!")
    else:
        print("⚠️  WARNING: No reasoning tokens detected.")
        print("    This might mean:")
        print("    1. GPT-5 didn't use reasoning for this query")
        print("    2. The model settings aren't triggering reasoning")
        print("    3. The API doesn't support reasoning streaming yet")
    
    return reasoning_found, response_found


async def test_simple_query():
    """Test with a simple query that might not trigger reasoning."""
    
    print("\n\n🧪 Testing Simple Query (may not trigger reasoning)")
    print("=" * 60)
    
    agent = Agent(
        name="SimpleTester",
        instructions="Answer concisely.",
        model="gpt-5",
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="low"),      # minimal | low | medium | high
            verbosity="low"                         # low | medium | high
        )
    )
    
    result = Runner.run_streamed(agent, input="What is 2+2?")
    
    reasoning_found = False
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseReasoningTextDeltaEvent):
                reasoning_found = True
                print("🧠 Reasoning: ", event.data.delta, end="")
            elif isinstance(event.data, ResponseTextDeltaEvent):
                print("📝 Response: ", event.data.delta, end="")
    
    print(f"\n\nReasoning for simple query: {'Yes' if reasoning_found else 'No (expected)'}")


if __name__ == "__main__":
    import os
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set: export OPENAI_API_KEY=sk-...")
        exit(1)
    
    print("🚀 Starting GPT-5 Reasoning Stream Tests\n")
    
    # Run main test
    asyncio.run(test_reasoning_stream())
    
    # Run simple test
    asyncio.run(test_simple_query())
    
    print("\n✨ Tests complete!")
    print("\nNote: If reasoning tokens are not appearing, ensure:")
    print("1. You're using GPT-5 model")
    print("2. The reasoning.effort is set to 'high' or 'medium'")
    print("3. The query is complex enough to trigger reasoning")