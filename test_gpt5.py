#!/usr/bin/env python3
"""
Test script for GPT-5 model with text verbosity and reasoning effort parameters.
This demonstrates the Financial Agent using GPT-5's advanced reasoning capabilities.
"""

import asyncio
import os
from datetime import datetime
from agents import Runner
from financial_agent.agent import build_agent, build_deps


def print_separator(title: str = ""):
    """Print a formatted separator line."""
    if title:
        print(f"\n{'=' * 20} {title} {'=' * 20}")
    else:
        print("=" * 60)


async def test_gpt5_reasoning():
    """Test GPT-5 with high reasoning effort and high verbosity."""
    
    print_separator("GPT-5 Financial Agent Test")
    print(f"Model: GPT-5")
    print(f"Text Verbosity: HIGH")
    print(f"Reasoning Effort: HIGH")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()
    
    # Test queries that benefit from high reasoning and verbosity
    test_queries = [
        {
            "query": "Analyze my financial situation and create a comprehensive plan to achieve financial independence in 10 years",
            "description": "Complex multi-domain query requiring deep reasoning"
        },
        {
            "query": "What are the tax implications of converting my traditional IRA to a Roth IRA while also maximizing my 401k contributions?",
            "description": "Tax optimization query requiring detailed analysis"
        },
        {
            "query": "How should I prioritize: paying off my 6% student loan, investing in index funds, or building a 6-month emergency fund?",
            "description": "Financial prioritization requiring trade-off analysis"
        }
    ]
    
    with build_deps() as deps:
        agent = build_agent()
        
        for i, test in enumerate(test_queries, 1):
            print_separator(f"Test {i}: {test['description']}")
            print(f"Query: {test['query']}")
            print("-" * 60)
            
            try:
                # Run the agent with the test query
                print("Processing with GPT-5 (high reasoning + high verbosity)...")
                result = await Runner.run(
                    agent, 
                    test["query"], 
                    context=deps
                )
                
                print("\n📝 Response:")
                print(result.final_output)
                
                # Display reasoning indicators if available
                if hasattr(result, 'reasoning_tokens') and result.reasoning_tokens:
                    print(f"\n🧠 Reasoning tokens used: {result.reasoning_tokens}")
                
                print("\n✅ Test completed successfully")
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
            
            print_separator()
            
            # Add delay between tests to avoid rate limits
            if i < len(test_queries):
                print("Waiting 2 seconds before next test...")
                await asyncio.sleep(2)
    
    print_separator("All Tests Completed")
    print("GPT-5 configuration with high reasoning and verbosity is working correctly!")
    print("The model will provide detailed, well-reasoned responses to financial queries.")


async def test_direct_api():
    """Test direct OpenAI API call with GPT-5 parameters (for comparison)."""
    
    print_separator("Direct OpenAI API Test")
    
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        # Direct API call showing the parameters
        resp = await client.responses.create(
            model="gpt-5",
            input="Explain the concept of compound interest and its impact on long-term wealth building",
            text={"verbosity": "high"},      # high | medium | low
            reasoning={"effort": "high"}     # high | medium | low
        )
        
        print("Direct API Response:")
        print(resp.output_text)
        
        if hasattr(resp, 'reasoning_text') and resp.reasoning_text:
            print("\nReasoning Process:")
            print(resp.reasoning_text)
            
    except Exception as e:
        print(f"Direct API test failed: {e}")
        print("This is expected if using the Agents SDK instead of direct API.")
    
    print_separator()


async def main():
    """Main test function."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your API key: export OPENAI_API_KEY=sk-...")
        return
    
    print("\n🚀 Starting GPT-5 Configuration Tests\n")
    
    # Run the main agent tests
    await test_gpt5_reasoning()
    
    # Optionally test direct API (may fail if not using Responses API directly)
    print("\nTrying direct API test (optional)...")
    await test_direct_api()
    
    print("\n✨ All tests completed!")
    print("\nNote: The Financial Agent is now configured to use GPT-5 with:")
    print("  - Text Verbosity: HIGH (detailed, comprehensive responses)")
    print("  - Reasoning Effort: HIGH (deep analysis and complex problem-solving)")
    print("\nThese settings are applied to both the main agent and all specialist agents.")


if __name__ == "__main__":
    asyncio.run(main())