from __future__ import annotations
import argparse
import asyncio
import os
import sys
from typing import Optional

from agents import Runner, ItemHelpers
from openai.types.responses import ResponseTextDeltaEvent

from .agent import build_agent, build_deps


async def interactive_mode(agent, deps) -> None:
    """Interactive mode with streaming support"""
    print("🏦 Financial Agent Interactive Mode")
    print("Type 'quit' or 'exit' to leave, 'help' for commands")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\n💭 What would you like to know? > ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("""
Available commands:
• Ask questions about your finances
• 'bootstrap' - Ingest documents from documents/ folder
• 'recent' - Show recent transactions
• 'analyze' - Analyze your spending patterns
• 'help' - Show this help
• 'quit' - Exit the program
""")
                continue
            
            if user_input.lower() == 'bootstrap':
                from .bootstrap import bootstrap_documents
                print("📁 Ingesting documents...")
                result = bootstrap_documents()
                print(result)
                continue
            
            if user_input.lower() == 'recent':
                user_input = "Show me my 10 most recent transactions"
            elif user_input.lower() == 'analyze':
                user_input = "Analyze my recent spending patterns and give me insights"
            
            if not user_input:
                continue
            
            # Stream the response
            print("\n🤖 Thinking...")
            result = Runner.run_streamed(agent, input=user_input, context=deps)
            
            current_message = ""
            show_progress = True
            
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    # Stream text deltas for real-time output
                    if isinstance(event.data, ResponseTextDeltaEvent):
                        if show_progress:
                            print("\n📝 Response:", end=" ", flush=True)
                            show_progress = False
                        print(event.data.delta, end="", flush=True)
                        current_message += event.data.delta
                
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        tool_name = getattr(event.item, 'name', getattr(event.item, 'function', {}).get('name', 'Unknown Tool'))
                        print(f"\n🔧 Using tool: {tool_name}")
                    elif event.item.type == "tool_call_output_item":
                        print(f"✅ Tool completed")
                    elif event.item.type == "message_output_item":
                        # If we didn't get deltas, show the full message
                        if not current_message:
                            try:
                                message_text = ItemHelpers.text_message_output(event.item)
                                print(f"\n📝 Response: {message_text}")
                            except Exception:
                                print(f"\n📝 Response: {event.item}")
                
                elif event.type == "agent_updated_stream_event":
                    print(f"\n🔄 Agent updated: {event.new_agent.name}")
            
            print("\n")  # Add newline after response
            current_message = ""  # Reset for next iteration
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted! Type 'quit' to exit properly.")
            continue
        except EOFError:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


async def streaming_mode(agent, deps, user_input: str) -> None:
    """Single command with streaming output"""
    print("=== Financial Agent Streaming Mode ===")
    result = Runner.run_streamed(agent, input=user_input, context=deps)
    
    print(f"🤖 Processing: {user_input}")
    print("-" * 50)
    
    current_message = ""
    show_progress = True
    
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            # Stream text deltas for real-time output
            if isinstance(event.data, ResponseTextDeltaEvent):
                if show_progress:
                    print("📝 Response: ", end="", flush=True)
                    show_progress = False
                print(event.data.delta, end="", flush=True)
                current_message += event.data.delta
        
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                tool_name = getattr(event.item, 'name', getattr(event.item, 'function', {}).get('name', 'Unknown Tool'))
                print(f"\n🔧 Using tool: {tool_name}")
            elif event.item.type == "tool_call_output_item":
                print(f"✅ Tool completed")
            elif event.item.type == "message_output_item":
                # If we didn't get deltas, show the full message
                if not current_message:
                    try:
                        message_text = ItemHelpers.text_message_output(event.item)
                        print(f"📝 Response: {message_text}")
                    except Exception:
                        print(f"📝 Response: {event.item}")
        
        elif event.type == "agent_updated_stream_event":
            print(f"\n🔄 Agent updated: {event.new_agent.name}")
    
    print("\n" + "=" * 50)
    print("✅ Complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial Agent CLI")
    parser.add_argument("-i", "--input", help="Prompt or command for the agent", default=None)
    parser.add_argument("--stream", action="store_true", help="Stream output events")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--bootstrap", action="store_true", help="Ingest PDFs/CSVs from documents/")
    args = parser.parse_args()

    deps = build_deps()
    agent = build_agent()

    if args.bootstrap:
        # quick ingestion pass
        from .bootstrap import bootstrap_documents
        print("📁 Bootstrapping documents...")
        result = bootstrap_documents()
        print(result)
        return

    # Interactive mode - default if no specific input
    if args.interactive or (not args.input and not args.stream):
        asyncio.run(interactive_mode(agent, deps))
    elif args.stream:
        # Streaming mode for single command
        user_input = args.input or "Analyze my recent spending."
        asyncio.run(streaming_mode(agent, deps, user_input))
    else:
        # Non-streaming mode for single command
        result = Runner.run_sync(agent, args.input, context=deps)
        print(result.final_output)

if __name__ == "__main__":
    main()
