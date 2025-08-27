from __future__ import annotations
import argparse
import asyncio
import os
import sys
from typing import Optional

from agents import Runner, ItemHelpers, SQLiteSession
from openai.types.responses import ResponseTextDeltaEvent
from pathlib import Path

from .agent import build_agent, build_deps

# Centralized descriptions for all tools exposed by the agent. Keeping this
# mapping in a single place avoids duplication between interactive and
# streaming modes and makes maintenance easier when new tools are added.
TOOL_DESCRIPTIONS = {
    # Core tools
    "ingest_csv": "📊 Processing CSV file",
    "ingest_pdfs": "📄 Extracting PDF content",
    "list_recent_transactions": "📋 Fetching recent transactions",
    "search_transactions": "🔍 Searching transaction history",
    "analyze_and_advise": "💡 Analyzing financial data",
    "summarize_file": "📝 Summarizing document",
    "summarize_overview": "📈 Creating overview summary",
    "add_transaction": "➕ Adding transaction",
    "list_memories": "🧠 Retrieving memories",
    # Export tools
    "export_transactions": "📤 Exporting transactions",
    "export_recurring_payments": "🔄 Exporting recurring payments only",
    "generate_tax_report": "🏛️ Generating tax report",
    "export_budget_report": "📊 Exporting budget report",
    # Budget tools
    "set_budget": "💰 Setting budget",
    "check_budget": "💳 Checking budget status",
    "list_budgets": "📋 Listing budgets",
    "suggest_budgets": "💡 Suggesting budget plans",
    "delete_budget": "🗑️ Deleting budget",
    # Goal tools
    "create_goal": "🎯 Creating financial goal",
    "update_goal_progress": "📈 Updating goal progress",
    "check_goals": "🎯 Checking goals status",
    "suggest_savings_plan": "💰 Suggesting savings plan",
    "complete_goal": "✅ Completing goal",
    "pause_goal": "⏸️ Pausing goal",
    # Recurring transaction tools
    "detect_recurring": "🔄 Detecting recurring payments",
    "list_subscriptions": "📋 Listing subscriptions",
    "analyze_subscription_value": "💡 Analyzing subscription value",
    "predict_next_recurring": "🔮 Predicting next payments",
    # Handoff tools
    "handoff_to_tax_specialist": "🏛️ Consulting tax specialist",
    "handoff_to_budget_specialist": "💰 Consulting budget specialist",
    "handoff_to_goal_specialist": "🎯 Consulting goal specialist",
    "coordinate_multi_specialist_analysis": "🤝 Multi-specialist analysis",
    "route_user_query": "🧠 Analyzing query routing",
}


async def interactive_mode(agent, deps, use_session: bool = True) -> None:
    """Interactive mode with streaming support and session memory"""
    print("🏦 Financial Agent Interactive Mode")
    print("Type 'quit' or 'exit' to leave, 'help' for commands")
    print("-" * 50)
    
    # Initialize session for conversation persistence
    session = None
    if use_session:
        session_db_path = Path.home() / ".financial_agent" / "sessions.db"
        session_db_path.parent.mkdir(parents=True, exist_ok=True)
        session = SQLiteSession("default_session", str(session_db_path))
        print("💾 Session memory enabled - conversations will be remembered")
        print("   Use 'clear' to start a fresh conversation")
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
• 'clear' - Clear conversation history (start fresh)
• 'clear-db' - Clear all transactions and memories from database
• 'clear-sessions' - Clear all conversation session files
• 'clear-all' - Clear both database and session files
• 'history' - Show conversation history status
• 'help' - Show this help
• 'quit' - Exit the program
""")
                continue
            
            if user_input.lower() == 'clear' and session:
                await session.clear_session()
                print("🧹 Conversation history cleared")
                continue
            
            if user_input.lower() == 'clear-db':
                result = clear_database()
                print(result)
                continue
                
            if user_input.lower() == 'clear-sessions':
                result = clear_sessions()
                print(result)
                # Also clear current session if active
                if session:
                    await session.clear_session()
                    print("🧹 Current conversation history also cleared")
                continue
            
            if user_input.lower() == 'clear-all':
                result = clear_all()
                print(result)
                # Also clear current session if active
                if session:
                    await session.clear_session()
                    print("🧹 Current conversation history also cleared")
                continue
            
            if user_input.lower() == 'history' and session:
                items = await session.get_items()
                print(f"📜 Conversation has {len(items)} messages in history")
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
            
            # Stream the response with session
            print("\n🤖 Thinking...")
            result = Runner.run_streamed(agent, input=user_input, context=deps, session=session)
            
            current_message = ""
            show_progress = True
            reasoning_started = False
            
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    # Handle reasoning/thinking events if available
                    if hasattr(event.data, 'type') and 'reasoning' in str(event.data.type).lower():
                        if not reasoning_started:
                            print("\n🧠 Agent reasoning: ", end="", flush=True)
                            reasoning_started = True
                        if hasattr(event.data, 'delta'):
                            print(event.data.delta, end="", flush=True)
                    # Handle text deltas
                    elif isinstance(event.data, ResponseTextDeltaEvent):
                        if show_progress:
                            print("\n📝 Response:", end=" ", flush=True)
                            show_progress = False
                        print(event.data.delta, end="", flush=True)
                        current_message += event.data.delta
                
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        # Fixed tool name extraction - get from raw_item
                        tool_name = "Unknown Tool"  # Default fallback
                        
                        # Method 1: Try raw_item.function.name (OpenAI API structure)
                        if hasattr(event.item, 'raw_item') and event.item.raw_item:
                            raw = event.item.raw_item
                            if hasattr(raw, 'function') and hasattr(raw.function, 'name'):
                                tool_name = raw.function.name
                            elif hasattr(raw, 'name'):
                                tool_name = raw.name
                            # Handle dictionary format
                            elif isinstance(raw, dict):
                                if 'function' in raw and isinstance(raw['function'], dict):
                                    tool_name = raw['function'].get('name', 'Unknown Tool')
                                elif 'name' in raw:
                                    tool_name = raw['name']
                        
                        # Method 2: Try direct item attributes (fallback)
                        if tool_name == "Unknown Tool":
                            if hasattr(event.item, 'name') and event.item.name:
                                tool_name = event.item.name
                            elif hasattr(event.item, 'function') and hasattr(event.item.function, 'name'):
                                tool_name = event.item.function.name
                            elif hasattr(event.item, 'tool_name') and event.item.tool_name:
                                tool_name = event.item.tool_name
                        
                        # Use global tool descriptions
                        desc = TOOL_DESCRIPTIONS.get(tool_name, f"🔧 Using tool: {tool_name}")
                        print(f"\n{desc}")
                        
                        # Add progress estimates for known long-running tools
                        if tool_name in ["detect_recurring", "analyze_and_advise", "ingest_csv", "ingest_pdfs", "analyze_subscription_value"]:
                            print("   ⏳ This may take a moment...")
                    elif event.item.type == "tool_call_output_item":
                        # Show completion with more detailed feedback
                        output = getattr(event.item, 'output', '')
                        if output and len(str(output)) < 100:
                            print(f"   ✅ Completed: {str(output)[:100]}")
                        else:
                            print(f"   ✅ Tool completed successfully")
                    elif event.item.type == "message_output_item":
                        # If we didn't get deltas, show the full message
                        if not current_message:
                            try:
                                message_text = ItemHelpers.text_message_output(event.item)
                                print(f"\n📝 Response: {message_text}")
                            except Exception:
                                print(f"\n📝 Response: {event.item}")
                    
                    # Handle reasoning items if they exist
                    elif event.item.type == "reasoning_item" and hasattr(event.item, 'content'):
                        reasoning_content = str(event.item.content)[:100]
                        print(f"\n🧠 Agent reasoning: {reasoning_content}...")
                
                elif event.type == "agent_updated_stream_event":
                    print(f"\n🔄 Agent updated: {event.new_agent.name}")
                    print(f"   💼 Specializing in: {event.new_agent.name.lower().replace('_', ' ')}")
            
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


async def streaming_mode(agent, deps, user_input: str, use_session: bool = False) -> None:
    """Single command with streaming output"""
    print("=== Financial Agent Streaming Mode ===")
    
    session = None
    if use_session:
        session_db_path = Path.home() / ".financial_agent" / "sessions.db"
        session_db_path.parent.mkdir(parents=True, exist_ok=True)
        session = SQLiteSession("streaming_session", str(session_db_path))
    
    result = Runner.run_streamed(agent, input=user_input, context=deps, session=session)
    
    print(f"🤖 Processing: {user_input}")
    print("-" * 50)
    
    current_message = ""
    show_progress = True
    reasoning_started = False
    
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            # Handle reasoning/thinking events if available
            if hasattr(event.data, 'type') and 'reasoning' in str(event.data.type).lower():
                if not reasoning_started:
                    print("\n🧠 Agent reasoning: ", end="", flush=True)
                    reasoning_started = True
                if hasattr(event.data, 'delta'):
                    print(event.data.delta, end="", flush=True)
            # Handle text deltas
            elif isinstance(event.data, ResponseTextDeltaEvent):
                if show_progress:
                    print("📝 Response: ", end="", flush=True)
                    show_progress = False
                print(event.data.delta, end="", flush=True)
                current_message += event.data.delta
        
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                # Fixed tool name extraction - get from raw_item
                tool_name = "Unknown Tool"  # Default fallback
                
                # Method 1: Try raw_item.function.name (OpenAI API structure)
                if hasattr(event.item, 'raw_item') and event.item.raw_item:
                    raw = event.item.raw_item
                    if hasattr(raw, 'function') and hasattr(raw.function, 'name'):
                        tool_name = raw.function.name
                    elif hasattr(raw, 'name'):
                        tool_name = raw.name
                    # Handle dictionary format
                    elif isinstance(raw, dict):
                        if 'function' in raw and isinstance(raw['function'], dict):
                            tool_name = raw['function'].get('name', 'Unknown Tool')
                        elif 'name' in raw:
                            tool_name = raw['name']
                
                # Method 2: Try direct item attributes (fallback)
                if tool_name == "Unknown Tool":
                    if hasattr(event.item, 'name') and event.item.name:
                        tool_name = event.item.name
                    elif hasattr(event.item, 'function') and hasattr(event.item.function, 'name'):
                        tool_name = event.item.function.name
                    elif hasattr(event.item, 'tool_name') and event.item.tool_name:
                        tool_name = event.item.tool_name
                
                # Use global tool descriptions
                desc = TOOL_DESCRIPTIONS.get(tool_name, f"🔧 Using tool: {tool_name}")
                print(f"\n{desc}")
                
                # Add progress estimates for known long-running tools
                if tool_name in ["detect_recurring", "analyze_and_advise", "ingest_csv", "ingest_pdfs", "analyze_subscription_value"]:
                    print("   ⏳ This may take a moment...")
            elif event.item.type == "tool_call_output_item":
                # Show completion with more detailed feedback
                output = getattr(event.item, 'output', '')
                if output and len(str(output)) < 100:
                    print(f"   ✅ Completed: {str(output)[:100]}")
                else:
                    print(f"   ✅ Tool completed successfully")
            elif event.item.type == "message_output_item":
                # If we didn't get deltas, show the full message
                if not current_message:
                    try:
                        message_text = ItemHelpers.text_message_output(event.item)
                        print(f"📝 Response: {message_text}")
                    except Exception:
                        print(f"📝 Response: {event.item}")
            
            # Handle reasoning items if they exist
            elif event.item.type == "reasoning_item" and hasattr(event.item, 'content'):
                reasoning_content = str(event.item.content)[:100]
                print(f"\n🧠 Agent reasoning: {reasoning_content}...")
        
        elif event.type == "agent_updated_stream_event":
            print(f"\n🔄 Agent updated: {event.new_agent.name}")
            print(f"   💼 Specializing in: {event.new_agent.name.lower().replace('_', ' ')}")
    
    print("\n" + "=" * 50)
    print("✅ Processing complete! Results shown above.")


def clear_database() -> str:
    """Clear all data from the financial database."""
    deps = build_deps()
    deps.ensure_ready()
    cursor = deps.db.conn.cursor()
    
    # Get counts before clearing
    cursor.execute('SELECT COUNT(*) FROM transactions')
    transaction_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM memories')
    memory_count = cursor.fetchone()[0]
    
    # Clear all data
    cursor.execute('DELETE FROM transactions')
    cursor.execute('DELETE FROM memories')
    deps.db.conn.commit()
    
    return f"🧹 Database cleared: {transaction_count} transactions and {memory_count} memories removed"


def clear_sessions() -> str:
    """Clear all session files."""
    session_path = Path.home() / ".financial_agent"
    if not session_path.exists():
        return "No session files found"
    
    removed_files = []
    for file in session_path.glob("*"):
        if file.is_file():
            file.unlink()
            removed_files.append(file.name)
    
    if removed_files:
        return f"🧹 Session files cleared: {', '.join(removed_files)}"
    else:
        return "No session files to clear"


def clear_all() -> str:
    """Clear both database and session files."""
    db_result = clear_database()
    session_result = clear_sessions()
    return f"{db_result}\n{session_result}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial Agent CLI")
    parser.add_argument("-i", "--input", help="Prompt or command for the agent", default=None)
    parser.add_argument("--stream", action="store_true", help="Stream output events")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--bootstrap", action="store_true", help="Ingest PDFs/CSVs from documents/")
    parser.add_argument("--no-session", action="store_true", help="Disable session memory")
    parser.add_argument("--clear-db", action="store_true", help="Clear all transactions and memories from database")
    parser.add_argument("--clear-sessions", action="store_true", help="Clear all conversation session files")
    parser.add_argument("--clear-all", action="store_true", help="Clear both database and session files")
    args = parser.parse_args()

    # Handle clearing commands first (before context manager)
    if args.clear_db:
        result = clear_database()
        print(result)
        return
    
    if args.clear_sessions:
        result = clear_sessions()
        print(result)
        return
    
    if args.clear_all:
        result = clear_all()
        print(result)
        return

    with build_deps() as deps:
        agent = build_agent()
        if args.bootstrap:
            # quick ingestion pass
            from .bootstrap import bootstrap_documents
            print("📁 Bootstrapping documents...")
            result = bootstrap_documents()
            print(result)
            return

        # Interactive mode - default if no specific input
        use_session = not args.no_session

        if args.interactive or (not args.input and not args.stream):
            asyncio.run(interactive_mode(agent, deps, use_session=use_session))
        elif args.stream:
            # Streaming mode for single command
            user_input = args.input or "Analyze my recent spending."
            asyncio.run(streaming_mode(agent, deps, user_input, use_session=use_session))
        else:
            # Non-streaming mode for single command
            if use_session:
                session_db_path = Path.home() / ".financial_agent" / "sessions.db"
                session_db_path.parent.mkdir(parents=True, exist_ok=True)
                session = SQLiteSession("cli_session", str(session_db_path))
                result = Runner.run_sync(agent, args.input, context=deps, session=session)
            else:
                result = Runner.run_sync(agent, args.input, context=deps)
            print(result.final_output)

if __name__ == "__main__":
    main()