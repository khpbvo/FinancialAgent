from __future__ import annotations
import argparse
import asyncio

from agents import Runner, ItemHelpers, SQLiteSession, trace
import sqlite3
from typing import Any
from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseOutputItemAddedEvent,
)
from pathlib import Path

from .agent import build_agent, build_deps
from .tools.export import export_recurring_payments


# Ensure SQLite session DB has required tables
def _ensure_session_db_schema(db_path: Path) -> None:
    """Ensure the session DB is compatible with Agents SDK SQLiteSession.

    We avoid creating our own schema. If a legacy schema exists without
    the expected columns (e.g., missing 'message_data'), we add them.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        # If agent_messages exists but lacks message_data, add it
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_messages'"
        )
        if cur.fetchone():
            cur.execute("PRAGMA table_info(agent_messages)")
            cols = [row[1] for row in cur.fetchall()]
            if "message_data" not in cols:
                try:
                    cur.execute(
                        "ALTER TABLE agent_messages ADD COLUMN message_data TEXT"
                    )
                except Exception:
                    pass
                # Simple index on session_id if missing
                try:
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_agent_messages_session ON agent_messages(session_id)"
                    )
                except Exception:
                    pass
        # If agent_sessions exists, ensure updated_at exists (best effort)
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_sessions'"
        )
        if cur.fetchone():
            cur.execute("PRAGMA table_info(agent_sessions)")
            cols = [row[1] for row in cur.fetchall()]
            if "updated_at" not in cols:
                try:
                    cur.execute(
                        "ALTER TABLE agent_sessions ADD COLUMN updated_at TEXT DEFAULT (datetime('now'))"
                    )
                except Exception:
                    pass
        conn.commit()
    except Exception:
        # Silent best-effort migration; SQLiteSession will create if needed
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


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
        session_dir = Path.home() / ".financial_agent"
        session_dir.mkdir(parents=True, exist_ok=True)
        session_db_path = session_dir / "sessions.db"
        # Do not mutate schema up front; let SDK manage it. We'll self-heal on errors.
        session = SQLiteSession("default_session", str(session_db_path))
        # Quick health check
        try:
            # Ensure DB/tables are usable
            await session.get_items()
        except Exception:
            # Backup and re-create with a clean DB name
            try:
                backup = (
                    session_dir
                    / f"sessions_corrupt_{int(asyncio.get_event_loop().time()*1000)}.db"
                )
                if session_db_path.exists():
                    session_db_path.replace(backup)
            except Exception:
                pass
            session_db_path = session_dir / "sessions_v2.db"
            session = SQLiteSession("default_session", str(session_db_path))
        print("💾 Session memory enabled - conversations will be remembered")
        print("   Use 'clear' to start a fresh conversation")
        print("-" * 50)

    while True:
        try:
            # Get user input
            user_input = input("\n💭 What would you like to know? > ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if user_input.lower() == "help":
                print(
                    """
Available commands:
• Ask questions about your finances
• 'bootstrap' - Ingest documents from documents/ folder
• 'recent' - Show recent transactions
• 'analyze' - Analyze your spending patterns
• 'bills' - Export bills-only recurring payments (CSV)
• 'bills-pdf' - Export bills-only recurring payments (PDF)
• 'bills-excel' - Export bills-only recurring payments (Excel)
• 'clear' - Clear conversation history (start fresh)
• 'clear-db' - Clear all transactions and memories from database
• 'clear-sessions' - Clear all conversation session files
• 'clear-all' - Clear both database and session files
• 'history' - Show conversation history status
• 'help' - Show this help
• 'quit' - Exit the program
"""
                )
                continue

            if user_input.lower() == "clear" and session:
                await session.clear_session()
                print("🧹 Conversation history cleared")
                continue

            if user_input.lower() == "clear-db":
                result = clear_database()
                print(result)
                continue

            if user_input.lower() == "clear-sessions":
                result = clear_sessions()
                print(result)
                # Also clear current session if active
                if session:
                    try:
                        await session.clear_session()
                        print("🧹 Current conversation history also cleared")
                    except Exception:
                        # Session table might not exist after clearing
                        pass
                continue

            if user_input.lower() == "clear-all":
                result = clear_all()
                print(result)
                # Also clear current session if active
                if session:
                    try:
                        await session.clear_session()
                        print("🧹 Current conversation history also cleared")
                    except Exception:
                        # Session table might not exist after clearing
                        pass
                continue

            if user_input.lower() == "history" and session:
                items = await session.get_items()
                print(f"📜 Conversation has {len(items)} messages in history")
                continue

            if user_input.lower() == "bootstrap":
                from .bootstrap import bootstrap_documents

                print("📁 Ingesting documents...")
                result = bootstrap_documents()
                print(result)
                continue

            if user_input.lower() == "recent":
                user_input = "Show me my 10 most recent transactions"
            elif user_input.lower() == "analyze":
                user_input = "Analyze my recent spending patterns and give me insights"
            elif user_input.lower() in [
                "bills",
                "bills-csv",
                "bills-excel",
                "bills-pdf",
            ]:
                fmt = "csv"
                if user_input.lower() == "bills-excel":
                    fmt = "excel"
                elif user_input.lower() == "bills-pdf":
                    fmt = "pdf"
                try:
                    payload = {
                        "format": fmt,
                        "exclude_credit_repayment": True,
                        "bills_only": True,
                        "min_confidence": 0.7,
                    }
                    from agents import RunContextWrapper, ItemHelpers

                    result = await export_recurring_payments.on_invoke_tool(RunContextWrapper(deps), ItemHelpers.json_dumps(payload))  # type: ignore
                    print(result)
                except Exception as e:
                    print(f"❌ Failed to export bills-only recurring: {e}")
                continue

            if not user_input:
                continue

            # Fast short-circuit for monthly cost queries to avoid LLM latency
            mc_phrases = [
                "monthly cost",
                "monthly costs",
                "monthly spending",
                "spending last month",
                "last month spend",
                "expenses last month",
                "monthly expenses",
                "costs last month",
            ]
            if any(p in user_input.lower() for p in mc_phrases):
                try:
                    from .tools.costs import monthly_cost_summary
                    from agents import RunContextWrapper, ItemHelpers

                    print("\n⚡ Fast path: computing monthly costs...")
                    payload = {
                        "last_full_month": True,
                        "bills_only": False,
                        "include_breakdown": True,
                    }
                    fast_res = await monthly_cost_summary.on_invoke_tool(RunContextWrapper(deps), ItemHelpers.json_dumps(payload))  # type: ignore
                    print(fast_res)
                    continue
                except Exception as e:
                    print(f"❌ Fast path failed, falling back to agent: {e}")

            # Stream the response with session
            print("\n🤖 Processing your request...")
            try:
                result = Runner.run_streamed(
                    agent, input=user_input, context=deps, session=session
                )
            except Exception as e:
                print(f"\n❌ Failed to start streaming: {e}")
                print("🔄 Attempting non-streaming fallback...")
                try:
                    sync_result = await Runner.run(
                        agent, input=user_input, context=deps, session=session
                    )
                    print(f"\n📝 Response (fallback): {sync_result.final_output}")
                except Exception as sync_error:
                    print(f"❌ Both streaming and sync failed: {sync_error}")
                continue

            current_message = ""
            show_progress = True
            reasoning_started = False
            # (reserved for future reasoning summary output)
            displayed_messages = set()  # Track which messages we've already shown

            # Watchdog to avoid perceived hangs: print keepalive and fallback after timeout
            timeout_seconds = 30.0

            try:
                # Stream with watchdog + reasoning support to avoid perceived hangs
                stream_iter = result.stream_events().__aiter__()
                last_event_time = asyncio.get_event_loop().time()
                timeout_seconds = 60.0  # hard timeout to avoid indefinite hang
                keepalive_interval = 10.0
                last_keepalive = last_event_time

                any_output_streamed = False
                while True:
                    # Emit keepalive if we're still waiting
                    now = asyncio.get_event_loop().time()
                    if (
                        now - last_event_time > keepalive_interval
                        and now - last_keepalive >= keepalive_interval
                    ):
                        print("\n⏳ Still working... waiting on model/stream...")
                        last_keepalive = now

                    try:
                        event = await asyncio.wait_for(
                            stream_iter.__anext__(), timeout=5.0
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        # If we've exceeded total timeout without any new events, fallback
                        if (
                            asyncio.get_event_loop().time() - last_event_time
                        ) > timeout_seconds:
                            print(
                                "\n⏱️ No stream events for 60s. Falling back to non-streaming..."
                            )
                            try:
                                # Fallback to a one-shot run to complete the response
                                sync_result = await Runner.run(
                                    agent,
                                    input=user_input,
                                    context=deps,
                                    session=session,
                                )
                                print(
                                    f"\n📝 Response (fallback): {sync_result.final_output}"
                                )
                            except Exception as fallback_error:
                                # Session schema might be incompatible. Retry with fresh session and then no-session.
                                err_text = str(fallback_error)
                                print(f"❌ Fallback failed: {err_text}")
                                if any(
                                    t in err_text
                                    for t in [
                                        "message_data",
                                        "agent_messages.role",
                                        "no such column",
                                    ]
                                ):
                                    try:
                                        new_path = (
                                            Path.home()
                                            / ".financial_agent"
                                            / "sessions_v2.db"
                                        )
                                        session = SQLiteSession(
                                            "default_session", str(new_path)
                                        )
                                        sync_result = await Runner.run(
                                            agent,
                                            input=user_input,
                                            context=deps,
                                            session=session,
                                        )
                                        print(
                                            f"\n📝 Response (fresh session): {sync_result.final_output}"
                                        )
                                    except Exception as second_error:
                                        print(
                                            f"❌ Fresh session failed: {second_error}. Retrying without session..."
                                        )
                                        nosess = await Runner.run(
                                            agent, input=user_input, context=deps
                                        )
                                        print(f"\n📝 Response: {nosess.final_output}")
                            break
                        # Otherwise continue waiting for the next event
                        continue

                    # We got an event; reset timers
                    last_event_time = asyncio.get_event_loop().time()

                    if event.type == "raw_response_event":
                        # DEBUG: Log event structure to understand what we're receiving
                        if hasattr(event, "data"):
                            data: Any = event.data
                            # Stream GPT-5 reasoning tokens for visibility
                            if isinstance(data, ResponseReasoningTextDeltaEvent):
                                any_output_streamed = True
                                if not reasoning_started:
                                    print("\n🧠 Reasoning:\n" + "─" * 50)
                                    reasoning_started = True
                                print(getattr(event.data, "delta", ""), end="", flush=True)
                                continue
                            if isinstance(data, ResponseReasoningSummaryTextDeltaEvent):
                                any_output_streamed = True
                                if not reasoning_started:
                                    print(
                                        "\n📊 Reasoning summary: ", end="", flush=True
                                    )
                                    reasoning_started = True
                                print(getattr(event.data, "delta", ""), end="", flush=True)
                                continue
                            # Handle text deltas from the LLM
                            if isinstance(data, ResponseTextDeltaEvent):
                                any_output_streamed = True
                                if show_progress:
                                    if reasoning_started:
                                        print(
                                            "\n📝 Final Response:\n" + "=" * 50,
                                            flush=True,
                                        )
                                    else:
                                        print("\n📝 Response: ", end="", flush=True)
                                    show_progress = False
                                print(getattr(event.data, "delta", ""), end="", flush=True)
                                continue
                            # Handle text done events (some models emit only 'done' without deltas)
                            if isinstance(data, ResponseTextDoneEvent):
                                any_output_streamed = True
                                if show_progress:
                                    if reasoning_started:
                                        print(
                                            "\n📝 Final Response:\n" + "=" * 50,
                                            flush=True,
                                        )
                                    else:
                                        print("\n📝 Response: ", end="", flush=True)
                                    show_progress = False
                                # ResponseTextDoneEvent has a .text property for the chunk
                                text = getattr(event.data, "text", "")
                                if text:
                                    print(text, end="", flush=True)
                                continue
                            # Some SDK paths emit only output-item-added with a text chunk
                            if isinstance(data, ResponseOutputItemAddedEvent):
                                # Best-effort extraction of text from the added item
                                added = data
                                text = None
                                # Try common attributes
                                text = getattr(added, "text", None) or getattr(
                                    getattr(added, "item", None), "text", None
                                )
                                if text:
                                    any_output_streamed = True
                                    if show_progress:
                                        print("\n📝 Response: ", end="", flush=True)
                                        show_progress = False
                                    print(text, end="", flush=True)
                                    continue

                            # Fallback: Try to extract text from raw event data
                            # Check for common text fields in event.data
                            if hasattr(data, "type"):
                                # Check for specific event types we may have missed
                                if "text" in str(getattr(data, "type", "")).lower():
                                    # Try to extract text content
                                    text_content = (
                                        getattr(data, "delta", None)
                                        or getattr(data, "text", None)
                                        or getattr(data, "content", None)
                                    )

                                    if text_content:
                                        any_output_streamed = True
                                        if show_progress:
                                            print("\n📝 Response: ", end="", flush=True)
                                            show_progress = False
                                        print(text_content, end="", flush=True)
                                        continue
                        # Ignore other raw events (created, in_progress, item_added, etc.) per Docs/streaming.md
                        # We'll rely on deltas and run item events for user-visible output.

                    elif event.type == "run_item_stream_event":
                        # Handle tool calls and completions
                        if event.item.type == "tool_call_item":
                            any_output_streamed = True
                            # Extract tool name for better feedback
                            tool_name = "Unknown Tool"
                            if hasattr(event.item, "raw_item") and event.item.raw_item:
                                raw = event.item.raw_item
                                if hasattr(raw, "function") and hasattr(
                                    raw.function, "name"
                                ):
                                    tool_name = raw.function.name
                                elif hasattr(raw, "name"):
                                    tool_name = raw.name
                                elif isinstance(raw, dict):
                                    if "function" in raw and isinstance(
                                        raw["function"], dict
                                    ):
                                        tool_name = raw["function"].get(
                                            "name", "Unknown Tool"
                                        )
                                    elif "name" in raw:
                                        tool_name = raw["name"]

                            # Fallback checks
                            if tool_name == "Unknown Tool":
                                if hasattr(event.item, "name") and event.item.name:
                                    tool_name = event.item.name
                                elif hasattr(event.item, "function") and hasattr(
                                    event.item.function, "name"
                                ):
                                    tool_name = event.item.function.name

                            desc = TOOL_DESCRIPTIONS.get(
                                tool_name, f"🔧 Using tool: {tool_name}"
                            )
                            print(f"\n{desc}")
                        elif event.item.type == "message_output_item":
                            any_output_streamed = True
                            # Only show the final message if we didn't already stream it via deltas
                            if not current_message:
                                message_text = None
                                try:
                                    from agents import ItemHelpers

                                    message_text = ItemHelpers.text_message_output(
                                        event.item
                                    )
                                except Exception:
                                    pass

                                # Fallback methods if ItemHelpers didn't work
                                if not message_text:
                                    # Try to extract text from item
                                    if hasattr(event.item, "text") and event.item.text:
                                        message_text = event.item.text
                                    elif (
                                        hasattr(event.item, "content")
                                        and event.item.content
                                    ):
                                        message_text = event.item.content
                                    elif (
                                        hasattr(event.item, "output")
                                        and event.item.output
                                    ):
                                        message_text = event.item.output
                                    elif hasattr(event.item, "raw_item"):
                                        # Try to get from raw_item
                                        raw = event.item.raw_item
                                        if isinstance(raw, dict):
                                            message_text = (
                                                raw.get("text")
                                                or raw.get("content")
                                                or raw.get("output")
                                            )
                                        elif hasattr(raw, "text") and raw.text:
                                            message_text = raw.text
                                        elif hasattr(raw, "content") and raw.content:
                                            message_text = raw.content

                                # Only display if we haven't shown this exact message before
                                if message_text:
                                    # Create a hash of the first 100 chars to identify duplicates
                                    message_hash = hash(
                                        message_text[:100]
                                        if len(message_text) > 100
                                        else message_text
                                    )
                                    if message_hash not in displayed_messages:
                                        print(f"\n📝 Response: {message_text}")
                                        displayed_messages.add(message_hash)
                                        current_message = message_text
                        elif event.item.type == "tool_call_output_item":
                            # Don't mark as output streamed just for tool completion
                            # We need actual response text, not just tool execution
                            # Signal tool completion when available
                            output = getattr(event.item, "output", "")
                            if output and len(str(output)) < 100:
                                print(f"   ✅ Completed: {str(output)[:100]}")
                            else:
                                print("   ✅ Tool completed successfully")

                    elif event.type == "agent_updated_stream_event":
                        if hasattr(event, "new_agent") and event.new_agent:
                            print(f"\n🔄 Agent updated: {event.new_agent.name}")
                        else:
                            print("\n🔄 Agent updated")

                # End of stream. If nothing was streamed or we didn't show a message, try to recover a final result.
                if not any_output_streamed or not current_message:
                    if not any_output_streamed:
                        print("\nℹ️ No tokens streamed. Checking for final result...")
                    else:
                        print(
                            "\nℹ️ No response text displayed. Checking for final result..."
                        )

                    # Try multiple approaches to get the final result
                    response_found = False

                    try:
                        # Method 1: Try to access final_output directly from the streaming result
                        # Wait a moment for the stream to complete
                        import time

                        time.sleep(0.5)

                        # Try to get the result synchronously
                        if hasattr(result, "__dict__"):
                            # Look for common result attributes
                            for attr_name in [
                                "_result",
                                "result",
                                "_final_result",
                                "final_output",
                            ]:
                                if hasattr(result, attr_name):
                                    attr_value = getattr(result, attr_name)
                                    if attr_value:
                                        if (
                                            hasattr(attr_value, "final_output")
                                            and attr_value.final_output
                                        ):
                                            print(
                                                f"📝 Response: {attr_value.final_output}"
                                            )
                                            response_found = True
                                            break
                                        elif (
                                            isinstance(attr_value, str)
                                            and attr_value.strip()
                                        ):
                                            print(f"📝 Response: {attr_value}")
                                            response_found = True
                                            break

                        # Method 2: Try to extract from new_items if available
                        if not response_found and hasattr(result, "new_items"):
                            for item in reversed(result.new_items):
                                if (
                                    hasattr(item, "type")
                                    and item.type == "message_output_item"
                                ):
                                    try:
                                        from agents import ItemHelpers

                                        message_text = ItemHelpers.text_message_output(
                                            item
                                        )
                                        if message_text:
                                            print(f"📝 Response: {message_text}")
                                            response_found = True
                                            break
                                    except Exception:
                                        # Try other extraction methods
                                        for field in ["text", "content", "output"]:
                                            if hasattr(item, field):
                                                text = getattr(item, field)
                                                if (
                                                    text
                                                    and isinstance(text, str)
                                                    and text.strip()
                                                ):
                                                    print(f"📝 Response: {text}")
                                                    response_found = True
                                                    break
                                        if response_found:
                                            break

                        # Method 3: Force a sync run to get the result
                        if not response_found:
                            print("ℹ️ Using fallback sync run...")
                            try:
                                final_res = await Runner.run(
                                    agent,
                                    input=user_input,
                                    context=deps,
                                    session=session,
                                )
                                if (
                                    hasattr(final_res, "final_output")
                                    and final_res.final_output
                                ):
                                    print(f"📝 Response: {final_res.final_output}")
                                    response_found = True
                            except Exception as sync_error:
                                print(f"❌ Sync fallback failed: {sync_error}")

                        if not response_found:
                            print("❌ Could not retrieve response text")

                    except Exception as e:
                        print(f"❌ Error retrieving response: {e}")
                        # Final fallback attempt
                        try:
                            final_res = await Runner.run(
                                agent, input=user_input, context=deps, session=session
                            )
                            print(f"📝 Response: {final_res.final_output}")
                        except Exception as final_error:
                            print(f"❌ All retrieval methods failed: {final_error}")

            except asyncio.TimeoutError:
                print("\n⏱️ Stream timeout after 2 minutes. Attempting fallback...")
                try:
                    sync = await Runner.run(
                        agent, input=user_input, context=deps, session=session
                    )
                    print(f"\n📝 Response (timeout fallback): {sync.final_output}")
                except Exception as fallback_error:
                    print(f"❌ Fallback also failed: {fallback_error}")

            except Exception as stream_error:
                print(f"\n❌ Streaming error: {stream_error}")
                print("🔄 Attempting sync fallback...")
                try:
                    sync = await Runner.run(
                        agent, input=user_input, context=deps, session=session
                    )
                    print(f"\n📝 Response (error fallback): {sync.final_output}")
                except Exception as fallback_error:
                    print(f"❌ All methods failed: {fallback_error}")

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


async def streaming_mode(
    agent, deps, user_input: str, use_session: bool = False
) -> None:
    """Single command with streaming output"""
    print("=== Financial Agent Streaming Mode ===")

    session = None
    if use_session:
        session_db_path = Path.home() / ".financial_agent" / "sessions.db"
        session_db_path.parent.mkdir(parents=True, exist_ok=True)
        _ensure_session_db_schema(session_db_path)
        session = SQLiteSession("streaming_session", str(session_db_path))

    result = Runner.run_streamed(agent, input=user_input, context=deps, session=session)

    print(f"🤖 Processing: {user_input}")
    print("-" * 50)

    current_message = ""
    show_progress = True
    reasoning_started = False
    reasoning_text = ""

    async for event in result.stream_events():
        if event.type == "raw_response_event":
            # Handle GPT-5 reasoning text streaming
            if isinstance(event.data, ResponseReasoningTextDeltaEvent):
                if not reasoning_started:
                    print(
                        "\n🧠 Agent reasoning (thinking step-by-step):\n",
                        end="",
                        flush=True,
                    )
                    print("─" * 50, flush=True)
                    reasoning_started = True
                delta = getattr(event.data, "delta", "")
                print(delta, end="", flush=True)
                reasoning_text += delta

            # Handle reasoning completion
            elif isinstance(event.data, ResponseReasoningTextDoneEvent):
                if reasoning_started:
                    print("\n" + "─" * 50)
                    print("✅ Reasoning complete\n", flush=True)

            # Handle reasoning summary (if available)
            elif isinstance(event.data, ResponseReasoningSummaryTextDeltaEvent):
                if not reasoning_started:
                    print("\n📊 Reasoning summary: ", end="", flush=True)
                print(getattr(event.data, "delta", ""), end="", flush=True)

            # Handle normal text response
            elif isinstance(event.data, ResponseTextDeltaEvent):
                if reasoning_started and show_progress:
                    # Add separator after reasoning
                    print("\n📝 Final Response:\n" + "=" * 50, flush=True)
                    show_progress = False
                elif show_progress:
                    print("📝 Response: ", end="", flush=True)
                    show_progress = False
                delta = getattr(event.data, "delta", "")
                print(delta, end="", flush=True)
                current_message += delta
            # Handle text done events (if deltas aren't emitted)
            elif isinstance(event.data, ResponseTextDoneEvent):
                if reasoning_started and show_progress:
                    print("\n📝 Final Response:\n" + "=" * 50, flush=True)
                    show_progress = False
                elif show_progress:
                    print("📝 Response: ", end="", flush=True)
                    show_progress = False
                text = getattr(event.data, "text", "")
                if text:
                    print(text, end="", flush=True)
                    current_message += text

        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                # Fixed tool name extraction - get from raw_item
                tool_name = "Unknown Tool"  # Default fallback

                # Method 1: Try raw_item.function.name (OpenAI API structure)
                if hasattr(event.item, "raw_item") and event.item.raw_item:
                    raw = event.item.raw_item
                    if hasattr(raw, "function") and hasattr(raw.function, "name"):
                        tool_name = raw.function.name
                    elif hasattr(raw, "name"):
                        tool_name = raw.name
                    # Handle dictionary format
                    elif isinstance(raw, dict):
                        if "function" in raw and isinstance(raw["function"], dict):
                            tool_name = raw["function"].get("name", "Unknown Tool")
                        elif "name" in raw:
                            tool_name = raw["name"]

                # Method 2: Try direct item attributes (fallback)
                if tool_name == "Unknown Tool":
                    if hasattr(event.item, "name") and event.item.name:
                        tool_name = event.item.name
                    elif hasattr(event.item, "function") and hasattr(
                        event.item.function, "name"
                    ):
                        tool_name = event.item.function.name
                    elif hasattr(event.item, "tool_name") and event.item.tool_name:
                        tool_name = event.item.tool_name

                # Use global tool descriptions
                desc = TOOL_DESCRIPTIONS.get(tool_name, f"🔧 Using tool: {tool_name}")
                print(f"\n{desc}")

                # Add progress estimates for known long-running tools
                if tool_name in [
                    "detect_recurring",
                    "analyze_and_advise",
                    "ingest_csv",
                    "ingest_pdfs",
                    "analyze_subscription_value",
                ]:
                    print("   ⏳ This may take a moment...")
            elif event.item.type == "tool_call_output_item":
                # Show completion with more detailed feedback
                output = getattr(event.item, "output", "")
                if output and len(str(output)) < 100:
                    print(f"   ✅ Completed: {str(output)[:100]}")
                else:
                    print("   ✅ Tool completed successfully")
            elif event.item.type == "message_output_item":
                # If we didn't get deltas, show the full message
                if not current_message:
                    try:
                        message_text = ItemHelpers.text_message_output(event.item)
                        if message_text:
                            print(f"📝 Response: {message_text}")
                    except Exception:
                        # Fallback: try to extract text from item
                        if hasattr(event.item, "text"):
                            print(f"📝 Response: {event.item.text}")
                        elif hasattr(event.item, "content"):
                            print(f"📝 Response: {event.item.content}")
                        elif hasattr(event.item, "output"):
                            print(f"📝 Response: {event.item.output}")

        elif event.type == "agent_updated_stream_event":
            print(f"\n🔄 Agent updated: {event.new_agent.name}")
            print(
                f"   💼 Specializing in: {event.new_agent.name.lower().replace('_', ' ')}"
            )

    # If no message was streamed, try to get final output
    if not current_message:
        try:
            if hasattr(result, "final_output") and result.final_output:
                print(f"\n📝 Response: {result.final_output}")
        except Exception:
            pass

    print("\n" + "=" * 50)
    print("✅ Processing complete.")


def clear_database() -> str:
    """Clear all data from the financial database."""
    with build_deps() as deps:
        deps.ensure_ready()
        cursor = deps.db.conn.cursor()

        # Get counts before clearing
        cursor.execute("SELECT COUNT(*) FROM transactions")
        transaction_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM memories")
        memory_count = cursor.fetchone()[0]

        # Clear all data
        cursor.execute("DELETE FROM transactions")
        cursor.execute("DELETE FROM memories")
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
    parser.add_argument(
        "-i", "--input", help="Prompt or command for the agent", default=None
    )
    parser.add_argument("--stream", action="store_true", help="Stream output events")
    parser.add_argument(
        "--interactive", action="store_true", help="Start interactive mode"
    )
    parser.add_argument(
        "--bootstrap", action="store_true", help="Ingest PDFs/CSVs from documents/"
    )
    parser.add_argument(
        "--no-session", action="store_true", help="Disable session memory"
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear all transactions and memories from database",
    )
    parser.add_argument(
        "--clear-sessions",
        action="store_true",
        help="Clear all conversation session files",
    )
    parser.add_argument(
        "--clear-all", action="store_true", help="Clear both database and session files"
    )
    parser.add_argument(
        "--export-bills",
        action="store_true",
        help="Export bills-only recurring payments",
    )
    parser.add_argument(
        "--bills-format",
        choices=["csv", "pdf", "excel", "json"],
        default="csv",
        help="Format for --export-bills",
    )
    parser.add_argument(
        "--monthly-cost",
        action="store_true",
        help="Print monthly cost summary (last full month)",
    )
    parser.add_argument(
        "--bills-only",
        action="store_true",
        help="When used with --monthly-cost, show bills-only view",
    )
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
        # Handle bills-only export in one-shot mode
        if args.export_bills:

            async def run_bills():
                payload = {
                    "format": args.bills_format,
                    "exclude_credit_repayment": True,
                    "bills_only": True,
                    "min_confidence": 0.7,
                }
                from agents import RunContextWrapper, ItemHelpers

                return await export_recurring_payments.on_invoke_tool(RunContextWrapper(deps), ItemHelpers.json_dumps(payload))  # type: ignore

            try:
                print(asyncio.run(run_bills()))
            except Exception as e:
                print(f"❌ Failed to export bills-only recurring: {e}")
            return

        # Handle monthly cost summary one-shot
        if args.monthly_cost:

            async def run_monthly():
                from .tools.costs import monthly_cost_summary

                payload = {
                    "last_full_month": True,
                    "bills_only": bool(args.bills_only),
                    "include_breakdown": True,
                }
                from agents import RunContextWrapper, ItemHelpers

                return await monthly_cost_summary.on_invoke_tool(RunContextWrapper(deps), ItemHelpers.json_dumps(payload))  # type: ignore

            try:
                with trace("Financial Agent Monthly Cost"):
                    print(asyncio.run(run_monthly()))
            except Exception as e:
                print(f"❌ Failed to compute monthly cost summary: {e}")
            return
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
            with trace("Financial Agent Interactive"):
                asyncio.run(interactive_mode(agent, deps, use_session=use_session))
        elif args.stream:
            # Streaming mode for single command
            user_input = args.input or "Analyze my recent spending."
            with trace("Financial Agent Streaming"):
                asyncio.run(
                    streaming_mode(agent, deps, user_input, use_session=use_session)
                )
        else:
            # Non-streaming mode for single command
            with trace("Financial Agent One-shot"):
                if use_session:
                    session_db_path = Path.home() / ".financial_agent" / "sessions.db"
                    session_db_path.parent.mkdir(parents=True, exist_ok=True)
                    _ensure_session_db_schema(session_db_path)
                    session = SQLiteSession("cli_session", str(session_db_path))
                    result = Runner.run_sync(
                        agent, args.input, context=deps, session=session
                    )
                else:
                    result = Runner.run_sync(agent, args.input, context=deps)
                print(result.final_output)


if __name__ == "__main__":
    main()
