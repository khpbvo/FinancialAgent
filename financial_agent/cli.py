from __future__ import annotations
import argparse
import asyncio
import os

from agents import Runner

from .agent import build_agent, build_deps


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial Agent CLI")
    parser.add_argument("-i", "--input", help="Prompt or command for the agent", default=None)
    parser.add_argument("--stream", action="store_true", help="Stream output events")
    parser.add_argument("--bootstrap", action="store_true", help="Ingest PDFs/CSVs from documents/")
    args = parser.parse_args()

    deps = build_deps()
    agent = build_agent()

    if args.bootstrap:
        # quick ingestion pass
        from .bootstrap import bootstrap_documents
        print(bootstrap_documents())

    if args.stream:
        async def main_stream() -> None:
            result = Runner.run_streamed(agent, input=args.input or "Analyze my recent spending.", context=deps)
            print("=== Run starting ===")
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    # print deltas if needed; skip to keep concise
                    pass
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        print("-- Tool called")
                    elif event.item.type == "tool_call_output_item":
                        print(f"-- Tool output: {event.item.output}")
                    elif event.item.type == "message_output_item":
                        # best-effort: some SDKs offer ItemHelpers to extract text
                        try:
                            from agents import ItemHelpers
                            print(ItemHelpers.text_message_output(event.item))
                        except Exception:
                            pass
        asyncio.run(main_stream())
    else:
        result = Runner.run_sync(agent, args.input or "Analyze my recent spending.", context=deps)
        print(result.final_output)

if __name__ == "__main__":
    main()
