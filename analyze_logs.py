#!/usr/bin/env python3
"""
Analyze Financial Agent logs.

This script provides comprehensive analysis of both structured agent logs
and OpenAI API call logs.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add the project to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from financial_agent.debug_analyzer import LogAnalyzer
from financial_agent.openai_analyzer import OpenAIAnalyzer


def main():
    print("📊 FINANCIAL AGENT LOG ANALYSIS")
    print("=" * 50)

    # Analyze debug logs
    print("\n🔍 AGENT DEBUG LOGS")
    print("-" * 30)

    debug_analyzer = LogAnalyzer()
    debug_count = debug_analyzer.load_logs()

    if debug_count > 0:
        # Get recent sessions
        recent_sessions = debug_analyzer.get_recent_sessions(limit=5)
        if recent_sessions:
            print(f"\nRecent Sessions ({len(recent_sessions)}):")
            for session in recent_sessions:
                print(f"  Session: {session['session_id']}")
                print(f"    Time: {session['start_time']} - {session['end_time']}")
                print(
                    f"    Events: {session['total_events']}, Errors: {session['errors']}"
                )
                print(
                    f"    Tools used: {', '.join(session['tools_used'][:3])}{'...' if len(session['tools_used']) > 3 else ''}"
                )
                print()

        # Performance summary
        tool_perf = debug_analyzer.get_tool_performance_analysis()
        if tool_perf and "error" not in tool_perf and "message" not in tool_perf:
            print("Tool Performance Summary:")
            print(f"  Total tool calls: {tool_perf['summary']['total_tool_calls']}")
            print(f"  Success rate: {tool_perf['summary']['overall_success_rate']:.2%}")
            print()

            # Top 3 most used tools
            top_tools = sorted(
                tool_perf["tool_performance"].items(),
                key=lambda x: x[1]["total_calls"],
                reverse=True,
            )[:3]

            if top_tools:
                print("  Most used tools:")
                for tool, stats in top_tools:
                    print(
                        f"    {tool}: {stats['total_calls']} calls, {stats['success_rate']:.2%} success"
                    )
                    if "avg_execution_time_ms" in stats:
                        print(f"      Avg time: {stats['avg_execution_time_ms']:.1f}ms")
                print()

        # Deep profiling: top slow calls and time aggregation by tool
        slow_calls = debug_analyzer.get_top_slow_tool_calls(top_n=10)
        if slow_calls:
            print("Top 10 slowest tool calls:")
            for c in slow_calls:
                print(
                    f"  {c['timestamp']} | {c['tool_name']} | {c['execution_time_ms']:.1f}ms"
                )
            print()

        agg = debug_analyzer.get_tool_time_aggregation()
        if agg:
            print("Time spent by tool (total/avg/max):")
            for row in agg[:10]:
                print(
                    f"  {row['tool_name']}: total {row['total_ms']:.0f}ms, avg {row['avg_ms']:.1f}ms over {row['calls']} calls (max {row['max_ms']:.1f}ms)"
                )
            print()
        else:
            print("No tool call timing data found in logs.")
            print()

    # Analyze OpenAI API logs
    print("\n🤖 OPENAI API LOGS")
    print("-" * 30)

    openai_analyzer = OpenAIAnalyzer()
    api_count = openai_analyzer.load_logs()

    if api_count > 0:
        # Cost analysis
        cost_analysis = openai_analyzer.get_cost_analysis()
        if "error" not in cost_analysis:
            print("API Cost Summary:")
            print(f"  Total cost: ${cost_analysis['total_cost']:.6f}")
            print(f"  Total calls: {cost_analysis['total_calls']}")
            print(f"  Success rate: {cost_analysis['success_rate']:.2%}")
            print(f"  Avg cost per call: ${cost_analysis['average_cost_per_call']:.6f}")
            print()

            if cost_analysis["costs_by_model"]:
                print("  Costs by model:")
                for model, cost in sorted(
                    cost_analysis["costs_by_model"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    print(f"    {model}: ${cost:.6f}")
                print()

        # Performance analysis
        perf_analysis = openai_analyzer.get_performance_analysis()
        if "error" not in perf_analysis:
            overall = perf_analysis["overall_performance"]
            print("API Performance Summary:")
            print(f"  Avg response time: {overall['avg_duration_ms']:.0f}ms")
            print(f"  Avg tokens/sec: {overall['avg_tokens_per_second']:.1f}")
            print(f"  Avg tokens/call: {overall['avg_tokens_per_call']:.0f}")
            print()

        # Model usage
        model_analysis = openai_analyzer.get_model_usage_analysis()
        if "error" not in model_analysis:
            print("Model Usage Summary:")
            print(f"  Unique models: {model_analysis['unique_models']}")
            if model_analysis["most_used_model"]:
                most_used, count = model_analysis["most_used_model"]
                print(f"  Most used: {most_used} ({count} calls)")
            print(
                f"  Reasoning calls: {model_analysis['reasoning_calls']['count']} ({model_analysis['reasoning_calls']['percentage']:.1f}%)"
            )
            print()
    else:
        print("  No OpenAI API calls found in logs.")
        print("  Note: API call interception may need additional setup.")

    print("\n📁 LOG FILES:")
    logs_dir = Path("financial_agent/logs")
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            size_kb = log_file.stat().st_size / 1024
            modified = datetime.fromtimestamp(log_file.stat().st_mtime)
            print(
                f"  {log_file.name}: {size_kb:.1f}KB (modified: {modified.strftime('%Y-%m-%d %H:%M')})"
            )

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
