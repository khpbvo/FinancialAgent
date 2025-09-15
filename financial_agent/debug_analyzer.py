"""
Debug log analyzer and visualization tool for Financial Agent.

This module provides tools to analyze agent execution logs, identify performance
bottlenecks, trace tool call sequences, and visualize agent behavior patterns.
"""

from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .logging_utils import LogEvent


class LogAnalyzer:
    """Analyze and visualize Financial Agent debug logs."""

    def __init__(self, log_file: Optional[Path] = None):
        """Initialize the log analyzer.

        Args:
            log_file: Path to log file (defaults to standard debug.log)
        """
        self.log_file = log_file or Path(__file__).parent / "logs" / "debug.log"
        self.events: List[LogEvent] = []
        self.df: Optional[pd.DataFrame] = None

    def load_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Load logs from file with optional filtering.

        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time
            session_id: Filter events for specific session

        Returns:
            Number of events loaded
        """
        self.events = []

        if not self.log_file.exists():
            print(f"Log file not found: {self.log_file}")
            return 0

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                buffer: List[str] = []
                for raw in f:
                    line = raw.rstrip("\n")
                    if not buffer:
                        # Start a JSON buffer only if this looks like JSON
                        if line.lstrip().startswith("{"):
                            buffer = [line]
                        else:
                            continue
                    else:
                        buffer.append(line)

                    # Try to parse the current buffer as a JSON object
                    joined = "\n".join(buffer)
                    try:
                        event_data = json.loads(joined)
                    except json.JSONDecodeError:
                        # Keep buffering until valid JSON
                        continue

                    # Coerce schema variants to LogEvent
                    if "event_type" not in event_data and "type" in event_data:
                        event_data["event_type"] = event_data.pop("type")
                    event_data.setdefault("level", "INFO")
                    event_data.setdefault(
                        "session_id", event_data.get("session", "unknown_session")
                    )
                    event_data.setdefault("agent_name", None)
                    event_data.setdefault("tool_name", None)
                    event_data.setdefault("message", "")
                    event_data.setdefault("data", {})
                    event_data.setdefault("execution_time_ms", None)
                    event_data.setdefault("error", None)
                    event_data.setdefault("stack_trace", None)

                    # Build event
                    try:
                        event = LogEvent(**event_data)
                    except TypeError:
                        buffer = []
                        continue

                    # Apply filters
                    if session_id and event.session_id != session_id:
                        buffer = []
                        continue

                    try:
                        event_time = datetime.fromisoformat(
                            event.timestamp.replace("Z", "+00:00")
                        )
                    except Exception:
                        event_time = None
                    if start_time and event_time and event_time < start_time:
                        buffer = []
                        continue
                    if end_time and event_time and event_time > end_time:
                        buffer = []
                        continue

                    self.events.append(event)
                    buffer = []  # Reset after successful parse

                # If leftover buffer, attempt one last parse
                if buffer:
                    try:
                        event_data = json.loads("\n".join(buffer))
                        if "event_type" not in event_data and "type" in event_data:
                            event_data["event_type"] = event_data.pop("type")
                        event = LogEvent(**event_data)
                        self.events.append(event)
                    except Exception:
                        pass

        except Exception as e:
            print(f"Error loading log file: {e}")
            return 0

        # Create DataFrame for easier analysis
        if self.events:
            self.df = pd.DataFrame([event.to_dict() for event in self.events])
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        print(f"Loaded {len(self.events)} log events")
        return len(self.events)

    def get_top_slow_tool_calls(
        self, session_id: Optional[str] = None, top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Return the top-N slowest tool call completions by execution_time_ms."""
        if self.df is None or self.df.empty:
            return []
        df = self.df
        if session_id:
            df = df[df["session_id"] == session_id]
        tool_done = df[
            (df["event_type"] == "tool_call_complete")
            & (df["execution_time_ms"].notna())
        ]
        if tool_done.empty:
            return []
        top = tool_done.sort_values("execution_time_ms", ascending=False).head(top_n)
        results = []
        for _, row in top.iterrows():
            results.append(
                {
                    "timestamp": row["timestamp"].isoformat(),
                    "tool_name": row["tool_name"],
                    "agent_name": row.get("agent_name"),
                    "execution_time_ms": row["execution_time_ms"],
                    "message": row["message"],
                }
            )
        return results

    def get_tool_time_aggregation(
        self, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Aggregate total and average execution time per tool."""
        if self.df is None or self.df.empty:
            return []
        df = self.df
        if session_id:
            df = df[df["session_id"] == session_id]
        done = df[
            (df["event_type"] == "tool_call_complete")
            & (df["execution_time_ms"].notna())
        ]
        if done.empty:
            return []
        grouped = done.groupby("tool_name")["execution_time_ms"]
        agg = (
            grouped.agg(["count", "mean", "sum", "max"])
            .reset_index()
            .rename(
                columns={
                    "count": "calls",
                    "mean": "avg_ms",
                    "sum": "total_ms",
                    "max": "max_ms",
                }
            )
        )
        agg = agg.sort_values("total_ms", ascending=False)
        return agg.to_dict(orient="records")

    def get_session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for a session or all sessions.

        Args:
            session_id: Specific session to analyze (None for all)

        Returns:
            Dictionary with session statistics
        """
        if not self.df is not None or self.df.empty:
            return {"error": "No events loaded"}

        df = self.df
        if session_id:
            df = df[df["session_id"] == session_id]

        if df.empty:
            return {"error": f"No events found for session {session_id}"}

        summary = {
            "session_info": {
                "session_id": session_id or "ALL",
                "start_time": df["timestamp"].min().isoformat(),
                "end_time": df["timestamp"].max().isoformat(),
                "duration_minutes": (
                    df["timestamp"].max() - df["timestamp"].min()
                ).total_seconds()
                / 60,
                "total_events": len(df),
            },
            "event_counts": dict(df["event_type"].value_counts()),
            "level_counts": dict(df["level"].value_counts()),
            "agent_activity": dict(
                df[df["agent_name"].notna()]["agent_name"].value_counts()
            ),
            "tool_usage": dict(df[df["tool_name"].notna()]["tool_name"].value_counts()),
            "errors": len(df[df["level"] == "ERROR"]),
            "warnings": len(df[df["level"] == "WARNING"]),
        }

        # Performance metrics
        perf_events = df[df["event_type"] == "performance_metric"]
        if not perf_events.empty:
            summary["performance"] = {
                "metric_count": len(perf_events),
                "metrics_recorded": list(
                    perf_events["data"]
                    .apply(lambda x: x.get("metric_name", "unknown"))
                    .unique()
                ),
            }

        # Tool execution times
        tool_complete_events = df[df["event_type"] == "tool_call_complete"]
        if not tool_complete_events.empty and "execution_time_ms" in df.columns:
            exec_times = tool_complete_events["execution_time_ms"].dropna()
            if not exec_times.empty:
                summary["tool_performance"] = {
                    "avg_execution_time_ms": exec_times.mean(),
                    "median_execution_time_ms": exec_times.median(),
                    "min_execution_time_ms": exec_times.min(),
                    "max_execution_time_ms": exec_times.max(),
                    "total_tool_calls": len(exec_times),
                }

        return summary

    def get_tool_performance_analysis(self) -> Dict[str, Any]:
        """Analyze tool performance metrics."""
        if self.df is None or self.df.empty:
            return {"error": "No events loaded"}

        tool_events = self.df[
            (
                self.df["event_type"].isin(
                    ["tool_call_start", "tool_call_complete", "tool_call_error"]
                )
            )
            & (self.df["tool_name"].notna())
        ].copy()

        if tool_events.empty:
            return {"message": "No tool call events found"}

        # Group by tool name
        tool_stats = {}

        for tool_name in tool_events["tool_name"].unique():
            tool_data = tool_events[tool_events["tool_name"] == tool_name]

            starts = tool_data[tool_data["event_type"] == "tool_call_start"]
            completions = tool_data[tool_data["event_type"] == "tool_call_complete"]
            errors = tool_data[tool_data["event_type"] == "tool_call_error"]

            exec_times = completions["execution_time_ms"].dropna()

            stats = {
                "total_calls": len(starts),
                "successful_calls": len(completions),
                "failed_calls": len(errors),
                "success_rate": (
                    len(completions) / len(starts) if len(starts) > 0 else 0
                ),
            }

            if not exec_times.empty:
                stats.update(
                    {
                        "avg_execution_time_ms": exec_times.mean(),
                        "median_execution_time_ms": exec_times.median(),
                        "min_execution_time_ms": exec_times.min(),
                        "max_execution_time_ms": exec_times.max(),
                        "std_execution_time_ms": exec_times.std(),
                    }
                )

            tool_stats[tool_name] = stats

        # Sort by total calls
        sorted_tools = sorted(
            tool_stats.items(), key=lambda x: x[1]["total_calls"], reverse=True
        )

        return {
            "tool_performance": dict(sorted_tools),
            "summary": {
                "total_unique_tools": len(tool_stats),
                "total_tool_calls": sum(
                    stats["total_calls"] for stats in tool_stats.values()
                ),
                "overall_success_rate": sum(
                    stats["successful_calls"] for stats in tool_stats.values()
                )
                / max(1, sum(stats["total_calls"] for stats in tool_stats.values())),
            },
        }

    def get_error_analysis(self) -> Dict[str, Any]:
        """Analyze errors and failures."""
        if self.df is None or self.df.empty:
            return {"error": "No events loaded"}

        error_events = self.df[self.df["level"] == "ERROR"].copy()

        if error_events.empty:
            return {"message": "No errors found - excellent!"}

        analysis = {
            "total_errors": len(error_events),
            "error_types": dict(error_events["event_type"].value_counts()),
            "errors_by_agent": dict(
                error_events[error_events["agent_name"].notna()][
                    "agent_name"
                ].value_counts()
            ),
            "errors_by_tool": dict(
                error_events[error_events["tool_name"].notna()][
                    "tool_name"
                ].value_counts()
            ),
            "error_timeline": [],
        }

        # Error timeline (last 10 errors)
        recent_errors = error_events.sort_values("timestamp").tail(10)
        for _, error in recent_errors.iterrows():
            analysis["error_timeline"].append(
                {
                    "timestamp": error["timestamp"].isoformat(),
                    "event_type": error["event_type"],
                    "agent_name": error.get("agent_name"),
                    "tool_name": error.get("tool_name"),
                    "message": error["message"],
                    "error": error.get("error"),
                }
            )

        # Common error patterns
        error_messages = error_events["error"].dropna()
        if not error_messages.empty:
            # Count error message patterns (first 50 chars)
            error_patterns = error_messages.apply(
                lambda x: str(x)[:50] + "..." if len(str(x)) > 50 else str(x)
            )
            analysis["common_errors"] = dict(error_patterns.value_counts().head(10))

        return analysis

    def trace_execution_flow(
        self, session_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Trace the execution flow for a specific session.

        Args:
            session_id: Session to trace
            limit: Maximum number of events to return

        Returns:
            Chronological list of key events
        """
        if self.df is None or self.df.empty:
            return []

        session_events = self.df[self.df["session_id"] == session_id].copy()
        if session_events.empty:
            return []

        # Filter to key events for flow tracing
        key_event_types = [
            "user_input",
            "agent_start",
            "orchestrator_route",
            "handoff_start",
            "tool_call_start",
            "tool_call_complete",
            "tool_call_error",
            "agent_complete",
            "agent_error",
        ]

        flow_events = (
            session_events[session_events["event_type"].isin(key_event_types)]
            .sort_values("timestamp")
            .head(limit)
        )

        trace = []
        for _, event in flow_events.iterrows():
            trace_entry = {
                "timestamp": event["timestamp"].isoformat(),
                "event_type": event["event_type"],
                "message": event["message"],
                "agent_name": event.get("agent_name"),
                "tool_name": event.get("tool_name"),
                "execution_time_ms": event.get("execution_time_ms"),
            }

            # Add relevant data based on event type
            if event["event_type"] == "user_input":
                trace_entry["user_input"] = event["data"].get("user_input", "")
            elif event["event_type"] == "orchestrator_route":
                trace_entry["selected_agent"] = event["data"].get("selected_agent", "")
                trace_entry["reasoning"] = event["data"].get("reasoning", "")
            elif "tool_call" in event["event_type"]:
                if "inputs" in event["data"]:
                    trace_entry["tool_inputs"] = event["data"]["inputs"]
                if "output" in event["data"]:
                    trace_entry["tool_output"] = (
                        event["data"]["output"][:200] + "..."
                        if len(event["data"]["output"]) > 200
                        else event["data"]["output"]
                    )

            trace.append(trace_entry)

        return trace

    def get_performance_trends(
        self, metric_name: Optional[str] = None, time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Analyze performance trends over time.

        Args:
            metric_name: Specific metric to analyze (None for all)
            time_window_minutes: Time window for trend analysis

        Returns:
            Performance trend analysis
        """
        if self.df is None or self.df.empty:
            return {"error": "No events loaded"}

        perf_events = self.df[self.df["event_type"] == "performance_metric"].copy()
        if perf_events.empty:
            return {"message": "No performance metrics found"}

        # Extract metric data
        perf_data = []
        for _, event in perf_events.iterrows():
            data = event["data"]
            if metric_name and data.get("metric_name") != metric_name:
                continue

            perf_data.append(
                {
                    "timestamp": event["timestamp"],
                    "metric_name": data.get("metric_name"),
                    "value": data.get("value"),
                    "unit": data.get("unit", "unknown"),
                    "metadata": data.get("metadata", {}),
                }
            )

        if not perf_data:
            return {"message": f"No performance data found for metric: {metric_name}"}

        perf_df = pd.DataFrame(perf_data)

        # Group by metric name
        trends = {}
        for metric in perf_df["metric_name"].unique():
            metric_data = perf_df[perf_df["metric_name"] == metric].sort_values(
                "timestamp"
            )

            if len(metric_data) < 2:
                continue

            values = metric_data["value"].dropna()
            if values.empty:
                continue

            # Calculate trend statistics
            trends[metric] = {
                "count": len(values),
                "avg": values.mean(),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
                "std": values.std(),
                "unit": metric_data["unit"].iloc[0],
                "first_value": values.iloc[0],
                "last_value": values.iloc[-1],
                "trend": (
                    "improving"
                    if values.iloc[-1] < values.iloc[0]
                    else "degrading" if values.iloc[-1] > values.iloc[0] else "stable"
                ),
            }

            # Time-based analysis
            if len(metric_data) >= 10:  # Need enough points for trend analysis
                recent_data = metric_data[
                    metric_data["timestamp"]
                    >= (
                        metric_data["timestamp"].max()
                        - pd.Timedelta(minutes=time_window_minutes)
                    )
                ]
                if len(recent_data) > 1:
                    recent_values = recent_data["value"].dropna()
                    trends[metric]["recent_avg"] = recent_values.mean()
                    trends[metric]["recent_trend"] = (
                        "improving"
                        if recent_values.iloc[-1] < recent_values.iloc[0]
                        else "degrading"
                    )

        return {
            "performance_trends": trends,
            "analysis_window_minutes": time_window_minutes,
            "total_metrics_analyzed": len(trends),
        }

    def generate_debug_report(
        self, session_id: Optional[str] = None, output_file: Optional[Path] = None
    ) -> str:
        """Generate comprehensive debug report.

        Args:
            session_id: Specific session to analyze (None for all)
            output_file: File to write report to (None for return as string)

        Returns:
            Debug report as formatted string
        """
        if not self.events:
            return "No events loaded. Call load_logs() first."

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FINANCIAL AGENT DEBUG REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append(f"Log file: {self.log_file}")
        report_lines.append("")

        # Session summary
        summary = self.get_session_summary(session_id)
        if "error" not in summary:
            report_lines.append("SESSION SUMMARY")
            report_lines.append("-" * 40)
            session_info = summary["session_info"]
            report_lines.append(f"Session ID: {session_info['session_id']}")
            report_lines.append(
                f"Duration: {session_info['duration_minutes']:.2f} minutes"
            )
            report_lines.append(f"Total events: {session_info['total_events']}")
            report_lines.append(f"Errors: {summary['errors']}")
            report_lines.append(f"Warnings: {summary['warnings']}")
            report_lines.append("")

            # Event breakdown
            report_lines.append("EVENT BREAKDOWN")
            report_lines.append("-" * 40)
            for event_type, count in summary["event_counts"].items():
                report_lines.append(f"{event_type}: {count}")
            report_lines.append("")

            # Agent activity
            if summary["agent_activity"]:
                report_lines.append("AGENT ACTIVITY")
                report_lines.append("-" * 40)
                for agent, count in summary["agent_activity"].items():
                    report_lines.append(f"{agent}: {count} events")
                report_lines.append("")

            # Tool usage
            if summary["tool_usage"]:
                report_lines.append("TOOL USAGE")
                report_lines.append("-" * 40)
                for tool, count in summary["tool_usage"].items():
                    report_lines.append(f"{tool}: {count} calls")
                report_lines.append("")

        # Tool performance
        tool_perf = self.get_tool_performance_analysis()
        if "error" not in tool_perf:
            report_lines.append("TOOL PERFORMANCE ANALYSIS")
            report_lines.append("-" * 40)
            for tool, stats in tool_perf["tool_performance"].items():
                report_lines.append(f"{tool}:")
                report_lines.append(f"  Total calls: {stats['total_calls']}")
                report_lines.append(f"  Success rate: {stats['success_rate']:.2%}")
                if "avg_execution_time_ms" in stats:
                    report_lines.append(
                        f"  Avg time: {stats['avg_execution_time_ms']:.2f}ms"
                    )
                    report_lines.append(
                        f"  Max time: {stats['max_execution_time_ms']:.2f}ms"
                    )
            report_lines.append("")

        # Error analysis
        error_analysis = self.get_error_analysis()
        if "error" not in error_analysis and error_analysis.get("total_errors", 0) > 0:
            report_lines.append("ERROR ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(f"Total errors: {error_analysis['total_errors']}")

            if error_analysis.get("common_errors"):
                report_lines.append("Most common errors:")
                for error, count in list(error_analysis["common_errors"].items())[:5]:
                    report_lines.append(f"  {count}x: {error}")

            report_lines.append("")

        # Execution trace (if specific session)
        if session_id and self.df is not None:
            trace = self.trace_execution_flow(session_id, limit=20)
            if trace:
                report_lines.append("EXECUTION TRACE (Last 20 key events)")
                report_lines.append("-" * 40)
                for entry in trace:
                    timestamp = entry["timestamp"]
                    event_type = entry["event_type"]
                    message = entry["message"]

                    line = f"{timestamp} | {event_type:15} | {message}"
                    if entry.get("execution_time_ms"):
                        line += f" ({entry['execution_time_ms']:.2f}ms)"
                    report_lines.append(line)
                report_lines.append("")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Debug report written to: {output_file}")

        return report

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get information about recent sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session information dictionaries
        """
        if self.df is None or self.df.empty:
            return []

        # Group by session_id and get session info
        sessions = []
        session_groups = self.df.groupby("session_id")

        for session_id, group in session_groups:
            session_info = {
                "session_id": session_id,
                "start_time": group["timestamp"].min().isoformat(),
                "end_time": group["timestamp"].max().isoformat(),
                "duration_seconds": (
                    group["timestamp"].max() - group["timestamp"].min()
                ).total_seconds(),
                "total_events": len(group),
                "agents_used": list(
                    group[group["agent_name"].notna()]["agent_name"].unique()
                ),
                "tools_used": list(
                    group[group["tool_name"].notna()]["tool_name"].unique()
                ),
                "errors": len(group[group["level"] == "ERROR"]),
                "warnings": len(group[group["level"] == "WARNING"]),
            }

            # Get user inputs for this session
            user_inputs = (
                group[group["event_type"] == "user_input"]["data"]
                .apply(lambda x: x.get("user_input", "") if isinstance(x, dict) else "")
                .tolist()
            )
            session_info["user_inputs"] = [inp for inp in user_inputs if inp][
                :3
            ]  # First 3 inputs

            sessions.append(session_info)

        # Sort by start time (most recent first)
        sessions.sort(key=lambda x: x["start_time"], reverse=True)

        return sessions[:limit]


def create_debug_tool():
    """Create a debug analysis tool as a function tool."""
    from agents import function_tool, RunContextWrapper

    @function_tool
    def analyze_debug_logs(
        ctx: "RunContextWrapper",
        session_id: Optional[str] = None,
        hours_back: int = 24,
        generate_report: bool = True,
    ) -> str:
        """Analyze Financial Agent debug logs for performance and error analysis.

        Args:
            session_id: Specific session to analyze (None for recent sessions)
            hours_back: How many hours back to analyze logs
            generate_report: Whether to generate a full debug report
        """
        analyzer = LogAnalyzer()

        # Load logs from the specified time window
        start_time = datetime.now() - timedelta(hours=hours_back)
        count = analyzer.load_logs(start_time=start_time, session_id=session_id)

        if count == 0:
            return f"No logs found in the last {hours_back} hours"

        if generate_report:
            return analyzer.generate_debug_report(session_id)
        else:
            # Return quick summary
            summary = analyzer.get_session_summary(session_id)
            if "error" in summary:
                return summary["error"]

            result = "Debug Log Analysis Summary\n"
            result += "========================\n"
            result += f"Events analyzed: {summary['session_info']['total_events']}\n"
            result += f"Time span: {summary['session_info']['duration_minutes']:.1f} minutes\n"
            result += f"Errors: {summary['errors']}\n"
            result += f"Warnings: {summary['warnings']}\n"

            if summary["tool_usage"]:
                result += "\nTop tools used:\n"
                for tool, count in list(summary["tool_usage"].items())[:5]:
                    result += f"  {tool}: {count} calls\n"

            return result

    return analyze_debug_logs
