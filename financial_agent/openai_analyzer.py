"""
OpenAI API log analyzer for Financial Agent.

This module provides tools to analyze OpenAI API call logs, track costs, performance,
and understand model behavior patterns.
"""

from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict, Counter
import statistics

from .openai_logger import OpenAIAPICall


class OpenAIAnalyzer:
    """Analyze OpenAI API call logs."""

    def __init__(self, log_file: Optional[Path] = None):
        """Initialize the OpenAI API log analyzer."""
        self.log_file = log_file or Path(__file__).parent / "logs" / "openai_api.log"
        self.calls: List[OpenAIAPICall] = []
        self.df: Optional[pd.DataFrame] = None

    def load_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Load OpenAI API logs with optional filtering."""
        self.calls = []

        if not self.log_file.exists():
            print(f"OpenAI API log file not found: {self.log_file}")
            return 0

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        # Skip system events
                        if data.get("type") == "system_event":
                            continue

                        # Create OpenAIAPICall object
                        call = OpenAIAPICall(**data)

                        # Apply filters
                        if session_id and call.session_id != session_id:
                            continue

                        call_time = datetime.fromisoformat(
                            call.timestamp.replace("Z", "+00:00")
                        )
                        if start_time and call_time < start_time:
                            continue
                        if end_time and call_time > end_time:
                            continue

                        self.calls.append(call)

                    except (json.JSONDecodeError, TypeError, KeyError) as e:
                        print(f"Error parsing API log line: {e}")
                        continue

        except Exception as e:
            print(f"Error loading OpenAI API log file: {e}")
            return 0

        # Create DataFrame for analysis
        if self.calls:
            self.df = pd.DataFrame([call.to_dict() for call in self.calls])
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        print(f"Loaded {len(self.calls)} OpenAI API calls")
        return len(self.calls)

    def get_cost_analysis(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze API costs."""
        if not self.calls:
            return {"error": "No API calls loaded"}

        filtered_calls = self.calls
        if session_id:
            filtered_calls = [
                call for call in self.calls if call.session_id == session_id
            ]

        if not filtered_calls:
            return {"error": f"No calls found for session {session_id}"}

        # Calculate costs by model
        costs_by_model = defaultdict(float)
        token_usage_by_model = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "total": 0}
        )

        total_cost = 0.0
        total_calls = len(filtered_calls)
        successful_calls = 0

        for call in filtered_calls:
            if call.error:
                continue

            successful_calls += 1
            cost = call.metrics.cost_usd
            total_cost += cost
            costs_by_model[call.model] += cost

            token_usage_by_model[call.model]["prompt"] += call.token_usage.prompt_tokens
            token_usage_by_model[call.model][
                "completion"
            ] += call.token_usage.completion_tokens
            token_usage_by_model[call.model]["total"] += call.token_usage.total_tokens

        # Find most expensive calls
        expensive_calls = sorted(
            [call for call in filtered_calls if not call.error],
            key=lambda x: x.metrics.cost_usd,
            reverse=True,
        )[:5]

        return {
            "total_cost": total_cost,
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0,
            "costs_by_model": dict(costs_by_model),
            "token_usage_by_model": dict(token_usage_by_model),
            "average_cost_per_call": (
                total_cost / successful_calls if successful_calls > 0 else 0
            ),
            "most_expensive_calls": [
                {
                    "model": call.model,
                    "cost": call.metrics.cost_usd,
                    "tokens": call.token_usage.total_tokens,
                    "timestamp": call.timestamp,
                    "duration_ms": call.metrics.duration_ms,
                }
                for call in expensive_calls
            ],
        }

    def get_performance_analysis(
        self, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze API performance metrics."""
        if not self.calls:
            return {"error": "No API calls loaded"}

        filtered_calls = self.calls
        if session_id:
            filtered_calls = [
                call for call in self.calls if call.session_id == session_id
            ]

        successful_calls = [call for call in filtered_calls if not call.error]

        if not successful_calls:
            return {"message": "No successful calls to analyze"}

        # Performance metrics
        durations = [call.metrics.duration_ms for call in successful_calls]
        tokens_per_second = [
            call.metrics.tokens_per_second
            for call in successful_calls
            if call.metrics.tokens_per_second > 0
        ]
        token_counts = [call.token_usage.total_tokens for call in successful_calls]

        # Performance by model
        performance_by_model = defaultdict(list)
        for call in successful_calls:
            performance_by_model[call.model].append(
                {
                    "duration_ms": call.metrics.duration_ms,
                    "tokens_per_second": call.metrics.tokens_per_second,
                    "total_tokens": call.token_usage.total_tokens,
                }
            )

        # Calculate model averages
        model_stats = {}
        for model, calls_data in performance_by_model.items():
            model_stats[model] = {
                "avg_duration_ms": sum(c["duration_ms"] for c in calls_data)
                / len(calls_data),
                "avg_tokens_per_second": sum(
                    c["tokens_per_second"]
                    for c in calls_data
                    if c["tokens_per_second"] > 0
                )
                / max(1, len([c for c in calls_data if c["tokens_per_second"] > 0])),
                "avg_tokens": sum(c["total_tokens"] for c in calls_data)
                / len(calls_data),
                "call_count": len(calls_data),
            }

        return {
            "overall_performance": {
                "total_successful_calls": len(successful_calls),
                "avg_duration_ms": statistics.mean(durations),
                "median_duration_ms": statistics.median(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "avg_tokens_per_second": (
                    statistics.mean(tokens_per_second) if tokens_per_second else 0
                ),
                "avg_tokens_per_call": statistics.mean(token_counts),
            },
            "performance_by_model": model_stats,
            "slowest_calls": [
                {
                    "model": call.model,
                    "duration_ms": call.metrics.duration_ms,
                    "tokens": call.token_usage.total_tokens,
                    "tokens_per_second": call.metrics.tokens_per_second,
                    "timestamp": call.timestamp,
                }
                for call in sorted(
                    successful_calls, key=lambda x: x.metrics.duration_ms, reverse=True
                )[:5]
            ],
        }

    def get_model_usage_analysis(
        self, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze model usage patterns."""
        if not self.calls:
            return {"error": "No API calls loaded"}

        filtered_calls = self.calls
        if session_id:
            filtered_calls = [
                call for call in self.calls if call.session_id == session_id
            ]

        # Model usage counts
        model_usage = Counter(call.model for call in filtered_calls)

        # Reasoning model detection (o1 models)
        reasoning_calls = [
            call
            for call in filtered_calls
            if "o1" in call.model or call.reasoning_content
        ]

        # Tool call patterns
        tool_call_patterns = []
        for call in filtered_calls:
            if call.request_data.get("tools"):
                tool_count = len(call.request_data["tools"])
                tool_call_patterns.append(
                    {
                        "model": call.model,
                        "tool_count": tool_count,
                        "timestamp": call.timestamp,
                    }
                )

        return {
            "model_distribution": dict(model_usage),
            "total_calls": len(filtered_calls),
            "unique_models": len(model_usage),
            "reasoning_calls": {
                "count": len(reasoning_calls),
                "percentage": (
                    len(reasoning_calls) / len(filtered_calls) * 100
                    if filtered_calls
                    else 0
                ),
                "models_used": list(set(call.model for call in reasoning_calls)),
            },
            "tool_usage": {
                "calls_with_tools": len(tool_call_patterns),
                "avg_tools_per_call": sum(p["tool_count"] for p in tool_call_patterns)
                / max(1, len(tool_call_patterns)),
            },
            "most_used_model": model_usage.most_common(1)[0] if model_usage else None,
        }

    def get_error_analysis(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze API errors."""
        if not self.calls:
            return {"error": "No API calls loaded"}

        filtered_calls = self.calls
        if session_id:
            filtered_calls = [
                call for call in self.calls if call.session_id == session_id
            ]

        error_calls = [call for call in filtered_calls if call.error]

        if not error_calls:
            return {"message": "No errors found - excellent!"}

        # Error patterns
        error_types = Counter()
        error_models = Counter()

        for call in error_calls:
            # Extract error type
            error_msg = call.error.lower()
            if "rate limit" in error_msg:
                error_types["Rate Limit"] += 1
            elif "timeout" in error_msg:
                error_types["Timeout"] += 1
            elif "invalid" in error_msg:
                error_types["Invalid Request"] += 1
            elif "auth" in error_msg:
                error_types["Authentication"] += 1
            else:
                error_types["Other"] += 1

            error_models[call.model] += 1

        return {
            "total_errors": len(error_calls),
            "error_rate": (
                len(error_calls) / len(filtered_calls) * 100 if filtered_calls else 0
            ),
            "error_types": dict(error_types),
            "models_with_errors": dict(error_models),
            "recent_errors": [
                {
                    "timestamp": call.timestamp,
                    "model": call.model,
                    "error": (
                        call.error[:200] + "..."
                        if len(call.error) > 200
                        else call.error
                    ),
                }
                for call in sorted(
                    error_calls, key=lambda x: x.timestamp, reverse=True
                )[:5]
            ],
        }

    def generate_openai_report(
        self, session_id: Optional[str] = None, output_file: Optional[Path] = None
    ) -> str:
        """Generate comprehensive OpenAI API analysis report."""
        if not self.calls:
            return "No OpenAI API calls loaded. Call load_logs() first."

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("OPENAI API ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append(f"Log file: {self.log_file}")
        report_lines.append("")

        # Cost Analysis
        cost_analysis = self.get_cost_analysis(session_id)
        if "error" not in cost_analysis:
            report_lines.append("COST ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(f"Total API Cost: ${cost_analysis['total_cost']:.6f}")
            report_lines.append(f"Total API Calls: {cost_analysis['total_calls']}")
            report_lines.append(
                f"Successful Calls: {cost_analysis['successful_calls']}"
            )
            report_lines.append(f"Success Rate: {cost_analysis['success_rate']:.2%}")
            report_lines.append(
                f"Average Cost per Call: ${cost_analysis['average_cost_per_call']:.6f}"
            )
            report_lines.append("")

            if cost_analysis["costs_by_model"]:
                report_lines.append("Costs by Model:")
                for model, cost in sorted(
                    cost_analysis["costs_by_model"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    report_lines.append(f"  {model}: ${cost:.6f}")
                report_lines.append("")

        # Performance Analysis
        perf_analysis = self.get_performance_analysis(session_id)
        if "error" not in perf_analysis:
            report_lines.append("PERFORMANCE ANALYSIS")
            report_lines.append("-" * 40)
            overall = perf_analysis["overall_performance"]
            report_lines.append(
                f"Average Response Time: {overall['avg_duration_ms']:.0f}ms"
            )
            report_lines.append(
                f"Median Response Time: {overall['median_duration_ms']:.0f}ms"
            )
            report_lines.append(
                f"Average Tokens/Second: {overall['avg_tokens_per_second']:.1f}"
            )
            report_lines.append(
                f"Average Tokens/Call: {overall['avg_tokens_per_call']:.0f}"
            )
            report_lines.append("")

            if perf_analysis["slowest_calls"]:
                report_lines.append("Slowest API Calls:")
                for call in perf_analysis["slowest_calls"][:3]:
                    report_lines.append(
                        f"  {call['model']}: {call['duration_ms']:.0f}ms ({call['tokens']} tokens)"
                    )
                report_lines.append("")

        # Model Usage
        model_analysis = self.get_model_usage_analysis(session_id)
        if "error" not in model_analysis:
            report_lines.append("MODEL USAGE ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Models Used: {model_analysis['unique_models']}")
            report_lines.append(
                f"Reasoning Model Calls: {model_analysis['reasoning_calls']['count']} ({model_analysis['reasoning_calls']['percentage']:.1f}%)"
            )
            report_lines.append(
                f"Calls with Tools: {model_analysis['tool_usage']['calls_with_tools']}"
            )
            report_lines.append("")

            if model_analysis["most_used_model"]:
                most_used, count = model_analysis["most_used_model"]
                report_lines.append(f"Most Used Model: {most_used} ({count} calls)")
                report_lines.append("")

        # Error Analysis
        error_analysis = self.get_error_analysis(session_id)
        if "error" not in error_analysis and error_analysis.get("total_errors", 0) > 0:
            report_lines.append("ERROR ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Errors: {error_analysis['total_errors']}")
            report_lines.append(f"Error Rate: {error_analysis['error_rate']:.2f}%")

            if error_analysis.get("error_types"):
                report_lines.append("Error Types:")
                for error_type, count in error_analysis["error_types"].items():
                    report_lines.append(f"  {error_type}: {count}")
                report_lines.append("")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"OpenAI API analysis report written to: {output_file}")

        return report

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get information about recent sessions with API calls."""
        if not self.calls:
            return []

        # Group by session_id
        sessions = defaultdict(list)
        for call in self.calls:
            if call.session_id:
                sessions[call.session_id].append(call)

        session_summaries = []
        for session_id, session_calls in sessions.items():
            session_calls.sort(key=lambda x: x.timestamp)

            total_cost = sum(
                call.metrics.cost_usd for call in session_calls if not call.error
            )
            total_tokens = sum(
                call.token_usage.total_tokens
                for call in session_calls
                if not call.error
            )
            errors = len([call for call in session_calls if call.error])

            session_summaries.append(
                {
                    "session_id": session_id,
                    "start_time": session_calls[0].timestamp,
                    "end_time": session_calls[-1].timestamp,
                    "total_calls": len(session_calls),
                    "successful_calls": len(session_calls) - errors,
                    "errors": errors,
                    "total_cost": total_cost,
                    "total_tokens": total_tokens,
                    "models_used": list(set(call.model for call in session_calls)),
                    "avg_response_time": sum(
                        call.metrics.duration_ms
                        for call in session_calls
                        if not call.error
                    )
                    / max(1, len(session_calls) - errors),
                }
            )

        # Sort by start time (most recent first)
        session_summaries.sort(key=lambda x: x["start_time"], reverse=True)

        return session_summaries[:limit]
