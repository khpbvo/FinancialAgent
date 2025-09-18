"""
Comprehensive logging system for Financial Agent debugging.

This module provides structured logging capabilities for:
- Agent initialization and construction
- Tool calls with input/output capture
- Agent orchestration and handoffs
- Performance metrics and error tracking
- Debug log analysis and visualization
"""

from __future__ import annotations
import logging
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Callable
from pathlib import Path
from functools import wraps
from contextlib import contextmanager
import traceback
import inspect
from dataclasses import dataclass, asdict
from enum import Enum

from agents import RunContextWrapper


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEventType(str, Enum):
    AGENT_INIT = "agent_init"
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_COMPLETE = "tool_call_complete"
    TOOL_CALL_ERROR = "tool_call_error"
    ORCHESTRATOR_ROUTE = "orchestrator_route"
    HANDOFF_START = "handoff_start"
    HANDOFF_COMPLETE = "handoff_complete"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    CONTEXT_UPDATE = "context_update"
    PERFORMANCE_METRIC = "performance_metric"
    USER_INPUT = "user_input"
    SYSTEM_EVENT = "system_event"


@dataclass
class LogEvent:
    """Structured log event for comprehensive agent debugging."""

    timestamp: str
    event_type: LogEventType
    level: LogLevel
    session_id: str
    agent_name: Optional[str]
    tool_name: Optional[str]
    message: str
    data: Dict[str, Any]
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)


class AgentLogger:
    """Centralized logging system for agent debugging."""

    def __init__(
        self,
        log_file: Optional[Path] = None,
        console_level: LogLevel = LogLevel.INFO,
        file_level: LogLevel = LogLevel.DEBUG,
        max_data_size: int = 10000,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the agent logger.

        Args:
            log_file: Path to log file (defaults to financial_agent/logs/debug.log)
            console_level: Console logging level
            file_level: File logging level
            max_data_size: Maximum size of data field in characters
            session_id: Unique session identifier (auto-generated if None)
        """
        self.session_id = session_id or self._generate_session_id()
        self.max_data_size = max_data_size
        self.log_file = log_file or Path(__file__).parent / "logs" / "debug.log"

        # Ensure log directory exists
        self.log_file.parent.mkdir(exist_ok=True)

        # Setup Python logger
        self.logger = logging.getLogger(f"financial_agent.{self.session_id}")
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_level.value))
        console_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler for structured JSON logs
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, file_level.value))
        file_formatter = logging.Formatter("%(message)s")  # Raw JSON
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Track active tool calls and agents
        self._active_tools: Dict[str, Dict[str, Any]] = {}
        self._active_agents: Dict[str, Dict[str, Any]] = {}

        self.log_system_event(
            "AgentLogger initialized",
            {
                "session_id": self.session_id,
                "log_file": str(self.log_file),
                "console_level": console_level,
                "file_level": file_level,
            },
        )

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session_{int(time.time() * 1000)}"

    def _truncate_data(self, data: Any) -> Any:
        """Truncate large data for logging."""
        if isinstance(data, str) and len(data) > self.max_data_size:
            return data[: self.max_data_size] + "...[TRUNCATED]"
        if isinstance(data, dict):
            return {k: self._truncate_data(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._truncate_data(item) for item in data]
        return data

    def log_event(
        self,
        event_type: LogEventType,
        message: str,
        level: LogLevel = LogLevel.INFO,
        agent_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        execution_time_ms: Optional[float] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Log a structured event."""

        # Handle error information
        error_str = None
        stack_trace = None
        if error:
            error_str = str(error)
            stack_trace = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )

        # Create event
        event = LogEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            level=level,
            session_id=self.session_id,
            agent_name=agent_name,
            tool_name=tool_name,
            message=message,
            data=self._truncate_data(data or {}),
            execution_time_ms=execution_time_ms,
            error=error_str,
            stack_trace=stack_trace,
        )

        # Log to both console (formatted) and file (JSON)
        log_level = getattr(logging, level.value)
        formatted_message = self._format_console_message(event)
        self.logger.log(log_level, formatted_message)

        # Also write raw JSON to file for structured analysis
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")

    def _format_console_message(self, event: LogEvent) -> str:
        """Format event for console display."""
        parts = [event.message]

        if event.agent_name:
            parts.append(f"[Agent: {event.agent_name}]")
        if event.tool_name:
            parts.append(f"[Tool: {event.tool_name}]")
        if event.execution_time_ms:
            parts.append(f"({event.execution_time_ms:.2f}ms)")
        if event.error:
            parts.append(f"ERROR: {event.error}")

        return " ".join(parts)

    # Convenience methods for different event types
    def log_agent_init(self, agent_name: str, agent_config: Dict[str, Any]):
        """Log agent initialization."""
        self.log_event(
            LogEventType.AGENT_INIT,
            f"Initializing agent: {agent_name}",
            LogLevel.INFO,
            agent_name=agent_name,
            data={"agent_config": agent_config},
        )

    def log_agent_start(
        self, agent_name: str, user_input: str, context_info: Dict[str, Any]
    ):
        """Log agent execution start."""
        agent_id = f"{agent_name}_{int(time.time() * 1000)}"
        self._active_agents[agent_id] = {
            "start_time": time.time(),
            "agent_name": agent_name,
            "user_input": user_input,
        }

        self.log_event(
            LogEventType.AGENT_START,
            f"Starting agent execution: {agent_name}",
            LogLevel.INFO,
            agent_name=agent_name,
            data={
                "user_input": user_input,
                "context_info": context_info,
                "agent_id": agent_id,
            },
        )
        return agent_id

    def log_agent_complete(self, agent_id: str, result: Any):
        """Log agent execution completion."""
        if agent_id not in self._active_agents:
            return

        agent_info = self._active_agents.pop(agent_id)
        execution_time = (time.time() - agent_info["start_time"]) * 1000

        self.log_event(
            LogEventType.AGENT_COMPLETE,
            f"Agent execution completed: {agent_info['agent_name']}",
            LogLevel.INFO,
            agent_name=agent_info["agent_name"],
            execution_time_ms=execution_time,
            data={
                "agent_id": agent_id,
                "result": str(result)[:1000],  # Truncate result
                "success": True,
            },
        )

    def log_agent_error(self, agent_id: str, error: Exception):
        """Log agent execution error."""
        if agent_id not in self._active_agents:
            return

        agent_info = self._active_agents.pop(agent_id)
        execution_time = (time.time() - agent_info["start_time"]) * 1000

        self.log_event(
            LogEventType.AGENT_ERROR,
            f"Agent execution failed: {agent_info['agent_name']}",
            LogLevel.ERROR,
            agent_name=agent_info["agent_name"],
            execution_time_ms=execution_time,
            error=error,
            data={"agent_id": agent_id, "user_input": agent_info["user_input"]},
        )

    def log_tool_call_start(
        self, tool_name: str, agent_name: str, inputs: Dict[str, Any]
    ):
        """Log tool call start."""
        call_id = f"{tool_name}_{int(time.time() * 1000000)}"  # Microsecond precision
        self._active_tools[call_id] = {
            "start_time": time.time(),
            "tool_name": tool_name,
            "agent_name": agent_name,
            "inputs": inputs,
        }

        self.log_event(
            LogEventType.TOOL_CALL_START,
            f"Tool call started: {tool_name}",
            LogLevel.DEBUG,
            agent_name=agent_name,
            tool_name=tool_name,
            data={"call_id": call_id, "inputs": inputs},
        )
        return call_id

    def log_tool_call_complete(self, call_id: str, output: Any):
        """Log tool call completion."""
        if call_id not in self._active_tools:
            return

        tool_info = self._active_tools.pop(call_id)
        execution_time = (time.time() - tool_info["start_time"]) * 1000

        self.log_event(
            LogEventType.TOOL_CALL_COMPLETE,
            f"Tool call completed: {tool_info['tool_name']}",
            LogLevel.DEBUG,
            agent_name=tool_info["agent_name"],
            tool_name=tool_info["tool_name"],
            execution_time_ms=execution_time,
            data={
                "call_id": call_id,
                "inputs": tool_info["inputs"],
                "output": str(output)[:2000],  # Truncate output
                "success": True,
            },
        )

    def log_tool_call_error(self, call_id: str, error: Exception):
        """Log tool call error."""
        if call_id not in self._active_tools:
            return

        tool_info = self._active_tools.pop(call_id)
        execution_time = (time.time() - tool_info["start_time"]) * 1000

        self.log_event(
            LogEventType.TOOL_CALL_ERROR,
            f"Tool call failed: {tool_info['tool_name']}",
            LogLevel.ERROR,
            agent_name=tool_info["agent_name"],
            tool_name=tool_info["tool_name"],
            execution_time_ms=execution_time,
            error=error,
            data={"call_id": call_id, "inputs": tool_info["inputs"]},
        )

    def log_orchestrator_route(
        self, user_input: str, selected_agent: str, reasoning: str
    ):
        """Log orchestrator routing decision."""
        self.log_event(
            LogEventType.ORCHESTRATOR_ROUTE,
            f"Routing to specialist: {selected_agent}",
            LogLevel.INFO,
            agent_name="Orchestrator",
            data={
                "user_input": user_input,
                "selected_agent": selected_agent,
                "reasoning": reasoning,
            },
        )

    def log_handoff(self, from_agent: str, to_agent: str, context_data: Dict[str, Any]):
        """Log agent handoff."""
        self.log_event(
            LogEventType.HANDOFF_START,
            f"Handoff: {from_agent} → {to_agent}",
            LogLevel.INFO,
            agent_name=from_agent,
            data={
                "from_agent": from_agent,
                "to_agent": to_agent,
                "context_data": context_data,
            },
        )

    def log_user_input(
        self, user_input: str, context_info: Optional[Dict[str, Any]] = None
    ):
        """Log user input."""
        self.log_event(
            LogEventType.USER_INPUT,
            "User input received",
            LogLevel.INFO,
            data={"user_input": user_input, "context_info": context_info or {}},
        )

    def log_system_event(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log system-level events."""
        self.log_event(LogEventType.SYSTEM_EVENT, message, LogLevel.INFO, data=data)

    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "ms",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log performance metrics."""
        self.log_event(
            LogEventType.PERFORMANCE_METRIC,
            f"Performance metric: {metric_name} = {value}{unit}",
            LogLevel.DEBUG,
            data={
                "metric_name": metric_name,
                "value": value,
                "unit": unit,
                "metadata": metadata or {},
            },
        )


# Global logger instance
_global_logger: Optional[AgentLogger] = None


def get_logger() -> AgentLogger:
    """Get or create the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = AgentLogger()
    return _global_logger


def set_logger(logger: AgentLogger) -> None:
    """Set the global logger instance."""
    global _global_logger
    _global_logger = logger


def log_tool_calls(func: Callable) -> Callable:
    """Decorator to automatically log tool calls."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()

        # Extract agent name if available
        agent_name = "Unknown"
        if args and isinstance(args[0], RunContextWrapper):
            agent_name = "FinancialAgent"  # Default

        # Get function signature for input logging
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Filter out context from inputs for cleaner logging
        inputs = {k: v for k, v in bound_args.arguments.items() if k != "ctx"}

        # Start logging
        call_id = logger.log_tool_call_start(func.__name__, agent_name, inputs)

        try:
            # Execute the function
            start_time = time.time()
            result = func(*args, **kwargs)

            # Log completion
            logger.log_tool_call_complete(call_id, result)

            # Log performance
            execution_time = (time.time() - start_time) * 1000
            logger.log_performance_metric(
                f"tool_execution_{func.__name__}",
                execution_time,
                "ms",
                {"success": True},
            )

            return result

        except Exception as e:
            # Log error
            logger.log_tool_call_error(call_id, e)

            # Log performance
            execution_time = (time.time() - start_time) * 1000
            logger.log_performance_metric(
                f"tool_execution_{func.__name__}",
                execution_time,
                "ms",
                {"success": False, "error": str(e)},
            )

            raise

    return wrapper


@contextmanager
def log_agent_execution(
    agent_name: str, user_input: str, context_info: Optional[Dict[str, Any]] = None
):
    """Context manager to log agent execution."""
    logger = get_logger()

    # Log user input
    logger.log_user_input(user_input, context_info)

    # Start agent execution logging
    agent_id = logger.log_agent_start(agent_name, user_input, context_info or {})

    try:
        yield agent_id
    except Exception as e:
        logger.log_agent_error(agent_id, e)
        raise


def log_agent_result(agent_id: str, result: Any):
    """Log agent execution result."""
    logger = get_logger()
    logger.log_agent_complete(agent_id, result)


# Convenience functions for common logging patterns
def log_debug(message: str, **kwargs):
    """Log debug message."""
    get_logger().log_event(
        LogEventType.SYSTEM_EVENT, message, LogLevel.DEBUG, data=kwargs
    )


def log_info(message: str, **kwargs):
    """Log info message."""
    get_logger().log_event(
        LogEventType.SYSTEM_EVENT, message, LogLevel.INFO, data=kwargs
    )


def log_warning(message: str, **kwargs):
    """Log warning message."""
    get_logger().log_event(
        LogEventType.SYSTEM_EVENT, message, LogLevel.WARNING, data=kwargs
    )


def log_error(message: str, error: Optional[Exception] = None, **kwargs):
    """Log error message."""
    get_logger().log_event(
        LogEventType.SYSTEM_EVENT, message, LogLevel.ERROR, error=error, data=kwargs
    )
