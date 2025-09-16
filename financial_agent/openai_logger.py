"""
OpenAI API call logging system for Financial Agent.

This module provides comprehensive logging of all OpenAI API interactions including:
- Request/response content with full message history
- Token usage and cost tracking
- Performance metrics (latency, throughput)
- Error analysis and retry logic
- Model reasoning traces (for o1 models)
"""

from __future__ import annotations
import json
import time
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager


from .logging_utils import get_logger, LogEventType, LogLevel


@dataclass
class TokenUsage:
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0  # For cached prompts
    reasoning_tokens: int = 0  # For o1 models


@dataclass
class APICallMetrics:
    """Performance metrics for API calls."""

    start_time: float
    end_time: float
    duration_ms: float
    request_size_bytes: int
    response_size_bytes: int
    tokens_per_second: float
    cost_usd: float = 0.0


@dataclass
class OpenAIAPICall:
    """Complete record of an OpenAI API call."""

    call_id: str
    timestamp: str
    model: str
    endpoint: str
    request_data: Dict[str, Any]
    response_data: Optional[Dict[str, Any]]
    token_usage: TokenUsage
    metrics: APICallMetrics
    error: Optional[str] = None
    reasoning_content: Optional[str] = None  # For o1 models
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)


class OpenAILogger:
    """Logger specifically for OpenAI API calls."""

    # Token pricing (approximate, as of 2024)
    TOKEN_PRICES = {
        "gpt-4": {"input": 0.00003, "output": 0.00006},
        "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},
        "gpt-4o": {"input": 0.000005, "output": 0.000015},
        "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
        "gpt-5": {"input": 0.00001, "output": 0.00003},  # Estimated
        "o1-preview": {"input": 0.000015, "output": 0.00006},
        "o1-mini": {"input": 0.000003, "output": 0.000012},
    }

    def __init__(self, log_file: Optional[Path] = None, session_id: str = ""):
        """Initialize OpenAI API logger."""
        self.session_id = session_id
        self.log_file = log_file or Path(__file__).parent / "logs" / "openai_api.log"

        # Ensure log directory exists
        self.log_file.parent.mkdir(exist_ok=True)

        # Setup dedicated logger for OpenAI calls
        self.logger = logging.getLogger(f"openai_api.{session_id}")
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler for API calls
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(message)s")  # Raw JSON
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Also log to console with less detail
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s | OpenAI API | %(levelname)s | %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Track ongoing API calls
        self._active_calls: Dict[str, Dict[str, Any]] = {}

        self.log_system_event("OpenAI API Logger initialized")

    def _generate_call_id(self) -> str:
        """Generate unique call ID."""
        return f"api_{int(time.time() * 1000000)}"

    def _estimate_cost(self, model: str, token_usage: TokenUsage) -> float:
        """Estimate API call cost based on token usage."""
        if model not in self.TOKEN_PRICES:
            # Try to match partial model names
            for price_model in self.TOKEN_PRICES:
                if price_model in model.lower():
                    model = price_model
                    break
            else:
                return 0.0

        prices = self.TOKEN_PRICES[model]
        input_cost = token_usage.prompt_tokens * prices["input"]
        output_cost = token_usage.completion_tokens * prices["output"]

        # Add reasoning token cost for o1 models (if applicable)
        if "o1" in model and token_usage.reasoning_tokens > 0:
            # Reasoning tokens typically cost same as input tokens
            input_cost += token_usage.reasoning_tokens * prices["input"]

        return input_cost + output_cost

    def _extract_token_usage(self, response_data: Dict[str, Any]) -> TokenUsage:
        """Extract token usage from API response."""
        usage = response_data.get("usage", {})

        return TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            cached_tokens=usage.get("prompt_tokens_details", {}).get(
                "cached_tokens", 0
            ),
            reasoning_tokens=usage.get("completion_tokens_details", {}).get(
                "reasoning_tokens", 0
            ),
        )

    def _extract_reasoning_content(
        self, response_data: Dict[str, Any]
    ) -> Optional[str]:
        """Extract reasoning content from o1 model responses."""
        choices = response_data.get("choices", [])
        if not choices:
            return None

        # Look for reasoning in the first choice
        choice = choices[0]
        message = choice.get("message", {})

        # o1 models may include reasoning in a special field
        if "reasoning" in message:
            return message["reasoning"]

        # Or it might be in the content with special markers
        content = message.get("content", "")
        if "<thinking>" in content and "</thinking>" in content:
            start = content.find("<thinking>") + len("<thinking>")
            end = content.find("</thinking>")
            return content[start:end].strip()

        return None

    def _sanitize_request_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request data for logging (remove sensitive info if any)."""
        sanitized = data.copy()

        # Remove API key if accidentally included
        if "api_key" in sanitized:
            sanitized["api_key"] = "[REDACTED]"

        # Limit message content length for readability
        if "messages" in sanitized:
            messages = []
            for msg in sanitized["messages"]:
                msg_copy = msg.copy()
                if "content" in msg_copy and len(str(msg_copy["content"])) > 2000:
                    msg_copy["content"] = (
                        str(msg_copy["content"])[:2000] + "...[TRUNCATED]"
                    )
                messages.append(msg_copy)
            sanitized["messages"] = messages

        return sanitized

    def start_api_call(
        self, model: str, endpoint: str, request_data: Dict[str, Any]
    ) -> str:
        """Start logging an API call."""
        call_id = self._generate_call_id()
        start_time = time.time()

        # Store call info
        self._active_calls[call_id] = {
            "model": model,
            "endpoint": endpoint,
            "request_data": self._sanitize_request_data(request_data),
            "start_time": start_time,
            "request_size": len(json.dumps(request_data, default=str)),
        }

        # Log to main logger
        get_logger().log_event(
            LogEventType.LLM_REQUEST,
            f"OpenAI API call started: {model} {endpoint}",
            LogLevel.DEBUG,
            data={
                "call_id": call_id,
                "model": model,
                "endpoint": endpoint,
                "request_preview": (
                    str(request_data)[:200] + "..."
                    if len(str(request_data)) > 200
                    else str(request_data)
                ),
            },
        )

        return call_id

    def complete_api_call(
        self,
        call_id: str,
        response_data: Dict[str, Any],
        error: Optional[Exception] = None,
    ) -> None:
        """Complete logging an API call."""
        if call_id not in self._active_calls:
            return

        call_info = self._active_calls.pop(call_id)
        end_time = time.time()
        duration_ms = (end_time - call_info["start_time"]) * 1000

        # Extract token usage and metrics
        token_usage = (
            self._extract_token_usage(response_data) if response_data else TokenUsage()
        )
        response_size = (
            len(json.dumps(response_data, default=str)) if response_data else 0
        )

        # Calculate tokens per second
        tokens_per_second = (
            token_usage.total_tokens / (duration_ms / 1000) if duration_ms > 0 else 0
        )

        # Estimate cost
        cost = self._estimate_cost(call_info["model"], token_usage)

        # Create metrics
        metrics = APICallMetrics(
            start_time=call_info["start_time"],
            end_time=end_time,
            duration_ms=duration_ms,
            request_size_bytes=call_info["request_size"],
            response_size_bytes=response_size,
            tokens_per_second=tokens_per_second,
            cost_usd=cost,
        )

        # Extract reasoning content (for o1 models)
        reasoning_content = (
            self._extract_reasoning_content(response_data) if response_data else None
        )

        # Create complete API call record
        api_call = OpenAIAPICall(
            call_id=call_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=call_info["model"],
            endpoint=call_info["endpoint"],
            request_data=call_info["request_data"],
            response_data=response_data,
            token_usage=token_usage,
            metrics=metrics,
            error=str(error) if error else None,
            reasoning_content=reasoning_content,
            session_id=self.session_id,
        )

        # Log to OpenAI-specific log file
        self.logger.debug(api_call.to_json())

        # Also log summary to main logger
        level = LogLevel.ERROR if error else LogLevel.DEBUG
        message = f"OpenAI API call {'failed' if error else 'completed'}: {call_info['model']}"

        get_logger().log_event(
            LogEventType.LLM_RESPONSE,
            message,
            level,
            execution_time_ms=duration_ms,
            error=error,
            data={
                "call_id": call_id,
                "model": call_info["model"],
                "tokens_used": token_usage.total_tokens,
                "cost_usd": cost,
                "tokens_per_second": tokens_per_second,
                "has_reasoning": bool(reasoning_content),
            },
        )

        # Log to console with summary
        if error:
            self.logger.error(f"API call failed after {duration_ms:.0f}ms: {error}")
        else:
            self.logger.info(
                f"API call completed: {call_info['model']} | "
                f"{duration_ms:.0f}ms | "
                f"{token_usage.total_tokens} tokens | "
                f"${cost:.6f} | "
                f"{tokens_per_second:.1f} tok/s"
            )

    def log_system_event(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log system-level events."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "system_event",
            "session_id": self.session_id,
            "message": message,
            "data": data or {},
        }

        self.logger.debug(json.dumps(event, default=str))


# Global OpenAI logger instance
_openai_logger: Optional[OpenAILogger] = None


def get_openai_logger() -> OpenAILogger:
    """Get or create the global OpenAI logger instance."""
    global _openai_logger
    if _openai_logger is None:
        session_id = (
            get_logger().session_id if hasattr(get_logger(), "session_id") else ""
        )
        _openai_logger = OpenAILogger(session_id=session_id)
    return _openai_logger


def set_openai_logger(logger: OpenAILogger) -> None:
    """Set the global OpenAI logger instance."""
    global _openai_logger
    _openai_logger = logger


# Monkey patch OpenAI client to intercept API calls
def _patch_openai_client():
    """Monkey patch OpenAI client methods to add logging."""

    # Store original methods
    original_create = None
    original_acreate = None

    def logged_create(self, **kwargs):
        """Logged version of chat.completions.create."""
        nonlocal original_create
        if original_create is None:
            return self._original_create(**kwargs)

        logger = get_openai_logger()
        model = kwargs.get("model", "unknown")
        call_id = logger.start_api_call(model, "chat.completions.create", kwargs)

        try:
            response = original_create(self, **kwargs)
            logger.complete_api_call(
                call_id,
                (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else response.__dict__
                ),
            )
            return response
        except Exception as e:
            logger.complete_api_call(call_id, {}, error=e)
            raise

    async def logged_acreate(self, **kwargs):
        """Logged version of async chat.completions.create."""
        nonlocal original_acreate
        if original_acreate is None:
            return await self._original_acreate(**kwargs)

        logger = get_openai_logger()
        model = kwargs.get("model", "unknown")
        call_id = logger.start_api_call(model, "chat.completions.create", kwargs)

        try:
            response = await original_acreate(self, **kwargs)
            logger.complete_api_call(
                call_id,
                (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else response.__dict__
                ),
            )
            return response
        except Exception as e:
            logger.complete_api_call(call_id, {}, error=e)
            raise

    # Apply patches
    try:
        from openai.resources.chat import completions

        # Note: Avoid patching Responses API streaming methods to prevent
        # interference with SDK streaming internals.

        # Sync client
        if hasattr(completions.Completions, "create") and not hasattr(
            completions.Completions, "_original_create"
        ):
            original_create = completions.Completions.create
            completions.Completions._original_create = original_create
            completions.Completions.create = logged_create

        # Async client
        if hasattr(completions.AsyncCompletions, "create") and not hasattr(
            completions.AsyncCompletions, "_original_acreate"
        ):
            original_acreate = completions.AsyncCompletions.create
            completions.AsyncCompletions._original_acreate = original_acreate
            completions.AsyncCompletions.create = logged_acreate

        # Intentionally not patching responses.create/stream to avoid
        # pickling issues in streaming pipelines.

    except ImportError:
        pass  # OpenAI not available


# Auto-patch on import
_patch_openai_client()


@contextmanager
def openai_call_logging(model: str, endpoint: str, **kwargs):
    """Context manager for manual OpenAI call logging."""
    logger = get_openai_logger()
    call_id = logger.start_api_call(model, endpoint, kwargs)

    try:
        yield call_id
    except Exception as e:
        logger.complete_api_call(call_id, {}, error=e)
        raise


def log_openai_response(call_id: str, response: Any, error: Optional[Exception] = None):
    """Manually log OpenAI response."""
    logger = get_openai_logger()
    response_data = {}

    if response:
        if hasattr(response, "model_dump"):
            response_data = response.model_dump()
        elif hasattr(response, "__dict__"):
            response_data = response.__dict__
        else:
            response_data = {"response": str(response)}

    logger.complete_api_call(call_id, response_data, error=error)
