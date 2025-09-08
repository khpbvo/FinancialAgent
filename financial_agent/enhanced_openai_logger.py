"""
Enhanced OpenAI API call logging for Financial Agent.

This module provides deeper interception of OpenAI API calls by patching
the actual HTTP client calls made by the OpenAI SDK, ensuring we capture
all API interactions regardless of how they're wrapped by the Agents SDK.
"""

from __future__ import annotations
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
import httpx

from .logging_utils import get_logger, LogEventType, LogLevel


@dataclass
class EnhancedTokenUsage:
    """Enhanced token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancedTokenUsage":
        """Create from API response usage dict."""
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            cached_tokens=data.get("prompt_tokens_details", {}).get("cached_tokens", 0),
            reasoning_tokens=data.get("completion_tokens_details", {}).get(
                "reasoning_tokens", 0
            ),
        )


@dataclass
class EnhancedAPICall:
    """Enhanced record of an OpenAI API call with full HTTP details."""

    call_id: str
    timestamp: str
    session_id: str

    # HTTP details
    method: str
    url: str
    headers: Dict[str, Any]

    # Request/Response
    request_body: Dict[str, Any]
    response_body: Optional[Dict[str, Any]]
    status_code: Optional[int]

    # Timing and metrics
    start_time: float
    end_time: Optional[float]
    duration_ms: Optional[float]

    # Token and cost info
    model: str
    token_usage: Optional[EnhancedTokenUsage]
    estimated_cost_usd: float = 0.0

    # Error info
    error: Optional[str] = None
    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        if self.token_usage:
            result["token_usage"] = asdict(self.token_usage)
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)


class EnhancedOpenAILogger:
    """Enhanced OpenAI API logger that intercepts at the HTTP level."""

    # Updated token pricing (approximate, as of early 2024)
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
        """Initialize enhanced OpenAI API logger."""
        self.session_id = session_id or f"session_{int(time.time() * 1000)}"
        self.log_file = (
            log_file or Path(__file__).parent / "logs" / "openai_enhanced.log"
        )

        # Ensure log directory exists
        self.log_file.parent.mkdir(exist_ok=True)

        # Setup dedicated logger
        self.logger = logging.getLogger(f"openai_enhanced.{self.session_id}")
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler for detailed API calls
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(message)s")  # Raw JSON
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Track ongoing calls
        self._active_calls: Dict[str, EnhancedAPICall] = {}

        # Apply HTTP-level patches
        self._apply_patches()

        self.log_system_event("Enhanced OpenAI API Logger initialized")

    def _generate_call_id(self) -> str:
        """Generate unique call ID."""
        return f"enhanced_api_{uuid.uuid4().hex[:12]}"

    def _estimate_cost(self, model: str, token_usage: EnhancedTokenUsage) -> float:
        """Estimate API call cost."""
        # Find matching model in pricing
        pricing_model = model.lower()
        for price_model in self.TOKEN_PRICES:
            if price_model in pricing_model:
                pricing_model = price_model
                break
        else:
            return 0.0

        prices = self.TOKEN_PRICES[pricing_model]
        input_cost = token_usage.prompt_tokens * prices["input"]
        output_cost = token_usage.completion_tokens * prices["output"]

        # Add reasoning token cost for o1 models
        if "o1" in model and token_usage.reasoning_tokens > 0:
            input_cost += token_usage.reasoning_tokens * prices["input"]

        return input_cost + output_cost

    def _sanitize_headers(self, headers: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize headers to remove sensitive information."""
        sanitized = dict(headers)

        # Remove or mask authorization headers
        if "authorization" in sanitized:
            auth_val = sanitized["authorization"]
            if isinstance(auth_val, str) and len(auth_val) > 10:
                sanitized["authorization"] = f"{auth_val[:10]}...{auth_val[-4:]}"

        if "api-key" in sanitized:
            api_key = sanitized["api-key"]
            if isinstance(api_key, str) and len(api_key) > 10:
                sanitized["api-key"] = f"{api_key[:10]}...{api_key[-4:]}"

        return sanitized

    def _extract_model_from_request(self, request_body: Dict[str, Any]) -> str:
        """Extract model name from request body."""
        return request_body.get("model", "unknown")

    def _is_openai_request(self, url: str) -> bool:
        """Check if this is an OpenAI API request."""
        openai_indicators = [
            "api.openai.com",
            "openai.com",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/traces",
        ]
        return any(indicator in url for indicator in openai_indicators)

    def start_http_call(
        self, method: str, url: str, headers: Dict[str, Any], content: bytes
    ) -> Optional[str]:
        """Start tracking an HTTP call if it's OpenAI-related."""
        # Debug: log all HTTP calls to see what we might be missing
        if "openai" in url.lower() or "gpt" in url.lower() or "/v1/" in url:
            print(f"🔍 HTTP Call Debug: {method} {url}")

        if not self._is_openai_request(url):
            return None

        call_id = self._generate_call_id()

        # Parse request body
        request_body = {}
        try:
            if content:
                request_body = json.loads(content.decode("utf-8"))
        except Exception:
            request_body = {
                "raw_content": content.decode("utf-8", errors="ignore")[:500]
            }

        # Create call record
        api_call = EnhancedAPICall(
            call_id=call_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=self.session_id,
            method=method,
            url=url,
            headers=self._sanitize_headers(headers),
            request_body=request_body,
            response_body=None,
            status_code=None,
            start_time=time.time(),
            end_time=None,
            duration_ms=None,
            model=self._extract_model_from_request(request_body),
            token_usage=None,
            error=None,
            error_type=None,
        )

        self._active_calls[call_id] = api_call

        # Log to main logger
        get_logger().log_event(
            LogEventType.LLM_REQUEST,
            f"Enhanced OpenAI HTTP call started: {method} {url}",
            LogLevel.DEBUG,
            data={
                "call_id": call_id,
                "model": api_call.model,
                "method": method,
                "url": url[:100] + "..." if len(url) > 100 else url,
            },
        )

        return call_id

    def complete_http_call(
        self,
        call_id: str,
        status_code: int,
        response_content: bytes,
        error: Optional[Exception] = None,
    ):
        """Complete tracking an HTTP call."""
        if call_id not in self._active_calls:
            return

        api_call = self._active_calls.pop(call_id)
        api_call.end_time = time.time()
        api_call.duration_ms = (api_call.end_time - api_call.start_time) * 1000
        api_call.status_code = status_code

        # Parse response
        if response_content and not error:
            try:
                api_call.response_body = json.loads(response_content.decode("utf-8"))

                # Extract token usage if available
                if api_call.response_body and "usage" in api_call.response_body:
                    api_call.token_usage = EnhancedTokenUsage.from_dict(
                        api_call.response_body["usage"]
                    )
                    api_call.estimated_cost_usd = self._estimate_cost(
                        api_call.model, api_call.token_usage
                    )

            except Exception as parse_error:
                api_call.response_body = {
                    "raw_content": response_content.decode("utf-8", errors="ignore")[
                        :1000
                    ],
                    "parse_error": str(parse_error),
                }

        # Handle errors
        if error:
            api_call.error = str(error)
            api_call.error_type = type(error).__name__

        # Log the complete call
        self.logger.debug(api_call.to_json())

        # Log to main logger
        level = LogLevel.ERROR if error else LogLevel.DEBUG
        message = f"Enhanced OpenAI HTTP call {'failed' if error else 'completed'}: {api_call.model}"

        data = {
            "call_id": call_id,
            "model": api_call.model,
            "status_code": status_code,
            "duration_ms": api_call.duration_ms,
        }

        if api_call.token_usage:
            data.update(
                {
                    "tokens_used": api_call.token_usage.total_tokens,
                    "cost_usd": api_call.estimated_cost_usd,
                    "tokens_per_second": (
                        api_call.token_usage.total_tokens
                        / (api_call.duration_ms / 1000)
                        if api_call.duration_ms > 0
                        else 0
                    ),
                }
            )

        get_logger().log_event(
            LogEventType.LLM_RESPONSE,
            message,
            level,
            execution_time_ms=api_call.duration_ms,
            error=error,
            data=data,
        )

        # Console summary
        if error:
            print(
                f"❌ OpenAI API call failed after {api_call.duration_ms:.0f}ms: {error}"
            )
        else:
            tokens_info = ""
            cost_info = ""
            if api_call.token_usage:
                tokens_info = f" | {api_call.token_usage.total_tokens} tokens"
                cost_info = f" | ${api_call.estimated_cost_usd:.6f}"

            print(
                f"✅ OpenAI API call completed: {api_call.model} | {api_call.duration_ms:.0f}ms{tokens_info}{cost_info}"
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

    def _apply_patches(self):
        """Apply HTTP-level patches to intercept OpenAI calls."""
        # Patch httpx client methods
        self._patch_httpx()

        # Also try to patch the OpenAI client directly
        self._patch_openai_client()

    def _patch_httpx(self):
        """Patch httpx client to intercept HTTP calls."""
        original_request = httpx.Client.request
        original_arequest = httpx.AsyncClient.request

        def logged_request(self, method: str, url: str, **kwargs):
            """Logged version of httpx.Client.request."""
            # Extract headers and content
            headers = kwargs.get("headers", {})
            content = kwargs.get("content", b"")
            if isinstance(content, str):
                content = content.encode("utf-8")

            # Start logging if it's an OpenAI call
            logger = get_enhanced_openai_logger()
            call_id = logger.start_http_call(method, str(url), dict(headers), content)

            try:
                response = original_request(self, method, url, **kwargs)
                if call_id:
                    # Never consume response.content for OpenAI calls; it may be streaming (SSE)
                    logger.complete_http_call(
                        call_id, getattr(response, "status_code", 0), b""
                    )
                return response
            except Exception as e:
                if call_id:
                    logger.complete_http_call(call_id, 0, b"", error=e)
                raise

        async def logged_arequest(self, method: str, url: str, **kwargs):
            """Logged version of httpx.AsyncClient.request."""
            headers = kwargs.get("headers", {})
            content = kwargs.get("content", b"")
            if isinstance(content, str):
                content = content.encode("utf-8")

            logger = get_enhanced_openai_logger()
            call_id = logger.start_http_call(method, str(url), dict(headers), content)

            try:
                response = await original_arequest(self, method, url, **kwargs)
                if call_id:
                    # Never consume response.content for OpenAI calls; it may be streaming (SSE)
                    logger.complete_http_call(
                        call_id, getattr(response, "status_code", 0), b""
                    )
                return response
            except Exception as e:
                if call_id:
                    logger.complete_http_call(call_id, 0, b"", error=e)
                raise

        # Apply patches if not already patched
        if not hasattr(httpx.Client, "_original_request"):
            httpx.Client._original_request = original_request  # type: ignore[attr-defined]
            httpx.Client.request = logged_request  # type: ignore[assignment]

        if not hasattr(httpx.AsyncClient, "_original_arequest"):
            httpx.AsyncClient._original_arequest = original_arequest  # type: ignore[attr-defined]
            httpx.AsyncClient.request = logged_arequest  # type: ignore[assignment]

    def _patch_openai_client(self):
        """Patch OpenAI client methods as backup."""
        try:
            from openai.resources.chat import completions

            # Get original methods if not already patched
            if not hasattr(completions.Completions, "_enhanced_original_create"):
                original_create = completions.Completions.create

                def enhanced_logged_create(self, **kwargs):
                    model = kwargs.get("model", "unknown")

                    # Log via main logging system as fallback
                    get_logger().log_event(
                        LogEventType.LLM_REQUEST,
                        f"OpenAI Chat Completion API call: {model}",
                        LogLevel.DEBUG,
                        data={"model": model, "method": "chat.completions.create"},
                    )

                    start_time = time.time()
                    try:
                        response = original_create(self, **kwargs)
                        duration_ms = (time.time() - start_time) * 1000

                        # Extract token info if available
                        token_info = {}
                        if hasattr(response, "usage") and response.usage:
                            token_info = {
                                "tokens_used": response.usage.total_tokens,
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                            }

                        get_logger().log_event(
                            LogEventType.LLM_RESPONSE,
                            f"OpenAI Chat Completion completed: {model}",
                            LogLevel.DEBUG,
                            execution_time_ms=duration_ms,
                            data={**token_info, "model": model},
                        )

                        return response
                    except Exception as e:
                        duration_ms = (time.time() - start_time) * 1000
                        get_logger().log_event(
                            LogEventType.LLM_RESPONSE,
                            f"OpenAI Chat Completion failed: {model}",
                            LogLevel.ERROR,
                            execution_time_ms=duration_ms,
                            error=e,
                            data={"model": model},
                        )
                        raise

                completions.Completions._enhanced_original_create = original_create  # type: ignore[attr-defined]
                completions.Completions.create = enhanced_logged_create  # type: ignore[assignment]

        except ImportError:
            pass  # OpenAI not available


# Global enhanced logger instance
_enhanced_openai_logger: Optional[EnhancedOpenAILogger] = None


def get_enhanced_openai_logger() -> EnhancedOpenAILogger:
    """Get or create the global enhanced OpenAI logger instance."""
    global _enhanced_openai_logger
    if _enhanced_openai_logger is None:
        session_id = (
            get_logger().session_id if hasattr(get_logger(), "session_id") else ""
        )
        _enhanced_openai_logger = EnhancedOpenAILogger(session_id=session_id)
    return _enhanced_openai_logger


def set_enhanced_openai_logger(logger: EnhancedOpenAILogger) -> None:
    """Set the global enhanced OpenAI logger instance."""
    global _enhanced_openai_logger
    _enhanced_openai_logger = logger
