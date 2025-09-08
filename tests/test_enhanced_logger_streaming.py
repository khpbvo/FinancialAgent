"""Tests to ensure the enhanced logger doesn't consume streaming bodies."""


class FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code

    @property
    def content(self):  # pragma: no cover
        # If the logger ever tries to access content, raise to catch it
        raise AssertionError(
            "content should not be accessed for streaming OpenAI calls"
        )


def test_http_logger_does_not_consume_streaming_body(monkeypatch):
    """Ensure EnhancedOpenAILogger never reads httpx response.content for OpenAI calls."""
    import httpx
    from financial_agent.enhanced_openai_logger import EnhancedOpenAILogger

    # Base stub that simulates the underlying transport used by logged_request
    def base_stub(self, method, url, **kwargs):  # noqa: ANN001
        # Simulate a successful HTTP response without usable content
        return FakeResponse(status_code=200)

    # Replace the current httpx.Client.request BEFORE logger patches capture it
    monkeypatch.setattr(httpx.Client, "request", base_stub, raising=False)

    # Initialize logger (applies patches, capturing our base_stub as original)
    EnhancedOpenAILogger(session_id="test")

    # Make a request that looks like an OpenAI streaming call
    client = httpx.Client()
    # Include Accept header for SSE to mimic streaming
    headers = {"Accept": "text/event-stream"}

    # Must not raise due to touching response.content
    client.request(
        "POST", "https://api.openai.com/v1/responses", headers=headers, content=b"{}"
    )

    # Also verify a traces ingest endpoint doesn't access content
    client.request(
        "POST",
        "https://api.openai.com/v1/traces/ingest",
        headers=headers,
        content=b"{}",
    )
