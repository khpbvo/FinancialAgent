import asyncio
from types import SimpleNamespace

import pytest


# Helpers to simulate streaming events without network
class FakeEvent:
    def __init__(self, type_, data=None, item=None, new_agent=None):
        self.type = type_
        self.data = data
        self.item = item
        self.new_agent = new_agent


class FakeRunResultStreaming:
    def __init__(self, events):
        self._events = events

    async def stream_events(self):
        for ev in self._events:
            await asyncio.sleep(0)  # yield control to simulate async
            yield ev


@pytest.mark.asyncio
async def test_streaming_mode_streams_text(monkeypatch, capsys):
    """Ensure streaming_mode prints deltas when ResponseTextDeltaEvent arrives."""
    from financial_agent import cli as cli_mod
    from financial_agent.cli import streaming_mode

    # Prepare fake streaming events: a few text deltas
    # Patch the symbol used in the CLI to a minimal dummy class
    class DummyTextDelta:
        def __init__(self, delta: str):
            self.delta = delta

    monkeypatch.setattr(cli_mod, "ResponseTextDeltaEvent", DummyTextDelta, raising=False)

    ev1 = FakeEvent("raw_response_event", data=DummyTextDelta("Hello "))
    ev2 = FakeEvent("raw_response_event", data=DummyTextDelta("world"))

    fake_stream = FakeRunResultStreaming([ev1, ev2])

    # Patch Runner.run_streamed to return our fake streaming result
    import agents

    def fake_run_streamed(agent, input, context=None, session=None):  # noqa: A002
        return fake_stream

    monkeypatch.setattr(agents.Runner, "run_streamed", staticmethod(fake_run_streamed))

    # Run streaming_mode with no session and dummy agent/deps
    await streaming_mode(agent=None, deps=None, user_input="Test", use_session=False)

    out = capsys.readouterr().out
    assert "📝 Response:" in out or "📝 Final Response:" in out
    assert "Hello " in out and "world" in out


@pytest.mark.asyncio
async def test_streaming_mode_fallback_on_no_events(monkeypatch, capsys):
    """If no text deltas arrive, ensure fallback sync run prints final output."""
    from financial_agent.cli import streaming_mode

    # Empty stream (no events)
    fake_stream = FakeRunResultStreaming([])

    import agents

    def fake_run_streamed(agent, input, context=None, session=None):  # noqa: A002
        return fake_stream

    class FakeSyncResult:
        final_output = "OK-FALLBACK"

    async def fake_run(agent, input, context=None, session=None):  # noqa: A002
        return FakeSyncResult()

    monkeypatch.setattr(agents.Runner, "run_streamed", staticmethod(fake_run_streamed))
    monkeypatch.setattr(agents.Runner, "run", staticmethod(fake_run))

    await streaming_mode(agent=None, deps=None, user_input="Test", use_session=False)

    out = capsys.readouterr().out
    assert (
        "Using fallback sync run" in out
        or "Response (fallback)" in out
        or "📝 Response: OK-FALLBACK" in out
    )


@pytest.mark.asyncio
async def test_streaming_mode_handles_message_output_item(monkeypatch, capsys):
    """Ensure we print message_output_item when no raw deltas are present."""
    from financial_agent.cli import streaming_mode

    # Build a run_item_stream_event with message_output_item and simple text attribute
    fake_item = SimpleNamespace(type="message_output_item", text="Hello from item")
    ev = FakeEvent("run_item_stream_event", item=fake_item)
    fake_stream = FakeRunResultStreaming([ev])

    import agents

    def fake_run_streamed(agent, input, context=None, session=None):  # noqa: A002
        return fake_stream

    monkeypatch.setattr(agents.Runner, "run_streamed", staticmethod(fake_run_streamed))

    await streaming_mode(agent=None, deps=None, user_input="Test", use_session=False)

    out = capsys.readouterr().out
    assert "📝 Response: Hello from item" in out
