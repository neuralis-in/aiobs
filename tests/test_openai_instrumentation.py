import json
import os
from types import SimpleNamespace
import pytest

pytest.importorskip("openai")

from aiobs import observer


def test_openai_chat_completions_instrumentation(monkeypatch, tmp_path):
    # Prepare a fake create implementation to avoid network calls
    from openai.resources.chat.completions import Completions

    def fake_create(self, *args, **kwargs):  # noqa: ARG001
        message = SimpleNamespace(content="hello world")
        choice = SimpleNamespace(message=message)
        usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        return SimpleNamespace(id="fake-id", model="gpt-test", choices=[choice], usage=usage)

    # Monkeypatch BEFORE observe() so the provider wraps our fake
    monkeypatch.setattr(Completions, "create", fake_create, raising=True)

    # Start observer (installs provider instrumentation)
    observer.observe("openai-instrumentation")

    # Make a call using OpenAI client (will hit the wrapped fake)
    from openai import OpenAI

    client = OpenAI(api_key="sk-test")
    _ = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    # Flush and verify event captured
    out_path = tmp_path / "obs.json"
    observer.end()
    written = observer.flush(str(out_path))
    assert os.path.exists(written)

    data = json.loads(out_path.read_text())
    events = data.get("events", [])
    assert events, "No events captured by OpenAI instrumentation"
    ev = events[0]
    assert ev["provider"] == "openai"
    assert ev["api"] == "chat.completions.create"
    assert ev["response"]["text"] == "hello world"
    assert ev["request"]["model"] == "gpt-4o-mini"
    assert ev["span_id"] is not None  # Verify span tracking is working
    # Note: callsite may be None in test environments due to frame filtering
