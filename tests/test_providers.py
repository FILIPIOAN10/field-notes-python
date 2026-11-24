"""Provider and persistence behaviour that must hold without a network: request shape sent to Claude,
stop-reason handling, the Lovable proxy contract, and UTF-8 saving on every OS."""
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from interviewer.__main__ import load_dotenv, save_interview
from interviewer.agent import TOOLS
from interviewer.llm import DEFAULT_MODEL, ClaudeLLM, LovableLLM


class FakeMessages:
    """Records each request and replays scripted responses."""

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def reply(stop_reason="end_turn", text=None, tool=None):
    blocks = [SimpleNamespace(type="thinking", thinking="")]
    if text is not None:
        blocks.append(SimpleNamespace(type="text", text=text))
    if tool is not None:
        blocks.append(SimpleNamespace(type="tool_use", name=tool[0], input=tool[1]))
    return SimpleNamespace(stop_reason=stop_reason, content=blocks)


def claude_with(*responses, effort="low"):
    llm = ClaudeLLM.__new__(ClaudeLLM)  # skip the real client; no network in tests
    llm.client = SimpleNamespace(messages=FakeMessages(*responses))
    llm.model, llm.effort, llm.calls = DEFAULT_MODEL, effort, 0
    return llm


def test_default_model_is_current(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    assert ClaudeLLM("anthropic").model == DEFAULT_MODEL == "claude-sonnet-5-5"


def test_complete_returns_only_text_and_sends_effort():
    llm = claude_with(reply(text="What changed for you?"))
    assert llm.complete("sys", "user", 4000) == "What changed for you?"
    assert llm.client.messages.requests[0]["output_config"] == {"effort": "low"}


def test_empty_effort_sends_no_output_config():
    llm = claude_with(reply(text="ok"), effort="")
    llm.complete("sys", "user", 100)
    assert "output_config" not in llm.client.messages.requests[0]


def test_agent_call_uses_auto_tool_choice_with_strict_tools():
    llm = claude_with(reply(stop_reason="tool_use", tool=("ask_follow_up", {"question": "Why?", "reason": "vague"})))
    assert llm.complete_tool("sys", "user", TOOLS, 4000) == ("ask_follow_up", {"question": "Why?", "reason": "vague"})
    req = llm.client.messages.requests[0]
    assert req["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert all(t["strict"] for t in req["tools"])


def test_agent_tool_schemas_are_closed_for_strict_mode():
    assert all(t["input_schema"]["additionalProperties"] is False for t in TOOLS)


def test_agent_call_retries_once_when_model_answers_in_text():
    llm = claude_with(reply(text="Let me think."), reply(stop_reason="tool_use", tool=("finish_interview", {"reason": "done"})))
    assert llm.complete_tool("sys", "user", TOOLS, 4000) == ("finish_interview", {"reason": "done"})
    assert llm.calls == 2


def test_agent_call_gives_up_after_two_text_answers():
    llm = claude_with(reply(text="a"), reply(text="b"))
    with pytest.raises(RuntimeError, match="did not choose"):
        llm.complete_tool("sys", "user", TOOLS, 4000)


@pytest.mark.parametrize("stop_reason, message", [("max_tokens", "ran out"), ("refusal", "declined")])
def test_truncated_or_refused_replies_become_clear_errors(stop_reason, message):
    with pytest.raises(RuntimeError, match=message):
        claude_with(reply(stop_reason=stop_reason)).complete("sys", "user", 100)


def test_lovable_proxy_without_text_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("LOVABLE_PROXY_TOKEN", "test-token")
    llm = LovableLLM()
    monkeypatch.setattr(llm, "_post", lambda payload: {"error": "nothing"})
    with pytest.raises(RuntimeError, match="no text"):
        llm.complete("sys", "user", 100)


def test_interview_is_saved_as_utf8(tmp_path):
    record = {"transcript": [{"question": "Ce s-a schimbat?", "answer": "Mă ajută, dar uneori greșește → ok"}]}
    path = save_interview(tmp_path / "interviews", record, datetime(2026, 11, 24, tzinfo=timezone.utc))
    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_dotenv_strips_quotes(tmp_path, monkeypatch):
    monkeypatch.setenv("FIELD_NOTES_TEST", "placeholder")  # recorded, so the variable is removed after the test
    monkeypatch.delenv("FIELD_NOTES_TEST")
    env = tmp_path / ".env"
    env.write_text('FIELD_NOTES_TEST="quoted value"\n', encoding="utf-8")
    load_dotenv(env)
    assert os.environ["FIELD_NOTES_TEST"] == "quoted value"
