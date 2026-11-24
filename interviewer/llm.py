"""Single place that talks to Claude. Provider is chosen by LLM_PROVIDER."""
import os
from typing import Protocol


class LLM(Protocol):
    calls: int

    def complete(self, system: str, user: str, max_tokens: int) -> str: ...

    def complete_tool(self, system: str, user: str, tools: list[dict], max_tokens: int) -> tuple[str, dict]: ...


class ClaudeLLM:
    def __init__(self, provider: str):
        import anthropic

        if provider == "bedrock":
            self.client = anthropic.AnthropicBedrock(aws_region=os.getenv("AWS_REGION", "eu-central-1"))
        else:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example).")
            self.client = anthropic.Anthropic()
        self.model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
        self.calls = 0

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        import anthropic

        try:
            msg = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.RateLimitError as e:
            raise RuntimeError("Rate limited by the AI provider — wait a moment and retry.") from e
        except anthropic.APIError as e:
            raise RuntimeError(f"AI request failed: {e}") from e
        self.calls += 1
        return "".join(b.text for b in msg.content if b.type == "text")

    def complete_tool(self, system: str, user: str, tools: list[dict], max_tokens: int) -> tuple[str, dict]:
        """Force exactly one tool call and return (tool_name, arguments)."""
        import anthropic

        try:
            msg = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system, tools=tools,
                tool_choice={"type": "any"}, messages=[{"role": "user", "content": user}],
            )
        except anthropic.APIError as e:
            raise RuntimeError(f"AI request failed: {e}") from e
        self.calls += 1
        for b in msg.content:
            if b.type == "tool_use":
                return b.name, dict(b.input)
        raise RuntimeError("The agent did not choose an action.")


class MockLLM:
    """Offline stand-in for demos and tests — no API key needed."""

    def __init__(self):
        self.calls = 0

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.calls += 1
        if system.startswith("You analyse"):
            return ('{"overview":"The participant shared mixed views.","themes":["focus","tools"],'
                    '"key_points":["Uses several tools","Struggles with distractions","Wants simpler workflows"],'
                    '"sentiment_score":0.1,"sentiment_label":"mixed","keywords":["focus","tools","workflow","distraction","habits"]}')
        n = user.split("Write question ")[-1].rstrip(".")
        return f"What stands out to you most about this topic so far? (mock question {n})"

    def complete_tool(self, system: str, user: str, tools: list[dict], max_tokens: int) -> tuple[str, dict]:
        self.calls += 1
        names = [t["name"] for t in tools]
        asked = int(user.split("Questions asked: ")[1].split(" ")[0])
        if "finish_interview" in names:
            return "finish_interview", {"reason": "Mock: minimum reached."}
        if asked % 2:
            return "ask_follow_up", {"question": f"Can you give a concrete example? (mock {asked + 1})", "reason": "Mock: dig deeper."}
        return "ask_new_subtopic", {"question": f"How does this affect your team? (mock {asked + 1})",
                                    "subtopic": "team", "reason": "Mock: new angle."}


def make_llm() -> LLM:
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    return MockLLM() if provider == "mock" else ClaudeLLM(provider)
