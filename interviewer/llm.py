"""Single place that talks to Claude. Provider is chosen by LLM_PROVIDER."""
import os
from typing import Protocol


class LLM(Protocol):
    calls: int

    def complete(self, system: str, user: str, max_tokens: int) -> str: ...

    def complete_tool(self, system: str, user: str, tools: list[dict], max_tokens: int) -> tuple[str, dict]: ...


DEFAULT_MODEL = "claude-sonnet-5-5"


class ClaudeLLM:
    def __init__(self, provider: str):
        import anthropic

        if provider == "bedrock":
            # Bedrock's Messages API endpoint ("Mantle"); Bedrock model ids carry an "anthropic." prefix.
            self.client = anthropic.AnthropicBedrockMantle(aws_region=os.getenv("AWS_REGION", "eu-central-1"))
            default_model = f"anthropic.{DEFAULT_MODEL}"
        else:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example).")
            self.client = anthropic.Anthropic()
            default_model = DEFAULT_MODEL
        self.model = os.getenv("CLAUDE_MODEL", default_model)
        # Interview turns are short and conversational: low effort keeps each question fast and cheap.
        # Set CLAUDE_EFFORT to medium/high to trade speed for depth, or to "" for models without effort.
        self.effort = os.getenv("CLAUDE_EFFORT", "low")
        self.calls = 0

    def _create(self, **kwargs):
        import anthropic

        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        try:
            msg = self.client.messages.create(model=self.model, **kwargs)
        except anthropic.RateLimitError as e:
            raise RuntimeError("Rate limited by the AI provider — wait a moment and retry.") from e
        except anthropic.APIError as e:
            raise RuntimeError(f"AI request failed: {e}") from e
        self.calls += 1
        if msg.stop_reason == "refusal":
            raise RuntimeError("The model declined this request.")
        if msg.stop_reason == "max_tokens":
            raise RuntimeError("The model ran out of output tokens before finishing — please retry.")
        return msg

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        msg = self._create(max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if b.type == "text")

    def complete_tool(self, system: str, user: str, tools: list[dict], max_tokens: int) -> tuple[str, dict]:
        """Ask for exactly one tool call and return (tool_name, arguments).

        Current Claude models reject a forced tool_choice ("any" / "tool"), so the request uses "auto"
        with strict schemas and at most one call; the system prompt requires a tool, and if the model
        answers in plain text instead, the request is retried once before giving up.
        """
        strict_tools = [{**t, "strict": True} for t in tools]
        for _ in range(2):
            msg = self._create(max_tokens=max_tokens, system=system, tools=strict_tools,
                               tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                               messages=[{"role": "user", "content": user}])
            for b in msg.content:
                if b.type == "tool_use":
                    return b.name, dict(b.input)
        raise RuntimeError("The agent did not choose an action.")


class LovableLLM:
    """Calls Claude through the Lovable web app proxy — needs only a shared token, no Claude key."""

    def __init__(self):
        self.url = os.getenv("LOVABLE_PROXY_URL", "https://fable-powered-plan-studio.lovable.app/api/public/llm")
        self.token = os.getenv("LOVABLE_PROXY_TOKEN")
        if not self.token:
            raise RuntimeError("LOVABLE_PROXY_TOKEN is not set (see .env.example).")
        self.calls = 0

    def _post(self, payload: dict) -> dict:
        import json
        import urllib.error
        import urllib.request

        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", "")
            except Exception:
                msg = ""
            raise RuntimeError(f"AI request failed ({e.code}): {msg or e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach the AI proxy: {e.reason}") from e
        self.calls += 1
        return data

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        text = self._post({"system": system, "user": user, "max_tokens": max_tokens}).get("text")
        if not isinstance(text, str):
            raise RuntimeError("The AI proxy returned no text.")
        return text

    def complete_tool(self, system: str, user: str, tools: list[dict], max_tokens: int) -> tuple[str, dict]:
        data = self._post({"system": system, "user": user + "\n\nYou must respond by calling exactly one tool.",
                           "tools": tools, "max_tokens": max_tokens})
        tool = data.get("tool")
        if not tool:
            raise RuntimeError("The agent did not choose an action.")
        return tool["name"], dict(tool.get("input") or {})


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
    if provider == "mock":
        return MockLLM()
    if provider == "lovable":
        return LovableLLM()
    return ClaudeLLM(provider)
