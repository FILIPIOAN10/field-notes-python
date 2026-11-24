"""Agentic interviewer: Claude chooses a tool each turn — dig deeper, change subtopic, or finish.

The code enforces the guardrails (min/max questions), the model makes the judgement calls.
"""
from dataclasses import dataclass

from .models import MAX_QUESTIONS, MIN_QUESTIONS, Turn, clean_question
from .prompts import format_transcript

AGENT_SYSTEM = """You are an expert qualitative interviewer running an adaptive interview.
Each turn you MUST call exactly one tool:
- ask_follow_up: the last answer was vague, surprising or emotional — dig deeper into it.
- ask_new_subtopic: the last answer was rich enough — move to an uncovered angle of the topic.
- finish_interview: you have enough insight (only allowed once the minimum is reached).
Questions: one open-ended, non-leading question under 35 words, never repeating an earlier one.
Always give a short reason so your decision is auditable."""

TOOLS = [
    {
        "name": "ask_follow_up",
        "description": "Ask a deeper follow-up about the participant's last answer.",
        "input_schema": {"type": "object", "properties": {
            "question": {"type": "string"}, "reason": {"type": "string"}}, "required": ["question", "reason"]},
    },
    {
        "name": "ask_new_subtopic",
        "description": "Move on to a new, not-yet-covered angle of the topic.",
        "input_schema": {"type": "object", "properties": {
            "question": {"type": "string"}, "subtopic": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["question", "subtopic", "reason"]},
    },
    {
        "name": "finish_interview",
        "description": "End the interview because there is enough insight.",
        "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
    },
]


@dataclass
class Decision:
    action: str  # ask_follow_up | ask_new_subtopic | finish_interview
    question: str
    reason: str


def agent_step(llm, topic: str, turns: list[Turn], min_q: int = MIN_QUESTIONS, max_q: int = MAX_QUESTIONS,
               extra_context: str = "") -> Decision:
    asked = len(turns)
    if asked >= max_q:  # hard guardrail, no model call needed
        return Decision("finish_interview", "", f"Reached the maximum of {max_q} questions.")
    tools = TOOLS if asked >= min_q else TOOLS[:2]  # cannot finish before the minimum
    user = (
        f'Topic: "{topic}"\nQuestions asked: {asked} (min {min_q}, max {max_q}).\n\n'
        f"Conversation so far:\n{format_transcript(turns) if turns else '(none yet — open the interview)'}"
        + (f"\n\nReference material:\n{extra_context}" if extra_context else "")
        + "\n\nChoose your next action."
    )
    name, args = llm.complete_tool(AGENT_SYSTEM, user, tools, 2000)
    if name not in {t["name"] for t in tools}:
        raise RuntimeError(f"The agent chose an unavailable action: {name}")
    if name == "finish_interview":
        return Decision(name, "", args.get("reason", ""))
    q = clean_question(args.get("question", ""))
    if not q:
        raise RuntimeError("The agent returned an empty question.")
    return Decision(name, q, args.get("reason", ""))
