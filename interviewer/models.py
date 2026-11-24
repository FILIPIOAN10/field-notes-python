"""Data models and pure analysis helpers (no I/O, fully unit-tested)."""
import json
import math
from typing import Literal

from pydantic import BaseModel, Field

MIN_QUESTIONS, MAX_QUESTIONS = 3, 5


class Turn(BaseModel):
    question: str
    answer: str


class Summary(BaseModel):
    overview: str
    themes: list[str] = Field(max_length=6)
    key_points: list[str] = Field(max_length=8)
    sentiment_score: float = Field(ge=-1, le=1)
    sentiment_label: Literal["positive", "neutral", "negative", "mixed"]
    keywords: list[str] = Field(max_length=12)


def clamp_question_count(n: float) -> int:
    if not isinstance(n, (int, float)) or not math.isfinite(n):
        return 4
    return min(MAX_QUESTIONS, max(MIN_QUESTIONS, round(n)))


def parse_summary(text: str) -> Summary:
    """Extract and validate the summary JSON (tolerates code fences / extra prose)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("The summary was not in the expected format.")
    return Summary.model_validate(json.loads(text[start : end + 1]))


def sentiment_label_from_score(score: float) -> str:
    if score > 0.2:
        return "positive"
    if score < -0.2:
        return "negative"
    return "neutral"


def clean_question(raw: str) -> str:
    return raw.strip().strip("\"'").strip()
