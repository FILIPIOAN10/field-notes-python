import pytest

from interviewer.llm import MockLLM
from interviewer.models import Turn, clamp_question_count, parse_summary, sentiment_label_from_score
from interviewer.prompts import question_prompt
from interviewer.session import next_question, summarize

GOOD = ('Sure:\n```json\n{"overview":"x","themes":["a"],"key_points":["b"],"sentiment_score":0.5,'
        '"sentiment_label":"positive","keywords":["k"]}\n```')


def test_question_count_is_clamped_to_3_5():
    assert clamp_question_count(1) == 3
    assert clamp_question_count(9) == 5
    assert clamp_question_count(float("nan")) == 4


def test_parse_summary_tolerates_fences():
    assert parse_summary(GOOD).sentiment_score == 0.5


def test_parse_summary_rejects_out_of_range_sentiment():
    with pytest.raises(ValueError):
        parse_summary(GOOD.replace("0.5", "2"))


def test_sentiment_thresholds():
    assert sentiment_label_from_score(0.3) == "positive"
    assert sentiment_label_from_score(-0.3) == "negative"
    assert sentiment_label_from_score(0.2) == "neutral"


def test_final_question_is_flagged():
    assert "(the final question)" in question_prompt("t", 3, 3, [])
    assert "(the final question)" not in question_prompt("t", 2, 3, [])


def test_full_interview_uses_questions_plus_one_calls():
    llm, turns = MockLLM(), []
    for _ in range(3):
        turns.append(Turn(question=next_question(llm, "t", 3, turns), answer="ok"))
    summarize(llm, "t", turns)
    assert llm.calls == 4


class BadThenGood(MockLLM):
    def complete(self, system, user, max_tokens):
        self.calls += 1
        return "not json" if self.calls == 1 else GOOD


def test_summary_retries_once_on_bad_json():
    assert summarize(BadThenGood(), "t", [Turn(question="q", answer="a")]).sentiment_label == "positive"
