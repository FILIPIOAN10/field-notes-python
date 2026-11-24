from interviewer.agent import agent_step
from interviewer.evals import run, score_case
from interviewer.llm import MockLLM
from interviewer.models import Summary, Turn
from interviewer.rag import VectorStore, chunk_text
from interviewer.session import next_question

DOC = ("Pricing: the Teams plan costs 6 EUR per user per month. " * 3
       + "Onboarding explains the VIP contacts list so urgent messages still arrive. " * 3)


def test_chunks_overlap_and_cover_all_words():
    chunks = chunk_text(" ".join(f"w{i}" for i in range(200)), max_words=80, overlap=20)
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]
    assert "w199" in chunks[-1]


def test_retrieval_returns_the_relevant_chunk():
    store = VectorStore(["The Teams plan costs 6 EUR per user per month.", "Onboarding explains the VIP list."])
    assert "EUR" in store.search("how much does the teams plan cost", k=1)[0][0]


class Capture(MockLLM):
    def complete(self, system, user, max_tokens):
        self.last = user
        return super().complete(system, user, max_tokens)


def test_rag_prompt_includes_retrieved_excerpts():
    llm = Capture()
    next_question(llm, "pricing", 3, [], VectorStore.from_text(DOC))
    assert "<document>" in llm.last and "6 EUR" in llm.last


def test_agent_cannot_finish_before_minimum():
    turns = [Turn(question="q", answer="a")]
    assert agent_step(MockLLM(), "t", turns, min_q=3, max_q=5).action != "finish_interview"


def test_agent_stops_at_maximum_without_calling_claude():
    llm = MockLLM()
    d = agent_step(llm, "t", [Turn(question="q", answer="a")] * 5, min_q=3, max_q=5)
    assert d.action == "finish_interview" and llm.calls == 0


def summary(label, score, kws):
    return Summary(overview="o", themes=[], key_points=[], sentiment_score=score, sentiment_label=label, keywords=kws)


CASE = {"id": "neg", "expect": {"labels": ["negative"], "score_range": [-1, -0.3], "keywords": ["stress|frustrat", "tool"]}}


def test_eval_passes_matching_summary():
    r = score_case(CASE, summary("negative", -0.7, ["frustration", "tools"]))
    assert r.passed and r.score == 1.0


def test_eval_fails_wrong_sentiment():
    r = score_case(CASE, summary("positive", 0.6, ["frustration", "tools"]))
    assert not r.passed and not r.label_ok and not r.range_ok


def test_eval_counts_broken_output_as_failure():
    class Broken(MockLLM):
        def complete(self, *a):
            return "no json"
    case = {**CASE, "topic": "t", "transcript": [{"question": "q", "answer": "a"}]}
    assert run([case], Broken())[0].passed is False
