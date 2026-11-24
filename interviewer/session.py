"""Interview orchestration: adaptive questions (optionally grounded in a document), then summarise."""
from .llm import LLM
from .models import Summary, Turn, clean_question, parse_summary
from .prompts import INTERVIEWER_SYSTEM, SUMMARY_SYSTEM, question_prompt, rag_question_prompt, summary_prompt
from .rag import VectorStore, context_block


def retrieval_query(topic: str, turns: list[Turn]) -> str:
    """Search with the topic plus the latest answer, so retrieval follows the conversation."""
    return f"{topic} {turns[-1].question} {turns[-1].answer}" if turns else topic


def next_question(llm: LLM, topic: str, total: int, turns: list[Turn], store: VectorStore | None = None) -> str:
    n = len(turns) + 1
    if store:
        ctx = context_block(store.search(retrieval_query(topic, turns), k=3))
        prompt = rag_question_prompt(topic, n, total, turns, ctx)
    else:
        prompt = question_prompt(topic, n, total, turns)
    q = clean_question(llm.complete(INTERVIEWER_SYSTEM, prompt, 4000))
    if not q:
        raise RuntimeError("The interviewer returned an empty question.")
    return q


def summarize(llm: LLM, topic: str, turns: list[Turn], retries: int = 1) -> Summary:
    """Ask for the JSON summary; retry once if the model breaks the format."""
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            return parse_summary(llm.complete(SUMMARY_SYSTEM, summary_prompt(topic, turns), 8000))
        except ValueError as e:  # pydantic ValidationError and JSONDecodeError are ValueErrors
            last = e
    raise RuntimeError(f"Could not get a valid summary: {last}")
