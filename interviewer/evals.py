"""Evaluation harness: run the real summary prompt on labelled transcripts and score the output.

Usage:  python -m interviewer.evals [--cases evals/cases.json] [--out evals/report.json]
"""
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .llm import make_llm
from .models import Summary, Turn
from .session import summarize


@dataclass
class CaseResult:
    id: str
    passed: bool
    score: float           # 0..1, average of the checks below
    label_ok: bool         # sentiment_label in the accepted labels
    range_ok: bool         # sentiment_score inside the expected interval
    keyword_recall: float  # share of expected keywords found in keywords/themes
    got_label: str = ""
    got_score: float = 0.0
    error: str = ""


def score_case(case: dict, s: Summary) -> CaseResult:
    exp = case["expect"]
    label_ok = s.sentiment_label in exp["labels"]
    lo, hi = exp["score_range"]
    range_ok = lo <= s.sentiment_score <= hi
    haystack = " ".join(s.keywords + s.themes + s.key_points).lower()
    kws = exp.get("keywords", [])
    recall = sum(any(alt in haystack for alt in k.split("|")) for k in kws) / len(kws) if kws else 1.0
    score = (label_ok + range_ok + recall) / 3
    return CaseResult(case["id"], label_ok and range_ok and recall >= 0.5, round(score, 3),
                      label_ok, range_ok, round(recall, 3), s.sentiment_label, s.sentiment_score)


def run(cases: list[dict], llm=None) -> list[CaseResult]:
    llm = llm or make_llm()
    results = []
    for c in cases:
        turns = [Turn(**t) for t in c["transcript"]]
        try:
            results.append(score_case(c, summarize(llm, c["topic"], turns)))
        except RuntimeError as e:  # a broken output counts as a failed case, not a crash
            results.append(CaseResult(c["id"], False, 0.0, False, False, 0.0, error=str(e)))
    return results


def main() -> None:
    from rich.console import Console
    from rich.table import Table

    from .__main__ import load_dotenv

    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="evals/cases.json")
    p.add_argument("--out", default="evals/report.json")
    a = p.parse_args()
    results = run(json.loads(Path(a.cases).read_text(encoding="utf-8")))

    t = Table(title="Summary prompt evaluation")
    for col in ("case", "label", "score", "label ok", "range ok", "keyword recall", "pass"):
        t.add_column(col)
    for r in results:
        t.add_row(r.id, r.got_label or "—", f"{r.got_score:+.2f}", "✓" if r.label_ok else "✗",
                  "✓" if r.range_ok else "✗", f"{r.keyword_recall:.0%}", "[green]PASS[/]" if r.passed else f"[red]FAIL[/] {r.error}")
    c = Console()
    c.print(t)
    passed = sum(r.passed for r in results)
    avg = sum(r.score for r in results) / len(results)
    c.print(f"[bold]{passed}/{len(results)} passed · average score {avg:.0%}[/]")
    Path(a.out).write_text(json.dumps({"passed": passed, "total": len(results), "average": avg,
                                       "results": [asdict(r) for r in results]}, indent=2), encoding="utf-8")
    c.print(f"[dim]Report saved to {a.out}[/]")


if __name__ == "__main__":
    main()
