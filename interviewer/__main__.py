"""Terminal UI:  python -m interviewer"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt

from .llm import make_llm
from .models import MAX_QUESTIONS, MIN_QUESTIONS, Turn, clamp_question_count
from .agent import agent_step
from .rag import VectorStore
from .session import next_question, summarize

console = Console()


def load_dotenv(path: Path = Path(".env")) -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def ask_answer(n: int) -> str:
    while True:
        a = Prompt.ask(f"[bold]Your answer {n}[/]").strip()
        if a:
            return a
        console.print("[yellow]Please write an answer (or press Ctrl+C to quit).[/]")


def show_summary(s) -> None:
    color = {"positive": "green", "negative": "red", "mixed": "yellow"}.get(s.sentiment_label, "white")
    body = (
        f"{s.overview}\n\n[bold]Themes:[/] {', '.join(s.themes)}\n\n[bold]Key points:[/]\n"
        + "\n".join(f"  • {p}" for p in s.key_points)
        + f"\n\n[bold]Sentiment:[/] [{color}]{s.sentiment_label} ({s.sentiment_score:+.2f})[/]"
        + f"\n[bold]Keywords:[/] {', '.join(s.keywords)}"
    )
    console.print(Panel(body, title="Summary", border_style="cyan"))


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description="Mini AI interviewer powered by Claude.")
    p.add_argument("--topic")
    p.add_argument("--questions", type=int)
    p.add_argument("--out", default="interviews", help="folder for saved JSON transcripts")
    p.add_argument("--doc", help="text/markdown file the questions should be grounded in (RAG)")
    p.add_argument("--agent", action="store_true", help="let the agent decide follow-ups and when to stop")
    args = p.parse_args()

    console.print(Panel("[bold]Field Notes[/] — a short AI-led interview", border_style="cyan"))
    topic = args.topic or Prompt.ask("Interview topic", default="Productivity tools")
    total = clamp_question_count(args.questions or IntPrompt.ask(
        f"How many questions ({MIN_QUESTIONS}-{MAX_QUESTIONS})", default=4))

    store = None
    if args.doc:
        store = VectorStore.from_text(Path(args.doc).read_text(encoding="utf-8"))
        console.print(f"[dim]Indexed {len(store.chunks)} chunks from {args.doc}[/]")

    llm = make_llm()
    turns: list[Turn] = []
    decisions: list[dict] = []
    try:
        if args.agent:
            console.print(f"[dim]Agent mode: {MIN_QUESTIONS}-{total} questions, the agent decides.[/]")
            while True:
                ctx = ""
                if store:
                    from .rag import context_block
                    from .session import retrieval_query
                    ctx = context_block(store.search(retrieval_query(topic, turns)))
                with console.status("Agent is deciding..."):
                    d = agent_step(llm, topic, turns, max_q=total, extra_context=ctx)
                decisions.append({"action": d.action, "reason": d.reason})
                console.print(f"[magenta dim]→ {d.action}: {d.reason}[/]")
                if d.action == "finish_interview":
                    break
                n = len(turns) + 1
                console.print(f"\n[cyan bold]Q{n}[/] {d.question}")
                turns.append(Turn(question=d.question, answer=ask_answer(n)))
        else:
            for n in range(1, total + 1):
                with console.status(f"Thinking of question {n}/{total}..."):
                    q = next_question(llm, topic, total, turns, store)
                console.print(f"\n[cyan bold]Q{n}/{total}[/] {q}")
                turns.append(Turn(question=q, answer=ask_answer(n)))
        with console.status("Analysing your answers..."):
            summary = summarize(llm, topic, turns)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interview cancelled — nothing saved.[/]")
        return
    except RuntimeError as e:
        console.print(f"\n[red]Error:[/] {e}")
        return

    console.print()
    show_summary(summary)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc)
    path = out / f"interview-{stamp:%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({
        "topic": topic, "mode": "agent" if args.agent else "fixed", "document": args.doc,
        "agent_decisions": decisions, "created_at": stamp.isoformat(), "claude_calls": llm.calls,
        "transcript": [t.model_dump() for t in turns], "summary": summary.model_dump(),
    }, indent=2, ensure_ascii=False))
    console.print(f"[dim]Saved to {path} · {llm.calls} Claude calls[/]")


if __name__ == "__main__":
    main()
