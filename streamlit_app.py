"""Streamlit web UI for the Python interviewer — password-protected, with a per-visit interview limit.

Run locally:  streamlit run streamlit_app.py
Secrets (Streamlit Cloud → Settings → Secrets, or .streamlit/secrets.toml):
  APP_PASSWORD = "..."          # shared password visitors type
  ANTHROPIC_API_KEY = "..."
  MAX_INTERVIEWS_PER_VISIT = 3  # optional
"""
import hmac
import json
import os

import streamlit as st

from interviewer.agent import agent_step
from interviewer.llm import make_llm
from interviewer.models import MAX_QUESTIONS, MIN_QUESTIONS, Turn
from interviewer.rag import VectorStore, context_block
from interviewer.session import next_question, retrieval_query, summarize

st.set_page_config(page_title="Field Notes — AI interviewer", page_icon="📝", layout="centered")

# Streamlit secrets → environment, so the same llm.py works locally and in the cloud.
for key in ("ANTHROPIC_API_KEY", "LLM_PROVIDER", "CLAUDE_MODEL", "AWS_REGION", "VOYAGE_API_KEY"):
    try:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
    except FileNotFoundError:
        break


def secret(name: str, default=None):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except FileNotFoundError:
        return os.getenv(name, default)


MAX_PER_VISIT = int(secret("MAX_INTERVIEWS_PER_VISIT", 3))
S = st.session_state
S.setdefault("unlocked", False)
S.setdefault("used", 0)


# ---------- Password gate ----------
def gate() -> None:
    expected = secret("APP_PASSWORD")
    if not expected:
        st.error("APP_PASSWORD is not configured, so the app is locked.")
        st.stop()
    if S.unlocked:
        return
    st.title("Field Notes")
    st.caption("A short AI-led interview. Enter the access password to continue.")
    with st.form("unlock"):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Enter"):
            if hmac.compare_digest(pw.encode(), str(expected).encode()):  # timing-safe
                S.unlocked = True
                st.rerun()
            st.error("Incorrect password.")
    st.stop()


def reset_interview() -> None:
    for k in ("iv", "turns", "question", "decisions", "summary", "store"):
        S.pop(k, None)


def ask_next() -> None:
    iv, turns = S.iv, S.turns
    llm = S.llm
    store = S.get("store")
    if iv["agent"]:
        ctx = context_block(store.search(retrieval_query(iv["topic"], turns))) if store else ""
        d = agent_step(llm, iv["topic"], turns, max_q=iv["total"], extra_context=ctx)
        S.decisions.append({"action": d.action, "reason": d.reason})
        S.question = None if d.action == "finish_interview" else d.question
    else:
        S.question = None if len(turns) >= iv["total"] else next_question(llm, iv["topic"], iv["total"], turns, store)
    if S.question is None:
        S.summary = summarize(llm, iv["topic"], turns)


gate()
st.title("Field Notes")
left = MAX_PER_VISIT - S.used
st.caption(f"Claude interviews you, then summarises themes, sentiment and keywords · {left} of {MAX_PER_VISIT} interviews left this visit")

try:
    # ---------- Start ----------
    if "iv" not in S:
        if left <= 0:
            st.warning("You've used all interviews for this visit. Thanks for trying it!")
            st.stop()
        with st.form("start"):
            topic = st.text_input("Topic", "Productivity tools", max_chars=200)
            total = st.select_slider("Number of questions (maximum in agent mode)", options=list(range(MIN_QUESTIONS, MAX_QUESTIONS + 1)), value=4)
            agent = st.toggle("Agent-led: the AI decides when to dig deeper, change angle or stop")
            doc = st.file_uploader("Optional: ground the questions in a document (.txt / .md)", type=["txt", "md"])
            if st.form_submit_button("Begin interview", type="primary") and topic.strip():
                S.iv = {"topic": topic.strip(), "total": total, "agent": agent, "doc": doc.name if doc else None}
                S.turns, S.decisions, S.llm = [], [], make_llm()
                if doc:
                    S.store = VectorStore.from_text(doc.getvalue().decode("utf-8", errors="ignore")[:30000])
                S.used += 1
                with st.spinner("Thinking of the first question…"):
                    ask_next()
                st.rerun()
        st.stop()

    iv = S.iv
    mode = "Agent-led" if iv["agent"] else "Standard"
    st.subheader(iv["topic"])
    st.caption(f"{mode}{' · grounded in ' + iv['doc'] if iv['doc'] else ''}")

    for i, t in enumerate(S.turns):
        if iv["agent"] and i < len(S.decisions):
            st.caption(f"→ {S.decisions[i]['action'].replace('_', ' ')}: {S.decisions[i]['reason']}")
        with st.chat_message("assistant"):
            st.write(t.question)
        with st.chat_message("user"):
            st.write(t.answer)

    # ---------- Answering ----------
    if S.get("question"):
        n = len(S.turns) + 1
        st.progress(len(S.turns) / iv["total"], text=f"Question {n}" + ("" if iv["agent"] else f" of {iv['total']}"))
        if iv["agent"] and S.decisions:
            st.caption(f"→ {S.decisions[-1]['action'].replace('_', ' ')}: {S.decisions[-1]['reason']}")
        with st.chat_message("assistant"):
            st.write(S.question)
        answer = st.chat_input("Your answer…")
        if answer and answer.strip():
            S.turns.append(Turn(question=S.question, answer=answer.strip()[:5000]))
            with st.spinner("Thinking…"):
                ask_next()
            st.rerun()

    # ---------- Summary ----------
    elif S.get("summary"):
        s = S.summary
        st.divider()
        st.header("Summary")
        st.write(s.overview)
        c1, c2 = st.columns(2)
        c1.metric("Sentiment", s.sentiment_label, f"{s.sentiment_score:+.2f}")
        c2.metric("Claude calls", S.llm.calls)
        st.markdown("**Themes:** " + ", ".join(s.themes))
        st.markdown("**Key points**\n" + "\n".join(f"- {p}" for p in s.key_points))
        st.markdown("**Keywords:** " + " · ".join(f"`{k}`" for k in s.keywords))
        st.download_button("Download JSON", json.dumps({
            "topic": iv["topic"], "mode": mode, "document": iv["doc"], "agent_decisions": S.decisions,
            "transcript": [t.model_dump() for t in S.turns], "summary": s.model_dump(),
        }, indent=2, ensure_ascii=False), file_name="interview.json", mime="application/json")
        if st.button("New interview", type="primary"):
            reset_interview()
            st.rerun()
except RuntimeError as e:
    st.error(str(e))
    if st.button("Start over"):
        reset_interview()
        st.rerun()
