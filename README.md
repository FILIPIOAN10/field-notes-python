# Field Notes — AI interviewer (Python)

A small but complete AI-powered interviewer for the terminal. Claude runs a short interview on any topic,
adapts each question to the previous answers, then produces a validated summary with
**sentiment scoring** and **keyword extraction**. On top of the core flow it has:

| Capability | What it shows | Command |
|---|---|---|
| Adaptive interview (3–5 questions) | prompt design, structured output | `python -m interviewer` |
| **Agent mode** — Claude picks a tool each turn: dig deeper, change subtopic, or finish | agentic workflow with tool use + code-enforced guardrails | `python -m interviewer --agent` |
| **RAG** — questions grounded in a document via vector search | chunking, embeddings, retrieval | `python -m interviewer --doc examples/product_brief.txt` |
| **Evaluation harness** — labelled transcripts, automatic scoring | LLM evaluation | `python -m interviewer.evals` |
| **Web demo (Streamlit)** — password gate + per-visit interview limit, all modes above | simple UI layer for demos | `streamlit run streamlit_app.py` |

A companion web app (TanStack Start + Lovable Cloud) uses the exact same prompts:
https://fable-powered-plan-studio.lovable.app

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # add ANTHROPIC_API_KEY
python -m interviewer --topic "Remote work" --questions 4
```
- Default model: `claude-sonnet-5-5` at `low` effort (short conversational turns); override with
  `CLAUDE_MODEL` / `CLAUDE_EFFORT`.
- No key? `LLM_PROVIDER=mock` runs every mode offline with a deterministic fake model.
- `LLM_PROVIDER=lovable` + `LOVABLE_PROXY_TOKEN` sends the same prompts through the companion web app's
  server-side proxy, so no model key is needed on this side.
- AWS Bedrock: `LLM_PROVIDER=bedrock` + `AWS_REGION` (Bedrock Messages API client; model id
  `anthropic.claude-sonnet-5-5`). Not tested against a real AWS account.
- Tests (27, no network): `pytest -q`
- Each interview is saved as UTF-8 JSON to `interviews/*.json` (transcript, summary, agent decisions, model call count).

## Live demo (Streamlit Community Cloud)
1. Push this folder to a GitHub repository.
2. On https://share.streamlit.io → **Create app** → pick the repo, branch `main`, main file `streamlit_app.py`.
3. **Advanced settings → Secrets** — either through the web app's proxy (no model key in Streamlit):
   ```toml
   APP_PASSWORD = "choose-a-password"
   LLM_PROVIDER = "lovable"
   LOVABLE_PROXY_TOKEN = "the web app's PYTHON_PROXY_TOKEN"
   MAX_INTERVIEWS_PER_VISIT = 3
   ```
   or directly with an Anthropic key: replace the two provider lines with `ANTHROPIC_API_KEY = "sk-ant-..."`.
4. Deploy. Visitors must enter the password; each browser visit can run at most `MAX_INTERVIEWS_PER_VISIT`
   interviews, which caps API spend. The password is compared with `hmac.compare_digest` (timing-safe) and
   never shown in the UI. Locally, put the same keys in `.streamlit/secrets.toml` (git-ignored).

## Architecture
```text
__main__.py       terminal UI (rich), argument parsing, saving JSON
streamlit_app.py  web UI: password gate, per-visit limit, same session/agent/rag modules
   │
   ├── session.py   fixed-length flow: next_question() / summarize()   ─┐
   ├── agent.py     agent_step(): tool choice + guardrails              ├─► llm.py  (only module that calls a model:
   ├── rag.py       chunk → embed → VectorStore.search()                │     Anthropic | Lovable proxy | Bedrock | Mock)
   └── evals.py     run cases → score_case() → report                  ─┘
prompts.py   every prompt in one place          models.py   Pydantic models + pure helpers
```
Design rules: the LLM sits behind a 2-method `LLM` protocol, so providers are swappable and every
piece of logic is testable with the mock; pure functions (parsing, scoring, chunking) have no I/O.

## Prompt design
- **Interviewer** — one open, non-leading question under 35 words, no preamble. The full conversation is
  sent each turn so questions build on what was said; the last question is flagged so it turns reflective.
- **Summary** — JSON-only contract, validated by Pydantic (score in −1..1, fixed label set, length caps).
  The parser tolerates code fences; on invalid output it retries once, then fails with a clear error.
- **RAG** — the top-3 chunks are retrieved with *topic + latest answer* as the query (retrieval follows the
  conversation), wrapped in `<document>` tags and marked as background facts only, to resist prompt injection.
- **Agent** — three tools (`ask_follow_up`, `ask_new_subtopic`, `finish_interview`), each requiring a
  `reason`, so every decision is auditable and saved. The *model* judges; the *code* enforces limits:
  `finish_interview` isn't even offered before the minimum, and at the maximum no model call is made.
  Current Claude models don't accept a *forced* tool choice, so the request uses `tool_choice: auto` with
  strict schemas and one call at most; if the model answers in plain text instead, the turn is retried once.
- **Failures are explicit** — a refusal or a reply cut off at `max_tokens` becomes a clear error instead of
  an empty question.
- **Cost is predictable** — fixed mode uses exactly N + 1 model calls; agent mode at most max + 1
  (plus a retry in the rare case above).

## Evaluation
`evals/cases.json` holds 6 labelled transcripts (positive, negative, mixed, neutral/factual, sarcasm,
very short answers). Each case defines accepted labels, a sentiment-score interval and expected keywords
(`a|b` = synonyms). Score = mean of label match, range match and keyword recall; pass needs all three
(recall ≥ 50%). Malformed model output counts as a failure, not a crash. Report → `evals/report.json`.

The last recorded run, on Claude Sonnet 4.5 before the move to `claude-sonnet-5-5`, passed **6/6**,
including the sarcasm case (scored −0.6, negative). Re-run `python -m interviewer.evals` after any model or
prompt change, and add a case whenever a regression is spotted — that is the point of the harness.

## Data & security
- API keys only from environment / `.env` (git-ignored); nothing is hard-coded or logged.
- Interviews stay on the local machine as JSON; nothing is sent anywhere except the model request
  (Anthropic API, Bedrock, or the web app's proxy, depending on `LLM_PROVIDER`).
- The proxy accepts only requests carrying its bearer token; the token lives in Streamlit secrets / `.env`, never in code.
- Document excerpts are treated as untrusted data in the prompt (see RAG above).
- Default embeddings are local (hashed bag-of-words, numpy) so documents never leave the machine;
  set `VOYAGE_API_KEY` for semantic embeddings. The in-memory store can be swapped for pgvector/OpenSearch.

## How AI tools were used to build this
Built with AI-assisted development (Lovable's agent for the web app, then this Python port).
The workflow: I wrote the requirements and constraints, the assistant drafted code, and every change was
checked with unit tests, an offline mock run and a live evaluation run before being accepted.
The evaluation harness itself was the main tool for judging prompt quality, rather than eyeballing outputs.

## Limitations / next steps
- Hashing embeddings match words, not meaning — use Voyage or Bedrock Titan embeddings for real documents.
- Agent quality is measured only informally; next step is an agent eval (e.g. does it follow up on vague answers?).
- The per-visit limit lives in the browser session; a determined visitor can reset it by reopening the
  page, so the password is the real gate. For public use, add per-user accounts or a server-side quota.
- A FastAPI layer would let other services call the interviewer as an API.
