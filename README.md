# Multi-Agent AI Analyst

A supervisor-led team of AI agents (retriever, web, SQL, code) with a critic that
verifies answers, long-term memory, an evaluation harness, and Langfuse tracing —
built to run at **$0, no credit card**, deployable on **Render** (backend) +
**Vercel** (frontend).

```
multi-agent-analyst/
├── backend/     FastAPI + LangGraph multi-agent system
└── frontend/    Next.js UI that streams the live agent trace
```

## How this maps to the project guide's features

| # | Feature | File |
|---|---|---|
| F1 | Shared state & config | `backend/app/state.py`, `backend/app/config.py` |
| F2 | Ingestion & vector store | `backend/app/vectorstore.py`, `backend/ingest.py` |
| F3 | Retriever agent | `backend/app/agents/retriever.py` |
| F4 | Web agent | `backend/app/agents/web.py` |
| F5 | Data (SQL) agent | `backend/app/agents/data_sql.py` |
| F6 | Code agent | `backend/app/agents/code_agent.py` |
| F7 | Supervisor / router | `backend/app/agents/supervisor.py` |
| F8 | Critic / verifier | `backend/app/agents/critic.py` |
| F9 | Graph wiring | `backend/app/graph.py` |
| F10 | Long-term memory | `backend/app/memory.py` |
| F11 | Evaluation harness | `backend/app/eval/harness.py` |
| F12 | Langfuse tracing | wired into `backend/app/main.py` |
| F13 | Streaming frontend | `frontend/app/page.tsx`, `/ask/stream` endpoint |
| F14 | Deployment | `backend/render.yaml`, `frontend/vercel.json` |

## 1. Get your own free keys (no card needed anywhere)

| Service | Required? | Get it at |
|---|---|---|
| Google Gemini | **Required** | https://aistudio.google.com/apikey |
| Qdrant | Optional (embedded mode needs no signup) | https://cloud.qdrant.io |
| Tavily (web search) | Optional | https://tavily.com |
| Langfuse (tracing) | Optional | https://cloud.langfuse.com |

Never commit your `.env` — it's already git-ignored.

## 2. Run locally

```bash
# backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your GOOGLE_API_KEY in
python ingest.py              # populates the vector store from data/docs
uvicorn app.main:app --reload --port 8000

# frontend, in a second terminal
cd frontend
npm install
cp .env.local.example .env.local   # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Open http://localhost:3000 and ask something like *"How many customers churned,
and what reasons are given in the docs?"* — you'll watch it route through
supervisor → data → retriever → critic live.

## 3. Run the evaluation harness (F11)

```bash
cd backend
python -m app.eval.harness
```

Prints RAGAS metrics (faithfulness, answer relevancy, context precision) and an
LLM-judge score, comparing the system **with vs without** the critic, and writes
per-question CSVs to `app/eval/results/`.

## 4. Deploy for free

**Backend → Render** (you already have an account):
1. Push this repo to GitHub.
2. On Render: New → Blueprint → point at the repo (it reads `backend/render.yaml`).
3. Add your `GOOGLE_API_KEY` (and any optional keys) in the Render dashboard's
   environment variables — `render.yaml` declares them as `sync: false` so
   Render prompts you for the values instead of storing them in the repo.
4. Render free tier is 512 MB RAM — this app uses Gemini for both the LLM
   *and* embeddings (no local embedding model), so it fits comfortably.

**Frontend → Vercel** (you already have an account):
1. Import the repo, set the project root to `frontend/`.
2. Add env var `NEXT_PUBLIC_API_URL` = your Render backend URL.
3. Deploy. Done — you have a public link.

**Even easier alternative (no server at all):** run everything in a Google
Colab notebook and call `demo.launch(share=True)` for a Gradio UI — free,
no card, public link for ~72h, and Colab's ~12 GB RAM lets you run local
embeddings too if you want to skip Gemini embeddings.

## 5. Submission checklist (from the rubric)

- [x] Own free API keys, not shared
- [x] Supervisor graph + 4 specialist agents + critic (`app/graph.py`)
- [x] Read-only text-to-SQL guard (`app/agents/data_sql.py` — rejects non-SELECT)
- [x] Code agent sandboxed with a subprocess timeout (`app/agents/code_agent.py`)
- [x] Long-term memory recalls earlier turns (`app/memory.py`)
- [x] RAGAS + LLM-judge eval harness on 10 questions (`app/eval/harness.py`)
- [x] Langfuse tracing wired (set the 3 LANGFUSE_* env vars to activate)
- [ ] Backend + frontend deployed — **you do this step** (accounts are ready)
- [ ] README error-analysis section — fill in after you run the eval harness
      and look at 3 wrong answers (template below)

### Error analysis template (fill in after running the harness)

| # | Question | What went wrong | Which agent failed | Fix |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

## Notes on the sample data

`backend/data/docs/*.txt` — three short docs (product overview, policies,
onboarding) so the retriever has something real to search.
`backend/data/company.db` — a small SQLite DB (12 customers, ~41 orders) so
the SQL agent has something real to query. Swap in your own docs/DB any time;
re-run `python ingest.py` after changing the docs.
