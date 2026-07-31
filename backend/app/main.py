"""
FastAPI entrypoint. Exposes:
  POST /ask        - run the graph once, return the final state
  POST /ask/stream  - Server-Sent Events stream of each step as the graph runs
                      (this is what the Next.js frontend consumes for F13)
Langfuse (F12) wraps every graph.invoke/stream call as callbacks so a
trace of supervisor -> agent -> critic with tokens appears automatically
whenever LANGFUSE_* keys are set; otherwise tracing is silently skipped.
"""
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.graph import graph
from app.state import new_state

app = FastAPI(title="Multi-Agent AI Analyst")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _langfuse_callbacks():
    if not settings.has_langfuse:
        return []
    from langfuse.callback import CallbackHandler

    return [CallbackHandler(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )]


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "web_enabled": settings.has_web, "tracing_enabled": settings.has_langfuse}


@app.post("/ask")
def ask(req: AskRequest):
    state = new_state(req.question)
    config = {"recursion_limit": 50, "callbacks": _langfuse_callbacks()}
    result = graph.invoke(state, config=config)
    return {
        "answer": result.get("answer", ""),
        "steps": result.get("steps", []),
        "sources": result.get("documents", [])[:5],
        "sql_result": result.get("sql_result"),
        "code_result": result.get("code_result"),
        "critic_ok": result.get("critic_ok"),
        "revisions": result.get("revisions"),
    }


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """Server-Sent Events: emits one JSON event per graph step, so the
    frontend can show 'supervisor -> data -> code -> critic' live."""

    def event_gen():
        state = new_state(req.question)
        config = {"recursion_limit": 50, "callbacks": _langfuse_callbacks()}
        try:
            for chunk in graph.stream(state, config=config, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    payload = {
                        "node": node_name,
                        "steps": node_output.get("steps", []),
                        "answer": node_output.get("answer"),
                        "sql_result": node_output.get("sql_result"),
                        "code_result": node_output.get("code_result"),
                        "critic_ok": node_output.get("critic_ok"),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
