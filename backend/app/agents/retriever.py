"""F3 - Retriever agent: RAG over the ingested documents."""
from app.state import AgentState
from app.vectorstore import get_vectorstore

_store = None


def _vs():
    global _store
    if _store is None:
        _store = get_vectorstore()
    return _store


def retriever_agent(state: AgentState) -> dict:
    docs = _vs().as_retriever(search_kwargs={"k": 4}).invoke(state["question"])
    return {
        "documents": state.get("documents", []) + [d.page_content for d in docs],
        "steps": state.get("steps", []) + ["retriever"],
    }
