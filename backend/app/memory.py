"""F10 - Long-term memory. A separate Qdrant collection of past Q/A
turns. After each completed turn we store it; before routing, the
supervisor is given the top-k most relevant past turns so it can
answer follow-ups ('and the previous year?')."""
from app.config import settings
from app.state import AgentState
from app.vectorstore import get_vectorstore

_memory_store = None


def _store():
    global _memory_store
    if _memory_store is None:
        _memory_store = get_vectorstore(settings.memory_collection)
    return _memory_store


def recall(question: str, k: int = 3) -> list[str]:
    try:
        hits = _store().as_retriever(search_kwargs={"k": k}).invoke(question)
        return [d.page_content for d in hits]
    except Exception:
        return []  # empty memory collection on first run is fine


def remember(state: AgentState) -> None:
    text = f"Q: {state['question']}\nA: {state.get('answer', '')}"
    _store().add_texts([text])


def load_memory_node(state: AgentState) -> dict:
    """Graph entry node: pulls relevant past turns into state before routing starts."""
    return {"memory_context": recall(state["question"])}
