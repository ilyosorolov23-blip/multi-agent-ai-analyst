"""F4 - Web agent: Tavily search for questions outside the local docs.
Skips gracefully (returns no new documents, logs the skip) if no Tavily
key is configured, so the whole system still works with zero optional keys.
"""
from app.config import settings
from app.state import AgentState


def web_agent(state: AgentState) -> dict:
    if not settings.has_web:
        return {"steps": state.get("steps", []) + ["web(skipped: no TAVILY_API_KEY)"]}

    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    hits = client.search(state["question"], max_results=5).get("results", [])
    new_docs = [f"[web] {h.get('title', '')}: {h.get('content', '')}" for h in hits]

    return {
        "documents": state.get("documents", []) + new_docs,
        "steps": state.get("steps", []) + ["web"],
    }
