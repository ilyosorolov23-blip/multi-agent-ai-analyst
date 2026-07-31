"""
F7 - Supervisor / router. An LLM with structured output decides which
specialist runs next, or that enough evidence is gathered ('finish').
Also drafts the final answer once evidence collection is done.
"""
from typing import Literal

from pydantic import BaseModel, Field

from app.config import settings
from app.llm import get_chat_llm
from app.state import AgentState

_llm = get_chat_llm()


class Route(BaseModel):
    next: Literal["retriever", "web", "data", "code", "finish"] = Field(
        description="Which specialist should run next, or 'finish' if enough evidence is gathered."
    )
    reasoning: str = Field(description="One short sentence explaining the choice.")


def supervisor(state: AgentState) -> dict:
    steps = state.get("steps", [])
    step_limit = settings.supervisor_step_limit
    if len([s for s in steps if not s.startswith("supervisor")]) >= step_limit:
        # step budget guard — always terminates even if routing loops
        return {"plan": "finish", "steps": steps + ["supervisor→finish(step limit)"]}

    memory_ctx = "\n".join(state.get("memory_context", []))
    prompt = (
        f"Question: {state['question']}\n"
        f"Relevant past turns: {memory_ctx or 'none'}\n"
        f"Evidence collected so far: {steps}\n"
        f"Documents: {len(state.get('documents', []))} chunks | "
        f"SQL result: {'yes' if state.get('sql_result') else 'no'} | "
        f"Code result: {'yes' if state.get('code_result') else 'no'}\n\n"
        "Decide the single next specialist to call (retriever = search local docs, "
        "web = search the internet, data = query the SQL database, code = run Python "
        "for a calculation), or 'finish' once you have what's needed to answer."
    )
    decision = _llm.with_structured_output(Route).invoke(prompt)
    return {
        "plan": decision.next,
        "steps": steps + [f"supervisor→{decision.next}"],
    }


def generate_answer(state: AgentState) -> dict:
    """Drafts the answer from all gathered evidence. Runs when plan == 'finish'."""
    evidence = (
        f"Document excerpts:\n{chr(10).join(state.get('documents', [])[:6]) or 'none'}\n\n"
        f"SQL result:\n{state.get('sql_result') or 'none'}\n\n"
        f"Code result:\n{state.get('code_result') or 'none'}\n\n"
        f"Past relevant turns:\n{chr(10).join(state.get('memory_context', [])) or 'none'}"
    )
    prompt = (
        f"Question: {state['question']}\n\nEvidence:\n{evidence}\n\n"
        "Write a clear, grounded answer using only this evidence. If evidence is "
        "insufficient, say so explicitly rather than guessing."
    )
    answer = _llm.invoke(prompt).content
    return {"answer": answer, "steps": state.get("steps", []) + ["generate"]}
