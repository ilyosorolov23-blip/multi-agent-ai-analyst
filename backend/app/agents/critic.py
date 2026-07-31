"""F8 - Critic / verifier. Checks the drafted answer against gathered
evidence; approves it or forces a revision (routed back to supervisor)."""
from pydantic import BaseModel, Field

from app.llm import get_chat_llm
from app.state import AgentState

_llm = get_chat_llm()


class Verdict(BaseModel):
    ok: bool = Field(description="True only if the answer is correct AND fully supported by the evidence.")
    reason: str = Field(description="One short sentence explaining the verdict.")


def critic(state: AgentState) -> dict:
    evidence = (
        f"Documents: {state.get('documents', [])}\n"
        f"SQL: {state.get('sql_result')}\nCode: {state.get('code_result')}"
    )
    prompt = (
        f"Question: {state['question']}\n"
        f"Evidence: {evidence}\n"
        f"Drafted answer: {state.get('answer')}\n\n"
        "Is this answer correct AND fully grounded in the evidence above? "
        "Reject vague, unsupported, or fabricated claims."
    )
    verdict = _llm.with_structured_output(Verdict).invoke(prompt)
    revisions = state.get("revisions", 0) + (0 if verdict.ok else 1)
    return {
        "critic_ok": verdict.ok,
        "critic_reason": verdict.reason,
        "revisions": revisions,
        "steps": state.get("steps", []) + [f"critic({'ok' if verdict.ok else 'revise'})"],
    }
