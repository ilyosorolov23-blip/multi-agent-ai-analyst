"""
F1 - The single AgentState that flows through every node in the graph.
Every agent reads from and writes to this same dict shape. LangGraph
merges partial returns from each node into this state automatically.
"""
from typing import List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    question: str            # the user's current question
    plan: str                # supervisor's chosen next node
    documents: List[str]     # chunks gathered by retriever + web agents
    sql_result: Optional[str]
    code_result: Optional[str]
    memory_context: List[str]  # relevant past turns, filled by memory (F10)
    answer: str               # the drafted / final answer
    steps: List[str]          # trace of every node visited, for the UI + Langfuse
    revisions: int            # how many times the critic sent this back
    critic_ok: bool
    critic_reason: str


def new_state(question: str) -> AgentState:
    return AgentState(
        question=question,
        plan="",
        documents=[],
        sql_result=None,
        code_result=None,
        memory_context=[],
        answer="",
        steps=[],
        revisions=0,
        critic_ok=False,
        critic_reason="",
    )
