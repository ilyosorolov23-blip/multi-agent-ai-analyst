"""
F9 - The LangGraph graph. Wires: memory -> supervisor -> {specialist} ->
supervisor -> ... -> generate -> critic -> {finish | back to supervisor}.
A step budget (see supervisor.py) plus a revision cap here guarantee the
graph always terminates even if the LLM keeps mis-routing.
"""
from langgraph.graph import END, StateGraph

from app.agents.code_agent import code_agent
from app.agents.critic import critic
from app.agents.data_sql import data_agent
from app.agents.retriever import retriever_agent
from app.agents.supervisor import generate_answer, supervisor
from app.agents.web import web_agent
from app.memory import load_memory_node, remember
from app.state import AgentState

MAX_REVISIONS = 3


def route_after_supervisor(state: AgentState) -> str:
    return state["plan"]


def route_after_critic(state: AgentState) -> str:
    if state.get("critic_ok") or state.get("revisions", 0) >= MAX_REVISIONS:
        return "finish"
    return "revise"


def remember_node(state: AgentState) -> dict:
    remember(state)
    return {}


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("memory", load_memory_node)
    g.add_node("supervisor", supervisor)
    g.add_node("retriever", retriever_agent)
    g.add_node("web", web_agent)
    g.add_node("data", data_agent)
    g.add_node("code", code_agent)
    g.add_node("generate", generate_answer)
    g.add_node("critic", critic)
    g.add_node("remember", remember_node)

    g.set_entry_point("memory")
    g.add_edge("memory", "supervisor")

    g.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "retriever": "retriever",
            "web": "web",
            "data": "data",
            "code": "code",
            "finish": "generate",
        },
    )
    for agent in ["retriever", "web", "data", "code"]:
        g.add_edge(agent, "supervisor")

    g.add_edge("generate", "critic")
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {"finish": "remember", "revise": "supervisor"},
    )
    g.add_edge("remember", END)

    return g.compile()


graph = build_graph()
