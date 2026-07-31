"""
F11 - Evaluation harness. Runs the graph over a fixed test set (>=10
questions), scores it with RAGAS metrics (faithfulness, answer
relevancy, context precision) plus an LLM-judge 1-5 score vs a
reference, and prints a comparison table with the critic ON vs OFF
(critic OFF = we take the first drafted answer, before verification).

Run: python -m app.eval.harness
"""
import json
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from pydantic import BaseModel, Field

from app.graph import graph
from app.llm import get_chat_llm
from app.state import new_state

_llm = get_chat_llm()

TESTSET_PATH = Path(__file__).parent / "testset.json"


class JudgeScore(BaseModel):
    score: int = Field(description="1-5, how well the answer matches the reference and question intent")
    justification: str


def llm_judge(question: str, answer: str, reference: str) -> int:
    prompt = (
        f"Question: {question}\nReference (loose guide, not exact wording): {reference}\n"
        f"Candidate answer: {answer}\n\nScore the candidate 1 (bad) to 5 (excellent) for "
        "correctness and relevance to the question."
    )
    return _llm.with_structured_output(JudgeScore).invoke(prompt).score


def run_once(question: str, disable_critic: bool = False) -> dict:
    """Runs the graph. If disable_critic, we intercept at the first 'generate'
    step by running the graph but capping revisions to 0, i.e. accepting the
    first drafted answer regardless of critic verdict."""
    state = new_state(question)
    config = {"recursion_limit": 50}
    if disable_critic:
        # Monkey-patch route_after_critic behaviour by capping MAX_REVISIONS via state
        state["revisions"] = 999  # forces route_after_critic -> "finish" on first pass
    result = graph.invoke(state, config=config)
    return result


def run_harness():
    testset = json.loads(TESTSET_PATH.read_text())
    if len(testset) < 10:
        raise ValueError("Test set must have at least 10 questions (F11 requirement).")

    rows_with_critic, rows_without_critic = [], []

    for item in testset:
        q, ref = item["question"], item["reference"]

        with_c = run_once(q, disable_critic=False)
        without_c = run_once(q, disable_critic=True)

        contexts = with_c.get("documents", []) or ["(no retrieved context)"]

        rows_with_critic.append({
            "question": q, "answer": with_c.get("answer", ""), "contexts": contexts,
            "reference": ref, "judge": llm_judge(q, with_c.get("answer", ""), ref),
        })
        rows_without_critic.append({
            "question": q, "answer": without_c.get("answer", ""), "contexts": contexts,
            "reference": ref, "judge": llm_judge(q, without_c.get("answer", ""), ref),
        })

    def score_with_ragas(rows):
        try:
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, faithfulness

            ds = Dataset.from_list([
                {"question": r["question"], "answer": r["answer"],
                 "contexts": r["contexts"], "reference": r["reference"]}
                for r in rows
            ])
            result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
            return result.to_pandas()
        except Exception as e:  # RAGAS needs an LLM+embeddings config of its own
            print(f"[warn] RAGAS scoring skipped: {e}")
            return None

    df_with = score_with_ragas(rows_with_critic)
    df_without = score_with_ragas(rows_without_critic)

    judge_with = sum(r["judge"] for r in rows_with_critic) / len(rows_with_critic)
    judge_without = sum(r["judge"] for r in rows_without_critic) / len(rows_without_critic)

    print("\n=== LLM-judge average score (1-5) ===")
    print(f"With critic:    {judge_with:.2f}")
    print(f"Without critic: {judge_without:.2f}")

    if df_with is not None:
        print("\n=== RAGAS metrics — WITH critic ===")
        print(df_with[["faithfulness", "answer_relevancy", "context_precision"]].mean())
    if df_without is not None:
        print("\n=== RAGAS metrics — WITHOUT critic ===")
        print(df_without[["faithfulness", "answer_relevancy", "context_precision"]].mean())

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(rows_with_critic).to_csv(out_dir / "with_critic.csv", index=False)
    pd.DataFrame(rows_without_critic).to_csv(out_dir / "without_critic.csv", index=False)
    print(f"\nDetailed rows written to {out_dir}/")


if __name__ == "__main__":
    run_harness()
