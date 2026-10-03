"""
Orchestrator — the full pipeline, built with LangGraph.

Flow: question -> router (decides sql/rag/ml) -> dispatch (calls the needed
agents) -> synthesis (combines results into one answer) -> done.

Usage:
    python backend/orchestrator.py "What is the total MRR this month?"

    or run without an argument for an interactive prompt.
"""

import sys
import os
from typing import TypedDict, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END

from backend.agents.router import route_question
from backend.agents.sql_agent import text_to_sql
from backend.agents.rag_agent import answer_question as rag_answer
from backend.agents.ml_agent import get_churn_predictions
from backend.agents.synthesis import synthesize
from backend.agents.verify import verify_answer


class PipelineState(TypedDict):
    question: str
    sources: list
    sql_result: Optional[dict]
    rag_result: Optional[dict]
    ml_result: Optional[list]
    answer: Optional[str]
    verification: Optional[dict]


def router_node(state: PipelineState) -> PipelineState:
    sources = route_question(state["question"])
    print(f"  [Router] decided sources: {sources}")
    state["sources"] = sources
    return state


def dispatch_node(state: PipelineState) -> PipelineState:
    sources = state["sources"]

    if "sql" in sources:
        print("  [Dispatch] calling SQL agent...")
        state["sql_result"] = text_to_sql(state["question"])

    if "rag" in sources:
        print("  [Dispatch] calling RAG agent...")
        state["rag_result"] = rag_answer(state["question"])

    if "ml" in sources:
        print("  [Dispatch] calling ML agent...")
        state["ml_result"] = get_churn_predictions(top_n=10)

    return state


def synthesis_node(state: PipelineState) -> PipelineState:
    print("  [Synthesis] combining results...")
    answer = synthesize(
        question=state["question"],
        sql_result=state.get("sql_result"),
        rag_result=state.get("rag_result"),
        ml_result=state.get("ml_result"),
    )
    state["answer"] = answer
    return state


def verify_node(state: PipelineState) -> PipelineState:
    print("  [Verify] checking answer against raw data...")
    verification = verify_answer(
        answer=state["answer"],
        sql_result=state.get("sql_result"),
        rag_result=state.get("rag_result"),
        ml_result=state.get("ml_result"),
    )
    state["verification"] = verification
    if verification["warnings"]:
        print(f"  [Verify] confidence={verification['confidence']}  warnings={verification['warnings']}")
    else:
        print(f"  [Verify] confidence={verification['confidence']}  (no issues found)")
    return state


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("router", router_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "dispatch")
    graph.add_edge("dispatch", "synthesis")
    graph.add_edge("synthesis", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


_app = None


def ask(question: str) -> dict:
    global _app
    if _app is None:
        _app = build_graph()

    initial_state: PipelineState = {
        "question": question,
        "sources": [],
        "sql_result": None,
        "rag_result": None,
        "ml_result": None,
        "answer": None,
        "verification": None,
    }
    final_state = _app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = ask(question)
        print(f"\nQuestion: {question}")
        print(f"\nAnswer:\n{result['answer']}")
        v = result["verification"]
        print(f"\nConfidence: {v['confidence']}")
        if v["warnings"]:
            print("Warnings:")
            for w in v["warnings"]:
                print(f"  - {w}")
    else:
        print("Enterprise Intelligence Platform — ask a question (Ctrl+C to quit)\n")
        while True:
            try:
                question = input("You: ").strip()
                if not question:
                    continue
                result = ask(question)
                print(f"\nAnswer:\n{result['answer']}")
                v = result["verification"]
                print(f"Confidence: {v['confidence']}")
                if v["warnings"]:
                    for w in v["warnings"]:
                        print(f"  Warning: {w}")
                print()
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break
