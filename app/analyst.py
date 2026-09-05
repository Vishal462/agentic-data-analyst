from pathlib import Path
from langgraph.graph import END, StateGraph
from app.analyst_state import AnalystState
from app.lanchain_sql_agent import (discover_tables,execute_sql,generate_answer,
        generate_combined_answer,generate_rag_aware_answer,generate_sql,get_schema,
        repair_sql,validate_sql)
from app.planner import AnalysisPlan, build_validated_plan
from app.python_analysis import run_python_analysis
from app.rag.retriever import retrieve
from app.rag.vector_store import get_index
from app.visualization import render_chart

MAX_SQL_RETRIES = 2 #Can be changed, deterministic pipeline not used though

def plan_analysis(state: AnalystState) -> AnalystState:
    session_id = state.get("session_id", "default")
    _, collection = get_index(session_id)
    plan = build_validated_plan(state["question"], state["catalog"], documents_available=collection.count() > 0)
    return {**state, "intent": plan.intent, "plan": plan.model_dump(), "session_id": session_id}

def run_python(state: AnalystState) -> AnalystState:
    plan = AnalysisPlan.model_validate(state["plan"])
    return {**state, "python_result": run_python_analysis(state["analysis_data"], plan)}

def run_rag(state: AnalystState) -> AnalystState:
    return {**state, "rag_result": retrieve(state["question"], state.get("session_id", "default"))}

def create_visualization(state: AnalystState) -> AnalystState:
    plan = AnalysisPlan.model_validate(state["plan"])
    path = render_chart(state["analysis_data"], plan, Path(__file__).resolve().parents[1] / "artifacts")
    return {**state, "visualization_path": str(path) if path else None}

def analysis_error(state: AnalystState) -> AnalystState:
    reason = state.get("sql_error") or "The analysis could not be completed."
    return {
        **state,
        "analysis_error": reason,
        "final_answer": (
            "I could not complete the analysis reliably after "
            f"the maximum number of retries.\n\nReason: {reason}"
        )}

def final_answer(state: AnalystState) -> AnalystState:
    if state.get("rag_result"):
        return {**state, "final_answer": generate_rag_aware_answer(state)}
    if state["intent"] == "sql":
        return {**state, "final_answer": state.get("answer", "")}
    return {**state, "final_answer": generate_combined_answer(state)}

def _wants_chart(state: AnalystState) -> bool:
    return state["plan"]["visualization"] != "none"

def after_plan(state: AnalystState) -> str:
    # Only a pure document question skips the database; python/both still need rows.
    return "run_rag" if state["intent"] == "rag" else "generate_sql"

def after_execute(state: AnalystState) -> str:
    if state.get("sql_error"):
        return "repair_sql" if state.get("retry_count", 0) < MAX_SQL_RETRIES else "analysis_error"
    if state["intent"] == "sql":
        # Narrate the result first; charting is a decoration on top of that answer.
        return "run_rag" if state["plan"].get("use_rag") else "generate_answer"
    return "run_python"

def after_generate_answer(state: AnalystState) -> str:
    return "visualize" if _wants_chart(state) else "final_answer"

def after_python(state: AnalystState) -> str:
    if state["plan"].get("use_rag"):
        return "run_rag"
    return "visualize" if _wants_chart(state) else "final_answer"

def after_rag(state: AnalystState) -> str:
    return "visualize" if state.get("analysis_data") is not None and _wants_chart(state) else "final_answer"

graph = StateGraph(AnalystState)
for name, node in {
    "discover_tables": discover_tables,
    "get_schema": get_schema,
    "plan_analysis": plan_analysis,
    "generate_sql": generate_sql,
    "validate_sql": validate_sql,
    "execute_sql": execute_sql,
    "repair_sql": repair_sql,
    "generate_answer": generate_answer,
    "run_python": run_python,
    "run_rag": run_rag,
    "visualize": create_visualization,
    "analysis_error": analysis_error,
    "final_answer": final_answer,}.items():
    graph.add_node(name, node)

graph.set_entry_point("discover_tables")
graph.add_edge("discover_tables", "get_schema")
graph.add_edge("get_schema", "plan_analysis")
graph.add_conditional_edges("plan_analysis", after_plan, {"generate_sql": "generate_sql", "run_rag": "run_rag"})
graph.add_edge("generate_sql", "validate_sql")
graph.add_edge("validate_sql", "execute_sql")
graph.add_conditional_edges(
    "execute_sql",
    after_execute,
    {
        "repair_sql": "repair_sql",
        "analysis_error": "analysis_error",
        "run_python": "run_python",
        "run_rag": "run_rag",
        "generate_answer": "generate_answer",
    },
)
graph.add_edge("repair_sql", "validate_sql")
graph.add_conditional_edges("generate_answer", after_generate_answer, {"visualize": "visualize", "final_answer": "final_answer"})
graph.add_conditional_edges("run_python", after_python, {"run_rag": "run_rag", "visualize": "visualize", "final_answer": "final_answer"})
graph.add_conditional_edges("run_rag", after_rag, {"visualize": "visualize", "final_answer": "final_answer"})
graph.add_edge("visualize", "final_answer")
graph.add_edge("analysis_error", END)
graph.add_edge("final_answer", END)
analyst_graph = graph.compile()


if __name__ == "__main__":
    session_id = input("Document session [default]: ").strip() or "default"
    result = analyst_graph.invoke({"question": input("Ask a question: "), "session_id": session_id})
    print(result.get("final_answer"))
    if result.get("visualization_path"):
        print(f"Chart: {result['visualization_path']}")
