from typing import Any, Literal, TypedDict


class AnalystState(TypedDict, total=False):
    question: str
    intent: Literal["sql", "python", "both","rag"]
    tables: str
    schema: str
    sql: str
    query_result: str
    answer: str
    sql_error: str
    retry_count: int
    # Keep computed values structured; raw samples never go to the LLM.
    sql_result: dict[str, Any]
    analysis_data: Any
    python_result: dict[str, Any]
    final_answer: str
    catalog: dict[str, Any]
    plan: dict[str, Any]
    visualization_path: str | None
    session_id: str
    rag_result: dict[str, Any]
    analysis_error: str | None
