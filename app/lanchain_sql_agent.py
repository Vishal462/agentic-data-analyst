"""LangChain SQL toolkit workflow with schema-validated, generic plans."""
import json, re
from typing import Any

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage

from app.db import database_catalog, engine, query_df, quote_identifier
from app.llm import build_llm
from app.planner import AnalysisPlan
from app.semantic_validation import validate_sql_semantics

llm = build_llm()
db = SQLDatabase(engine)
tools = {tool.name: tool for tool in SQLDatabaseToolkit(db=db, llm=llm).get_tools()}
FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|COPY|ATTACH|DETACH|INSTALL|LOAD)\b", re.I)


def _sql_only(sql: str) -> bool:
    return bool(re.match(r"^\s*(SELECT|WITH)\b", sql, re.I)) and not FORBIDDEN.search(sql)


# Reasoning models (deepseek-r1) narrate inside <think> blocks that must never
# reach the SQL parser or the user.
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.I | re.S)
_THINK_OPEN = re.compile(r"<think\b[^>]*>.*", re.I | re.S)
_THINK_CLOSE = re.compile(r"</think\s*>", re.I)
_SQL_START = re.compile(r"\b(SELECT|WITH)\b", re.I)


def _message_text(value: Any) -> str:
    """Flatten a chat response to plain text, discarding reasoning content blocks."""
    content = getattr(value, "content", value)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in {"thinking", "reasoning"}:
                    continue
                parts.append(block.get("text") or "")
            else:
                parts.append(str(block))
        content = "".join(parts)
    return str(content)


def _strip_reasoning(value: Any) -> str:
    """Drop <think> scratchpads, including truncated or orphaned tags."""
    text = _THINK_BLOCK.sub(" ", _message_text(value))
    text = _THINK_OPEN.sub(" ", text)            # opener the model never closed
    return _THINK_CLOSE.split(text)[-1].strip()  # closer with no opener


def _clean_sql(value: Any) -> str:
    text = _strip_reasoning(value)
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.I | re.S)
    if match:
        text = match.group(1)
    text = text.strip().rstrip(";").strip()
    if not re.match(r"^\s*(SELECT|WITH)\b", text, re.I):
        # Trim any prose the model prepended to the statement.
        start = _SQL_START.search(text)
        text = text[start.start():].strip().rstrip(";").strip() if start else text
    return text


def _comparison_sql(plan: AnalysisPlan, question: str) -> str | None:
    """A generic SQL-first group comparison without any domain-specific names."""
    if not plan.group_by or not plan.metric: return None
    table, group, metric = map(quote_identifier, (plan.table, plan.group_by, plan.metric))
    limit = 10 if any(word in question.lower() for word in ("top", "highest", "lowest")) else 1_000_000
    sample = plan.sample_per_group or 5_000
    return f"""WITH selected_groups AS (
        SELECT {group} AS group_value, AVG({metric}) AS group_mean, COUNT({metric}) AS group_count
        FROM {table} WHERE {metric} IS NOT NULL AND {group} IS NOT NULL
        GROUP BY {group} ORDER BY group_mean DESC LIMIT {limit}
    ), bounded_sample AS (
        SELECT t.{group} AS {group}, t.{metric} AS {metric},
               ROW_NUMBER() OVER (PARTITION BY t.{group} ORDER BY hash(t.{group}, t.{metric})) AS sample_rank
        FROM {table} AS t JOIN selected_groups AS g ON t.{group} = g.group_value
        WHERE t.{metric} IS NOT NULL
    )
    SELECT {group}, {metric} FROM bounded_sample WHERE sample_rank <= {sample}"""


def _aggregate_sql(plan: AnalysisPlan) -> str | None:
    """Deterministic SQL for a fully specified simple aggregate plan."""
    if not (plan.group_by and plan.metric and plan.aggregation and plan.ranking): return None
    table, group, metric = map(quote_identifier, (plan.table, plan.group_by, plan.metric))
    aggregate = {"mean": "AVG", "sum": "SUM", "count": "COUNT", "min": "MIN", "max": "MAX"}[plan.aggregation]
    limit = f" LIMIT {plan.limit}" if plan.limit else ""
    return f"SELECT {group}, {aggregate}({metric}) AS metric_value FROM {table} WHERE {metric} IS NOT NULL GROUP BY {group} ORDER BY metric_value {plan.ranking.upper()}{limit}"


PYTHON_ROW_LIMIT = 200_000


def _projection_sql(plan: AnalysisPlan, catalog: dict[str, Any] | None) -> str | None:
    """Bounded projection of only the plan's columns, feeding Python-side statistics."""
    columns: list[str] = []
    profile = (catalog or {}).get(plan.table)
    if "correlation" in plan.operations and profile:
        # Correlation needs at least two numeric columns, which a single metric cannot supply.
        columns.extend(profile["numeric_columns"])
    for name in (plan.metric, plan.group_by, plan.date_column):
        if name and name not in columns:
            columns.append(name)
    if not columns:
        return None
    projection = ", ".join(quote_identifier(name) for name in columns)
    return f"SELECT {projection} FROM {quote_identifier(plan.table)} LIMIT {PYTHON_ROW_LIMIT}"


def discover_tables(state: dict) -> dict:
    # Toolkit remains the source for SQL-facing discovery; catalog adds typed metadata.
    return {**state, "tables": tools["sql_db_list_tables"].invoke(""), "catalog": database_catalog()}


def get_schema(state: dict) -> dict:
    return {**state, "schema": tools["sql_db_schema"].invoke(state["tables"])}


def generate_sql(state: dict) -> dict:
    plan = AnalysisPlan.model_validate(state["plan"])
    if plan.intent == "both" and "group_comparison" in plan.operations:
        sql = _comparison_sql(plan, state["question"])
        if sql: return {**state, "sql": sql}
    aggregate_sql = _aggregate_sql(plan)
    if aggregate_sql: return {**state, "sql": aggregate_sql}
    if plan.intent == "python":
        # Python-only questions still need rows; fetch just the planned columns.
        projection = _projection_sql(plan, state.get("catalog"))
        if projection: return {**state, "sql": projection}
    sampling = "For group comparison, first identify groups with a CTE, then return only group and metric columns with ROW_NUMBER() partitioned by group and a <= sample_per_group filter." if plan.intent == "both" else ""
    prompt = f"""Generate one DuckDB read-only query for this validated plan and schema.
Plan: {plan.model_dump_json()}
Schema: {state['schema']}
{sampling}
Use only plan.table and plan columns. Include no prose, no mutations, and never SELECT *.
"""
    return {**state, "sql": _clean_sql(llm.invoke([HumanMessage(content=prompt)]))}


def validate_sql(state: dict) -> dict:
    sql = state["sql"]
    if not _sql_only(sql):
        return {**state, "sql_error": "Only a single read-only SELECT or WITH query is allowed.", "retry_count": state.get("retry_count", 0) + 1}
    checked = _clean_sql(tools["sql_db_query_checker"].invoke(sql))
    sql = checked if _sql_only(checked) else sql
    issues = validate_sql_semantics(sql, AnalysisPlan.model_validate(state["plan"]))
    if issues:
        return {**state, "sql_error": "Semantic SQL mismatch: " + "; ".join(issues), "retry_count": state.get("retry_count", 0) + 1}
    return {**state, "sql": sql, "sql_error": ""}


def execute_sql(state: dict) -> dict:
    if state.get("sql_error"): return state
    try:
        frame = query_df(state["sql"])
        result: dict[str, Any] = {"columns": list(frame.columns), "row_count": len(frame), "preview": frame.head(100).to_dict("records")}
        return {**state, "analysis_data": frame, "sql_result": result, "query_result": json.dumps(result, default=str), "sql_error": ""}
    except Exception as exc:
        return {**state, "sql_error": str(exc), "retry_count": state.get("retry_count", 0) + 1}


def repair_sql(state: dict) -> dict:
    prompt = f"""Repair this DuckDB SELECT using only the validated schema. Return SQL only.
Schema: {state['schema']}
Plan: {state['plan']}
Failed SQL: {state['sql']}
Error: {state['sql_error']}"""
    return {**state, "sql": _clean_sql(llm.invoke([HumanMessage(content=prompt)]))}


def generate_answer(state: dict) -> dict:
    query_result = state.get("query_result")
    if query_result is None:
        return {
            **state,
            "answer": (
                "The SQL analysis did not produce a valid result, "
                "so I cannot provide a reliable answer."
            ),
        }
    prompt = (
        "Answer using only actual SQL results.\n"
        f"Question: {state['question']}\n"
        f"Result: {query_result}"
    )
    return {**state, "answer": _strip_reasoning(llm.invoke([HumanMessage(content=prompt)]))}


def generate_combined_answer(state: dict) -> str:
    prompt = f"""Give a concise answer using only these computed results. Do not invent values.
Question: {state['question']}
SQL: {json.dumps(state.get('sql_result', {}), default=str)}
Python: {json.dumps(state.get('python_result', {}), default=str)}
For ANOVA, say only that at least one group differs, never every pair."""
    return _strip_reasoning(llm.invoke([HumanMessage(content=prompt)]))


def generate_rag_aware_answer(state: dict) -> str:
    """Synthesize actual computation and retrieved context, keeping provenance explicit."""
    rag = state.get("rag_result", {})
    prompt = f"""Answer the question using only the supplied computed results and retrieved document context.
Question: {state['question']}
Computed SQL: {json.dumps(state.get('sql_result', {}), default=str)}
Computed Python: {json.dumps(state.get('python_result', {}), default=str)}
Retrieved context: {rag.get('context', '')}
Clearly label computed facts versus document-derived explanation. Never invent citations or numbers."""
    answer = _strip_reasoning(llm.invoke([HumanMessage(content=prompt)]))
    sources = rag.get("sources", [])
    return answer + ("\n\nSources: " + "; ".join(f"{s.get('filename')} (chunk {s.get('chunk_id')})" for s in sources) if sources else "")
