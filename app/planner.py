"""Validated, schema-driven analysis planning; no dataset vocabulary is embedded."""
from typing import Literal
import json
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import re

from app.llm import build_llm

llm = build_llm()

class AnalysisPlan(BaseModel):
    intent: Literal["sql", "python", "both", "rag"]
    table: str
    metric: str | None = None
    group_by: str | None = None
    date_column: str | None = None
    operations: list[str] = Field(default_factory=list)
    visualization: Literal["bar", "line", "scatter", "histogram", "composition", "none"] = "none"
    sample_per_group: int | None = None
    use_rag: bool = False
    aggregation: Literal["mean", "sum", "count", "min", "max"] | None = None
    ranking: Literal["asc", "desc"] | None = None
    limit: int | None = None
    filters: list[dict[str, str]] = Field(default_factory=list)
    rag_query: str | None = None


def _stem(word: str) -> str:
    """Crude plural stemming so 'categories' matches a 'category' column."""
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and not word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def _tokens(value: str) -> set[str]:
    return {_stem(word) for word in re.split(r"[^a-z0-9]+", value.lower()) if word}


def _match_column(question: str, columns: list[dict], kinds: set[str] | None = None, required: bool = False) -> str | None:
    """Pick the column sharing the most word stems with the question.

    Returns None when nothing actually matches, so a plan never silently invents
    a metric or grouping the user never asked for. Callers pass required=True only
    where an operation structurally cannot run without a column.
    """
    words = _tokens(question)
    candidates = [column for column in columns if not kinds or column["kind"] in kinds]
    if not candidates:
        return None
    best = max(candidates, key=lambda column: len(words & _tokens(column["name"])))
    if words & _tokens(best["name"]):
        return best["name"]
    return candidates[0]["name"] if required else None

def has_explicit_schema_reference(question: str,catalog: dict,) -> bool:
    normalized = question.lower()
    for profile in catalog.values():
        for column in profile["columns"]:
            column_name = column["name"].lower()
            if re.search(rf"(?<![\w]){re.escape(column_name)}(?![\w])",normalized):
                return True
    return False

def propose_plan_with_llm(question: str, catalog: dict, documents_available: bool = False) -> AnalysisPlan:
    #Changing it to such that llm proposes the plan
    prompt = f"""
    You are an analytics planning assistant.

    Create a structured analysis plan for the user's question.

    IMPORTANT:
    - Use ONLY tables and columns present in the catalog.
    - If the user explicitly names a table/column, preserve that exact mapping.
    - Do not invent columns.
    - Decide whether the question needs SQL, Python, BOTH, or RAG.
    - Use RAG only when document/knowledge context is requested.
    - Prefer simple plans when possible.
    - Do not execute SQL or Python.
    - Return ONLY a JSON object matching the AnalysisPlan schema.

    CATALOG:
    {json.dumps(catalog, indent=2)}

    DOCUMENTS AVAILABLE:
    {documents_available}

    USER QUESTION:
    {question}
    """
    structured_llm = llm.with_structured_output(AnalysisPlan)
    return structured_llm.invoke([HumanMessage(content=prompt)])


def build_validated_plan(question: str,catalog: dict,documents_available: bool = False,) -> AnalysisPlan:
    # Explicit schema references are trusted and remain fast.
    if has_explicit_schema_reference(question, catalog):
        plan = build_plan(question,catalog,documents_available=documents_available)
        return validate_plan(plan, catalog)
    # Natural-language/ambiguous query → LLM proposal.
    last_error = None
    for _ in range(2):
        try:
            plan = propose_plan_with_llm(question,catalog,documents_available)
            return validate_plan(plan, catalog)
        except Exception as exc:
            last_error = exc
    # Safe deterministic fallback.
    plan = build_plan(question,catalog,documents_available=documents_available)
    return validate_plan(plan, catalog)

def build_plan(question: str, catalog: dict, documents_available: bool = False) -> AnalysisPlan:
    if len(catalog) != 1:
        raise ValueError(
            "Multiple tables are available and the analysis plan "
            "could not identify the correct table.")
    table, profile = next(iter(catalog.items()))
    text = question.lower()
    columns = profile["columns"]

    # Everything derivable from the question alone is resolved first, because it
    # determines which columns the plan structurally requires.
    operations = []
    if any(word in text for word in ("significant", "significance", "compare", "difference")):
        operations.append("group_comparison")
    if any(word in text for word in ("correlation", "correlated")):
        operations.append("correlation")
    if any(word in text for word in ("outlier", "anomaly")):
        operations.append("outliers")
    if any(word in text for word in ("trend", "over time", "monthly", "daily")):
        operations.append("trend")
    if any(word in text for word in ("distribution", "histogram")):
        operations.append("distribution")
    if any(word in text for word in ("top", "highest", "lowest", "average", "count", "total", "rank")):
        operations.append("aggregate")
    aggregation = "mean" if any(cue in text for cue in ("average", "mean")) else ("sum" if "sum" in text else ("count" if "count" in text else None))
    ranking = "desc" if any(cue in text for cue in ("highest", "top", "largest")) else ("asc" if any(cue in text for cue in ("lowest", "smallest")) else None)
    limit_match = re.search(r"\btop\s+(\d+)|\b(\d+)\s+(?:highest|lowest)", text)
    limit = int(next(value for value in limit_match.groups() if value)) if limit_match else (10 if ranking and "which" in text else None)

    # A column is only guessed at when an operation cannot run without one.
    needs_metric = bool({"group_comparison", "correlation", "outliers", "distribution", "trend"} & set(operations)) or aggregation in {"mean", "sum", "min", "max"}
    needs_group = "group_comparison" in operations
    needs_date = "trend" in operations

    # Exact schema names always win. This is intentionally before semantic
    # matching, so direct queries never get "helpfully" reinterpreted.
    explicit = {c["name"] for c in columns if c["name"].lower() in text}
    metric = next((c["name"] for c in columns if c["name"] in explicit and c["kind"] == "numeric"), None) or _match_column(question, columns, {"numeric"}, required=needs_metric)
    group = next((c["name"] for c in columns if c["name"] in explicit and c["kind"] == "categorical"), None) or _match_column(question, columns, {"categorical"}, required=needs_group)
    date = next((c["name"] for c in columns if c["name"] in explicit and c["kind"] == "date"), None) or _match_column(question, columns, {"date"}, required=needs_date)

    visualization = "none"
    if any(word in text for word in ("chart", "plot", "visual", "graph")):
        visualization = "line" if date and "trend" in operations else "scatter" if "correlation" in operations else "histogram" if "distribution" in operations else "bar"
    rag_cues = ("document", "documentation", "according to", "methodology", "definition", "assumption", "glossary")
    use_rag = documents_available and any(cue in text for cue in rag_cues)
    analytical = any(cue in text for cue in ("calculate", "average", "count", "total", "top", "highest", "lowest", "correlation", "compare", "difference", "trend", "distribution", "outlier"))
    intent = "rag" if use_rag and not analytical else ("both" if "group_comparison" in operations else ("python" if analytical and "aggregate" not in operations else "sql"))
    return AnalysisPlan(intent=intent, table=table, metric=metric, group_by=group, date_column=date,
                        operations=operations or ["aggregate"], visualization=visualization,
                        sample_per_group=5_000 if "group_comparison" in operations else None, use_rag=use_rag,
                        aggregation=aggregation, ranking=ranking, limit=limit, rag_query=question if use_rag else None)


def validate_plan(plan: AnalysisPlan, catalog: dict) -> AnalysisPlan:
    if plan.table not in catalog:
        raise ValueError(f"Unknown table in plan: {plan.table}")
    valid = {column["name"] for column in catalog[plan.table]["columns"]}
    for field in ("metric", "group_by", "date_column"):
        value = getattr(plan, field)
        if value is not None and value not in valid:
            raise ValueError(f"Unknown {field} in plan: {value}")
    numeric = {column["name"] for column in catalog[plan.table]["columns"] if column["kind"] == "numeric"}
    if plan.aggregation in {"mean", "sum", "min", "max"} and plan.metric not in numeric:
        raise ValueError(f"{plan.aggregation} requires a numeric metric")
    for item in plan.filters:
        if item.get("column") not in valid:
            raise ValueError(f"Unknown filter column: {item.get('column')}")
    return plan
