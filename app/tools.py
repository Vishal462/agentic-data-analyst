"""Tools the analyst agent chooses between.

Every tool returns a compact JSON string: the model must never receive whole
result sets, only enough to decide the next step. Large intermediates stay in
process and are referred to by a `result_id` handle, so a 50k-row frame can be
charted or tested without ever passing through the context window.
"""
import contextvars
import json
import re
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.tools import tool
from sqlalchemy.exc import SQLAlchemyError

from app.db import _column_kind, dataset_profile, discover_tables, query_df, quote_identifier
from app.python_analysis import _anova
from app.rag.retriever import retrieve

MAX_ROWS = 200_000
# A preview shows shape, never content: raw rows in the context slow every later
# call and tempt the model to narrate the sample instead of analysing the result.
# Ranked group summaries come from compare_groups, which aggregates in the database.
PREVIEW_ROWS = 3
CELL_LIMIT = 60
MAX_GROUPS = 50
AGGREGATIONS = {"mean": "AVG", "sum": "SUM", "count": "COUNT", "min": "MIN", "max": "MAX"}
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

# Session for document retrieval, set by the caller rather than guessed by the model.
current_session: contextvars.ContextVar[str] = contextvars.ContextVar("current_session", default="default")

_RESULTS: dict[str, pd.DataFrame] = {}
_QUERIES: dict[str, str] = {}


def reset_results() -> None:
    _RESULTS.clear()
    _QUERIES.clear()


def _sampled(result_id: str) -> str | None:
    """Reject a statistical test over a LIMITed query.

    A test on a slice silently changes the conclusion rather than failing, so this
    is enforced by the tool rather than left to the prompt: an ANOVA over 1,000 of
    50,000 rows reported "not significant" for data that is significant at p<1e-46.
    """
    query = _QUERIES.get(result_id, "")
    if re.search(r"\bLIMIT\s+\d+", query, re.I):
        return (f"Result {result_id} came from a query with a LIMIT, so it is only a slice of the data. "
                f"A statistical test on a slice gives a different answer than the full data and would be "
                f"misleading. Re-run the same query without the LIMIT and use the new result_id.")
    return None


def _dump(payload: dict) -> str:
    return json.dumps(payload, default=str)


def _frame(result_id: str) -> pd.DataFrame | None:
    return _RESULTS.get(result_id)


def _missing(result_id: str) -> str:
    return _dump({"error": f"No result named {result_id}. Known results: {sorted(_RESULTS) or 'none yet'}. Run run_sql first."})


def _clip(value) -> str:
    text = str(value)
    return text if len(text) <= CELL_LIMIT else text[:CELL_LIMIT] + "..."


def _preview(frame: pd.DataFrame) -> list[dict]:
    return [{key: _clip(value) for key, value in row.items()}
            for row in frame.head(PREVIEW_ROWS).to_dict("records")]


def _table_columns(table: str) -> dict[str, str]:
    """Column name -> declared type, without the per-column profiling dataset_profile does."""
    description = query_df(f"DESCRIBE {quote_identifier(table)}")
    return {str(row[0]): str(row[1]) for row in description.itertuples(index=False)}


def _first_values(table: str, columns: list[str], rows: int = 5) -> dict[str, str]:
    """One representative value per column, from a single query.

    Names alone cannot separate columns that differ only by content - ORIGIN holds
    airport codes and ORIGIN_CITY holds city names, and picking the wrong one
    returns a confident zero rather than an error. An example makes them distinct.
    """
    try:
        sample = query_df(f"SELECT * FROM {quote_identifier(table)} LIMIT {int(rows)}")
    except Exception:
        return {}
    examples = {}
    for name in columns:
        if name not in sample.columns:
            continue
        values = sample[name].dropna()
        if not values.empty:
            examples[name] = _clip(values.iloc[0])
    return examples


def schema_summary(max_columns: int = 200) -> str:
    """Compact exact column names, kinds, and one example value each, for the prompt.

    Naming the columns up front removes a describe_table round trip on every
    question, and stops the model inventing lowercase names it then has to repair.
    """
    lines, budget = [], max_columns
    for table in discover_tables():
        try:
            columns = _table_columns(table)
        except Exception:
            continue
        shown = list(columns)[:budget]
        budget -= len(shown)
        examples = _first_values(table, shown)
        rendered = []
        for name in shown:
            entry = f"{name} [{_column_kind(columns[name])}]"
            if name in examples:
                entry += f' e.g. "{examples[name]}"'
            rendered.append(entry)
        lines.append(f"  {table}: " + ", ".join(rendered) + (" ..." if len(shown) < len(columns) else ""))
        if budget <= 0:
            break
    return "\n".join(lines) or "  (no tables)"


@tool
def list_tables() -> str:
    """List every table available in the database. Call this first when you do not know the schema."""
    return _dump({"tables": discover_tables()})


@tool
def describe_table(table: str) -> str:
    """Describe one table: its columns, their types, how many values are missing, and example values.

    Always call this before writing SQL, so you use real column names rather than guessing.
    """
    try:
        profile = dataset_profile(table)
    except ValueError as exc:
        return _dump({"error": str(exc), "tables": discover_tables()})
    columns = []
    for column in profile["columns"]:
        entry = {"name": column["name"], "type": column["data_type"], "kind": column["kind"]}
        sample = column["sample_values"][:1]
        if sample:
            entry["example"] = _clip(sample[0])
        if column["missing_count"]:
            entry["missing"] = column["missing_count"]
        columns.append(entry)
    return _dump({"table": profile["table"], "row_count": profile["row_count"], "columns": columns})


@tool
def run_sql(query: str) -> str:
    """Run one read-only DuckDB SELECT query and return a preview of the rows.

    The database is opened read-only, so anything that would modify it fails.
    The full result is kept server-side under the returned `result_id`; pass that
    id to the statistics and charting tools instead of copying rows around.
    Name the columns you need explicitly; SELECT * is rejected.
    If the query errors, the error text is returned so you can correct the SQL and retry.
    """
    if re.search(r"SELECT\s+\*|,\s*\*", query, re.I):
        return _dump({"error": "SELECT * is not allowed. Name the columns you need explicitly, so the result "
                               "stays small enough to analyse and has no duplicate column names."})
    try:
        frame = query_df(query)
    except SQLAlchemyError as exc:
        return _dump({"error": str(getattr(exc, "orig", exc)).strip()[:600], "hint": "Fix the SQL and call run_sql again."})
    except Exception as exc:
        return _dump({"error": f"{type(exc).__name__}: {exc}"[:600]})
    if len(frame) > MAX_ROWS:
        frame = frame.head(MAX_ROWS)
    result_id = f"r{uuid.uuid4().hex[:8]}"
    _RESULTS[result_id] = frame
    _QUERIES[result_id] = query
    preview = _preview(frame)
    payload = {
        "status": "ok",
        "result_id": result_id,
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "preview": preview,
        "truncated": bool(len(frame) == MAX_ROWS),
    }
    if len(preview) < len(frame):
        payload["note"] = (
            f"Large result: {len(frame)} rows. Only a preview is shown. Do not answer from these rows - "
            f"they are not the full result. For a ranked group summary use compare_groups; for other "
            f"analysis pass result_id '{result_id}' to correlate, find_outliers, describe_distribution, "
            f"or make_chart.")
    return _dump(payload)


@tool
def compare_groups(table: str, group_by: str, metric: str, aggregation: str = "mean",
                   ranking: str = "desc", limit: int | None = None, statistical_test: str = "anova") -> str:
    """Summarise a numeric metric across groups, and optionally test whether they really differ.

    This is the right tool both for "show <metric> by <group>" and for "which
    <groups> have the highest/lowest <metric>", with or without "is the difference
    significant". It aggregates inside the database and returns a compact summary
    plus the test result, so raw observations never enter the conversation. Prefer
    this over run_sql for any per-group summary - run_sql returns only a 3-row preview.

    table/group_by/metric must be real column names from the schema. If the user asks
    to group by, or measure, something that has no column in this table, do NOT pick a
    similar-sounding column - say the data is not available instead.
    aggregation: mean, sum, count, min, or max. Use sum for "total".
    ranking: desc for highest first, asc for lowest first.
    limit: how many groups to return. OMIT IT to return every group, which is what
        "show/display <metric> by <group>" asks for. Pass a number only when the user
        asked for a specific count or a shortlist ("top 10", "the 3 worst").
    statistical_test: "anova" for a one-way ANOVA across all groups, or "none".
    """
    aggregation, ranking = aggregation.lower().strip(), ranking.lower().strip()
    if aggregation not in AGGREGATIONS:
        return _dump({"error": f"Unsupported aggregation {aggregation!r}. Use one of {sorted(AGGREGATIONS)}."})
    if ranking not in {"asc", "desc"}:
        return _dump({"error": f"Unsupported ranking {ranking!r}. Use 'desc' or 'asc'."})
    if table not in discover_tables():
        return _dump({"error": f"Unknown table {table!r}.", "tables": discover_tables()})
    columns = _table_columns(table)
    unknown = [name for name in (group_by, metric) if name not in columns]
    if unknown:
        return _dump({"error": f"Unknown column(s) {unknown} in {table}.", "columns": sorted(columns)})
    if aggregation != "count" and _column_kind(columns[metric]) != "numeric":
        return _dump({"error": f"{metric} is {columns[metric]}, not numeric, so {aggregation} cannot be computed."})
    # No limit means every group; MAX_GROUPS still caps the payload for very high
    # cardinality groupings, and the caller is told when that cap bites.
    requested_all = limit is None
    limit = MAX_GROUPS if requested_all else max(1, min(int(limit), MAX_GROUPS))

    table_sql, group_sql, metric_sql = (quote_identifier(name) for name in (table, group_by, metric))
    try:
        ranked = query_df(
            f"SELECT {group_sql} AS group_value, {AGGREGATIONS[aggregation]}({metric_sql}) AS value, "
            f"COUNT({metric_sql}) AS observations FROM {table_sql} "
            f"WHERE {metric_sql} IS NOT NULL AND {group_sql} IS NOT NULL "
            f"GROUP BY {group_sql} ORDER BY value {ranking.upper()} LIMIT {limit}")
        total_groups = int(query_df(
            f"SELECT COUNT(DISTINCT {group_sql}) AS n FROM {table_sql} WHERE {group_sql} IS NOT NULL").iloc[0, 0])
    except Exception as exc:
        return _dump({"error": f"Aggregation failed: {type(exc).__name__}: {exc}"[:400]})

    groups = [{"rank": position, "group": str(row.group_value),
               aggregation: round(float(row.value), 4), "count": int(row.observations)}
              for position, row in enumerate(ranked.itertuples(index=False), start=1)]
    result = {"status": "ok", "table": table, "group_by": group_by, "metric": metric,
              "aggregation": aggregation, "ranking": ranking,
              "groups_returned": len(groups), "groups_total": total_groups, "groups": groups}
    if requested_all and len(groups) < total_groups:
        result["note"] = (f"{total_groups} groups exist; the {len(groups)} shown are capped at {MAX_GROUPS} "
                          f"to keep the result small. Say so if you list them.")
    elif len(groups) < total_groups:
        result["note"] = f"These are the {len(groups)} requested of {total_groups} groups, ranked {ranking}."

    if statistical_test.lower().strip() == "anova" and total_groups >= 2:
        # Tested across every group, not just the ranked slice. Testing only the top N
        # answers a different question: the top 5 airlines have similar means (p=0.60)
        # while all 18 differ strongly (p=5e-15), and the model reports whichever it is
        # given as a general claim about the groups.
        raw = query_df(f"SELECT {group_sql} AS group_value, {metric_sql} AS metric_value FROM {table_sql} "
                       f"WHERE {metric_sql} IS NOT NULL AND {group_sql} IS NOT NULL")
        try:
            test = _anova(raw, "group_value", "metric_value")
        except Exception as exc:
            test = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
        if test.get("status") == "ok":
            result["statistical_test"] = {
                "name": test["test"],
                "scope": f"all {total_groups} groups in {table}, not only the {len(groups)} returned above",
                "f_statistic": round(test["f_statistic"], 4),
                "p_value": test["p_value"], "alpha": test["alpha"], "significant": test["significant"],
                "eta_squared": round(test["eta_squared"], 4) if test.get("eta_squared") is not None else None,
                "observations": int(sum(test["sample_counts"].values())),
                "interpretation": test["conclusion"], "caution": test["practical_note"]}
        else:
            result["statistical_test"] = test
    return _dump(result)


@tool
def correlate(result_id: str) -> str:
    """Compute a Pearson correlation matrix across every numeric column in a result set."""
    frame = _frame(result_id)
    if frame is None:
        return _missing(result_id)
    sampled = _sampled(result_id)
    if sampled:
        return _dump({"error": sampled})
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return _dump({"status": "unavailable", "reason": "Two numeric columns are required.", "columns": list(frame.columns)})
    return _dump({"status": "ok", "correlation_matrix": numeric.corr().round(4).to_dict()})


@tool
def find_outliers(result_id: str, column: str) -> str:
    """Count outliers in a numeric column using the 1.5 x IQR rule, and report the bounds."""
    frame = _frame(result_id)
    if frame is None:
        return _missing(result_id)
    if column not in frame.columns:
        return _dump({"error": f"Result {result_id} has columns {list(frame.columns)}."})
    sampled = _sampled(result_id)
    if sampled:
        return _dump({"error": sampled})
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return _dump({"status": "unavailable", "reason": f"{column} holds no numeric values."})
    q1, q3 = values.quantile([.25, .75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return _dump({"status": "ok", "column": column, "lower_bound": float(low), "upper_bound": float(high),
                  "outlier_count": int(((values < low) | (values > high)).sum()), "total": int(len(values))})


@tool
def describe_distribution(result_id: str, column: str) -> str:
    """Summary statistics (count, mean, std, quartiles) for one numeric column."""
    frame = _frame(result_id)
    if frame is None:
        return _missing(result_id)
    if column not in frame.columns:
        return _dump({"error": f"Result {result_id} has columns {list(frame.columns)}."})
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return _dump({"status": "unavailable", "reason": f"{column} holds no numeric values."})
    return _dump({"status": "ok", "column": column, "statistics": values.describe().to_dict()})


@tool
def make_chart(result_id: str, kind: str, x_column: str, y_column: str | None = None) -> str:
    """Render a chart from a result set and save it as a PNG.

    `kind` is one of: bar, line, scatter, histogram. `y_column` is required for
    every kind except histogram. Returns the path of the saved image.
    """
    frame = _frame(result_id)
    if frame is None:
        return _missing(result_id)
    kind = kind.lower().strip()
    if kind not in {"bar", "line", "scatter", "histogram"}:
        return _dump({"error": f"Unsupported chart kind {kind!r}. Use bar, line, scatter, or histogram."})
    needed = [x_column] + ([y_column] if y_column else [])
    if not set(needed).issubset(frame.columns):
        return _dump({"error": f"Result {result_id} has columns {list(frame.columns)}."})
    if kind != "histogram" and not y_column:
        return _dump({"error": f"{kind} charts need a y_column."})
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    try:
        if kind == "histogram":
            pd.to_numeric(frame[x_column], errors="coerce").dropna().plot(ax=axis, kind="hist", bins=30)
            axis.set_xlabel(x_column)
        elif kind == "scatter":
            axis.scatter(frame[x_column], frame[y_column], alpha=.45)
            axis.set(xlabel=x_column, ylabel=y_column)
        else:
            series = frame.groupby(x_column, dropna=True)[y_column].mean()
            series = series.sort_index() if kind == "line" else series.sort_values(ascending=False).head(25)
            series.plot(ax=axis, kind=kind)
            axis.set(xlabel=x_column, ylabel=y_column)
        axis.set_title(f"{y_column or x_column} by {x_column}" if kind != "histogram" else f"Distribution of {x_column}")
        figure.tight_layout()
        target = ARTIFACTS / f"{result_id}_{kind}.png"
        figure.savefig(target, dpi=150)
    except Exception as exc:
        return _dump({"error": f"Could not render chart: {type(exc).__name__}: {exc}"})
    finally:
        plt.close(figure)
    return _dump({"status": "ok", "chart_path": str(target)})


@tool
def search_documents(query: str) -> str:
    """Search the user's uploaded documents for definitions, methodology, or context.

    Use this for questions the database cannot answer, such as what a metric means
    or how it is calculated. Returns passages with their source filenames; cite
    those filenames and never invent a source.
    """
    result = retrieve(query, current_session.get())
    if result["status"] == "empty":
        return _dump({"status": "empty", "message": "No documents have been indexed for this session."})
    filenames = sorted({source["filename"] for source in result["sources"] if source.get("filename")})
    return _dump({"status": "ok", "sources": result["sources"],
                  "cite_these_filenames": filenames,
                  "citation_instruction": (
                      f"Name {' and '.join(filenames)} in your answer as the source of this "
                      f"information, even if the answer also covers other topics."),
                  "passages": [chunk["text"][:800] for chunk in result["chunks"]]})


TOOLS = [list_tables, describe_table, run_sql, compare_groups, correlate,
         find_outliers, describe_distribution, make_chart, search_documents]
