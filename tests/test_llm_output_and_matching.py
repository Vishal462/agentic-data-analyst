import pytest

from app.planner import build_plan
from app.semantic_validation import validate_sql_semantics
from app.planner import AnalysisPlan

CATALOG = {"flights": {"columns": [
    {"name": "AIRLINE", "kind": "categorical"},
    {"name": "DEP_DELAY", "kind": "numeric"},
    {"name": "DISTANCE", "kind": "numeric"},
    {"name": "FL_DATE", "kind": "date"},
]}}


def test_unmatched_columns_are_not_invented():
    plan = build_plan("How many rows are there?", CATALOG)
    assert plan.metric is None and plan.group_by is None and plan.date_column is None


def test_plural_question_words_match_singular_columns():
    plan = build_plan("Compare DEP_DELAY across airlines for significance", CATALOG)
    assert plan.group_by == "AIRLINE" and plan.metric == "DEP_DELAY"


def test_required_columns_still_fall_back():
    # A trend needs a date column even when the question never names one.
    plan = build_plan("Show the trend over time", CATALOG)
    assert plan.date_column == "FL_DATE"


def test_quoted_identifiers_pass_semantic_validation():
    plan = AnalysisPlan(intent="sql", table="flights", group_by="AIRLINE", metric="DEP_DELAY",
                        aggregation="mean", ranking="desc", limit=10)
    sql = ('SELECT "AIRLINE", AVG("DEP_DELAY") AS metric_value FROM "flights" '
           'WHERE "DEP_DELAY" IS NOT NULL GROUP BY "AIRLINE" ORDER BY metric_value DESC LIMIT 10')
    assert validate_sql_semantics(sql, plan) == []


def test_order_by_does_not_satisfy_group_by():
    plan = AnalysisPlan(intent="sql", table="flights", group_by="AIRLINE", metric="DEP_DELAY", aggregation="mean")
    sql = 'SELECT "DEP_DELAY" FROM "flights" ORDER BY "AIRLINE" DESC'
    assert "expected GROUP BY AIRLINE" in validate_sql_semantics(sql, plan)


# The SQL agent opens DuckDB at import time; skip if the fixture is unavailable.
# importorskip only catches ImportError, but importing this module also opens the
# DuckDB fixture - absent in a fresh clone - which raises an OperationalError and
# would fail collection for the whole file, including the tests above that need no
# database at all.
try:
    import app.lanchain_sql_agent as agent
except Exception as exc:  # pragma: no cover - depends on local fixtures
    agent = None
    _NO_AGENT = f"requires the DuckDB fixture: {type(exc).__name__}"
else:
    _NO_AGENT = ""

needs_agent = pytest.mark.skipif(bool(_NO_AGENT), reason=_NO_AGENT or "available")


@needs_agent
@pytest.mark.parametrize("raw", [
    "<think>reasoning</think>SELECT a FROM t",
    "<think>reasoning</think> ```sql SELECT a FROM t; ```",
    "leaked reasoning</think>SELECT a FROM t",
    "Here is the query:\nSELECT a FROM t",
    [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "SELECT a FROM t"}],
])
def test_reasoning_tokens_never_reach_the_sql_parser(raw):
    cleaned = agent._clean_sql(raw)
    assert agent._sql_only(cleaned) and "think" not in cleaned.lower()


@needs_agent
def test_truncated_reasoning_yields_no_sql():
    assert agent._clean_sql("<think>cut off mid-thought") == ""


@needs_agent
def test_prose_answers_are_stripped_of_reasoning():
    assert agent._strip_reasoning("<think>hmm</think>The average is 12.") == "The average is 12."
