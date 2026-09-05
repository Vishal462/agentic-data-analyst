from app.planner import AnalysisPlan
from app.semantic_validation import validate_sql_semantics


def test_direct_columns_are_preserved_in_plan():
    from app.planner import build_plan
    catalog = {"records": {"columns": [{"name": "SEGMENT", "kind": "categorical"}, {"name": "VALUE", "kind": "numeric"}]}}
    plan = build_plan("Which SEGMENT has the highest average VALUE?", catalog)
    assert plan.group_by == "SEGMENT" and plan.metric == "VALUE"


def test_wrong_group_by_is_rejected_before_execution():
    plan = AnalysisPlan(intent="sql", table="records", group_by="company", metric="delay", aggregation="mean", ranking="desc", limit=10)
    wrong_sql = "SELECT city, AVG(delay) FROM records GROUP BY city ORDER BY AVG(delay) DESC LIMIT 10"
    assert "expected GROUP BY company" in validate_sql_semantics(wrong_sql, plan)


def test_valid_aggregate_sql_is_accepted():
    plan = AnalysisPlan(intent="sql", table="records", group_by="company", metric="delay", aggregation="mean", ranking="desc", limit=10)
    sql = "SELECT company, AVG(delay) FROM records GROUP BY company ORDER BY AVG(delay) DESC LIMIT 10"
    assert validate_sql_semantics(sql, plan) == []
