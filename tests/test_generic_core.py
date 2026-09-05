import pandas as pd

from app.planner import build_plan, validate_plan
from app.python_analysis import run_python_analysis


SALES_CATALOG = {"sales": {"columns": [
    {"name": "category", "kind": "categorical"},
    {"name": "revenue", "kind": "numeric"},
    {"name": "order_date", "kind": "date"},
]}}


def test_sales_group_comparison_is_schema_driven():
    plan = validate_plan(build_plan("Which categories have the highest revenue and are differences significant?", SALES_CATALOG), SALES_CATALOG)
    assert plan.table == "sales" and plan.metric == "revenue" and plan.group_by == "category" and plan.intent == "both"
    data = pd.DataFrame({"category": ["A"] * 4 + ["B"] * 4, "revenue": [100, 110, 105, 115, 20, 25, 30, 22]})
    result = run_python_analysis(data, plan)
    assert result["test"] == "one-way ANOVA" and "p_value" in result


def test_flights_like_columns_are_not_required():
    catalog = {"metrics": {"columns": [{"name": "segment", "kind": "categorical"}, {"name": "score", "kind": "numeric"}]}}
    plan = validate_plan(build_plan("Compare score by segment for significance", catalog), catalog)
    assert plan.table == "metrics" and plan.metric == "score" and plan.group_by == "segment"
