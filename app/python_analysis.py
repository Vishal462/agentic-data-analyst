"""Deterministic reusable statistics for schema-selected columns."""
import pandas as pd
from scipy import stats
from app.planner import AnalysisPlan


def _anova(data: pd.DataFrame, group: str, metric: str) -> dict:
    groups = [(str(name), pd.to_numeric(frame[metric], errors="coerce").dropna()) for name, frame in data.groupby(group)]
    groups = [(name, values) for name, values in groups if len(values) >= 2]
    if len(groups) < 2:
        return {"status": "unavailable", "reason": "At least two groups with two valid observations are required."}
    samples = [values for _, values in groups]
    f_statistic, p_value = stats.f_oneway(*samples)
    grand_mean = pd.concat(samples, ignore_index=True).mean()
    total_ss = sum(((values - grand_mean) ** 2).sum() for values in samples)
    between_ss = sum(len(values) * (values.mean() - grand_mean) ** 2 for values in samples)
    return {"status": "ok", "test": "one-way ANOVA", "f_statistic": float(f_statistic), "p_value": float(p_value), "alpha": .05,
            "significant": bool(p_value < .05), "sample_counts": {name: len(values) for name, values in groups},
            "eta_squared": float(between_ss / total_ss) if total_ss else None,
            "conclusion": "At least one group mean differs; ANOVA does not establish which pairs differ." if p_value < .05 else "Insufficient evidence that group means differ.",
            "practical_note": "Use eta-squared to judge magnitude; statistical significance alone is not practical importance."}


def run_python_analysis(data: pd.DataFrame, plan: AnalysisPlan) -> dict:
    if data.empty:
        return {"status": "empty", "message": "The reduced query returned no rows."}
    metric, group = plan.metric, plan.group_by
    numeric = data.select_dtypes(include="number")
    if "group_comparison" in plan.operations and metric and group and {metric, group}.issubset(data.columns):
        return _anova(data, group, metric)
    if "correlation" in plan.operations:
        return {"status": "ok", "analysis": "correlation", "correlation_matrix": numeric.corr().round(4).to_dict()} if numeric.shape[1] >= 2 else {"status": "unavailable", "reason": "Two numeric columns are required."}
    if "outliers" in plan.operations and metric in data:
        values = pd.to_numeric(data[metric], errors="coerce").dropna(); q1, q3 = values.quantile([.25, .75]); iqr = q3-q1
        return {"status": "ok", "analysis": "outliers", "column": metric, "count": int(((values < q1-1.5*iqr) | (values > q3+1.5*iqr)).sum())}
    if "distribution" in plan.operations and metric in data:
        return {"status": "ok", "analysis": "distribution", "column": metric, "statistics": pd.to_numeric(data[metric], errors="coerce").describe().to_dict()}
    if group and group in data and not metric:
        return {"status": "ok", "analysis": "categorical_frequency", "frequencies": data[group].value_counts(dropna=False).head(100).to_dict()}
    return {"status": "ok", "analysis": "descriptive_statistics", "statistics": numeric.describe().to_dict() if not numeric.empty else {}}
