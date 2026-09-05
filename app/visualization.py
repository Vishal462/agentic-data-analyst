"""Schema-neutral deterministic chart renderer."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from app.planner import AnalysisPlan


def render_chart(data: pd.DataFrame, plan: AnalysisPlan, output_dir: Path) -> Path | None:
    """Render only a validated plan's selected columns; never asks an LLM for values."""
    if plan.visualization == "none" or data.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 5))
    metric, group, date = plan.metric, plan.group_by, plan.date_column
    if plan.visualization == "line" and date and metric and {date, metric}.issubset(data):
        data.groupby(date, dropna=True)[metric].mean().sort_index().plot(ax=axis, kind="line")
    elif plan.visualization == "scatter" and metric:
        numeric = data.select_dtypes(include="number").columns.tolist()
        if len(numeric) < 2: return None
        axis.scatter(data[numeric[0]], data[numeric[1]], alpha=.45); axis.set(xlabel=numeric[0], ylabel=numeric[1])
    elif plan.visualization == "histogram" and metric and metric in data:
        pd.to_numeric(data[metric], errors="coerce").dropna().plot(ax=axis, kind="hist", bins=30); axis.set_xlabel(metric)
    elif group and metric and {group, metric}.issubset(data):
        data.groupby(group, dropna=True)[metric].mean().sort_values(ascending=False).head(25).plot(ax=axis, kind="bar")
    elif group and group in data:
        data[group].value_counts().head(25).plot(ax=axis, kind="bar")
    else: return None
    axis.set_title(f"{plan.visualization.title()} analysis")
    fig.tight_layout(); target = output_dir / "analysis_chart.png"; fig.savefig(target, dpi=150); plt.close(fig)
    return target
