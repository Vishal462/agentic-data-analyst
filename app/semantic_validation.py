"""Deterministic checks that generated SQL honors a validated analysis plan."""
import re
from app.planner import AnalysisPlan

# Identifiers may appear bare or quoted; DuckDB quotes with " and tolerates `.
_QUOTE = r'["`]?'
# Clauses that terminate a GROUP BY list.
_GROUP_BY_END = re.compile(r"\b(ORDER\s+BY|HAVING|LIMIT|OFFSET|WINDOW|QUALIFY|UNION|EXCEPT|INTERSECT)\b", re.I)

def _identifier_pattern(identifier: str) -> str:
    """Match an identifier bare or quoted, but never inside a longer word or string literal."""
    return rf"(?<![\w']){_QUOTE}{re.escape(identifier)}{_QUOTE}(?![\w'])"

def _mentioned(sql: str, identifier: str) -> bool:
    return bool(re.search(_identifier_pattern(identifier), sql, re.I))

def _group_by_clause(sql: str) -> str | None:
    """Return just the grouping expressions, stripped of any trailing clauses."""
    match = re.search(r"\bGROUP\s+BY\b(.*)", sql, re.I | re.S)
    if not match:
        return None
    clause = match.group(1)
    terminator = _GROUP_BY_END.search(clause)
    return clause[: terminator.start()] if terminator else clause

def validate_sql_semantics(sql: str, plan: AnalysisPlan) -> list[str]:
    """Return concrete plan/SQL mismatches; an empty list permits execution."""
    issues: list[str] = []
    if not re.search(rf"\bFROM\s+{_QUOTE}{re.escape(plan.table)}{_QUOTE}", sql, re.I):
        issues.append(f"expected table {plan.table}")
    if plan.metric and not _mentioned(sql, plan.metric):
        issues.append(f"expected metric {plan.metric}")
    if plan.aggregation is None:
        # Projection and sampling queries have no aggregate shape to verify.
        return issues
    if plan.group_by:
        clause = _group_by_clause(sql)
        if clause is None or not _mentioned(clause, plan.group_by):
            issues.append(f"expected GROUP BY {plan.group_by}")
    if plan.aggregation == "mean" and plan.metric:
        if not re.search(rf"\bAVG\s*\(\s*{_QUOTE}{re.escape(plan.metric)}{_QUOTE}", sql, re.I):
            issues.append(f"expected AVG({plan.metric})")
    if plan.ranking == "desc" and plan.group_by:
        if not re.search(r"\bORDER\s+BY\b[\s\S]*?\bDESC\b", sql, re.I):
            issues.append("expected descending ranking")
    if plan.limit and not re.search(rf"\bLIMIT\s+{plan.limit}\b", sql, re.I):
        issues.append(f"expected LIMIT {plan.limit}")
    return issues
