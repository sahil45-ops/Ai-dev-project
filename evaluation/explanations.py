"""Auditable heuristic contributions; these are not LLM causal attributions."""
from __future__ import annotations

import math

from agents.prompts import coverage_is_known

BASE = ("static_score", "callers", "coverage_gap")
HISTORY = ("func_fix_commits", "func_commits", "defect_commits")
ADVANCED = ("code_churn", "commit_authors", "recent_activity", "recurring_bug_file")


def feature_value(item, name):
    if name == "coverage_gap":
        value = item.get("function_coverage")
        return 100 - value if coverage_is_known(item) and value is not None else None
    if name == "recent_activity":
        days = item.get("last_modified_days", -1)
        return 1 / (1 + days) if days is not None and days >= 0 else None
    value = item.get(name)
    if value is None:
        return None
    return float(value) if name in ("static_score", "recurring_bug_file") else math.log1p(max(0, value))


def explain_heuristic(items, features=BASE + HISTORY):
    """Min-max contributions with fixed weights; unavailable features contribute zero.

    Constant features also contribute zero. Missingness is retained in the output,
    so absent coverage is never represented as measured zero coverage.
    """
    result = {x["id"]: {"score": 0.0, "factors": []} for x in items}
    for name in features:
        values = [feature_value(x, name) for x in items]
        known = [v for v in values if v is not None]
        lo, hi = (min(known), max(known)) if known else (0, 0)
        for item, value in zip(items, values):
            contribution = (value - lo) / (hi - lo) / len(features) if value is not None and hi > lo else 0.0
            row = result[item["id"]]
            row["score"] += contribution
            row["factors"].append({"feature": name, "transformed_value": value,
                                   "contribution": contribution, "available": value is not None})
    for row in result.values():
        row["score"] = round(row["score"], 5)
        row["factors"].sort(key=lambda f: -f["contribution"])
    return result


def evidence_for(item):
    """Recommendations tied to observed inputs, with no invented LLM weights."""
    recommendations = []
    if item.get("complexity", 0) >= 10:
        recommendations.append("Inspect branches and extract cohesive helpers; verify behavior with tests.")
    if not item.get("coverage_known", True):
        recommendations.append("Measure coverage before choosing a refactoring test plan.")
    elif item.get("function_coverage", 100) < 80:
        recommendations.append("Add tests for uncovered paths before refactoring.")
    if item.get("func_fix_commits", 0) or item.get("defect_commits", 0):
        recommendations.append("Review historical bug fixes and add regression tests for those cases.")
    if item.get("callers", 0):
        recommendations.append("Check callers and preserve the public interface.")
    return {"observations": {k: item.get(k) for k in (
        "complexity", "function_coverage", "coverage_known", "callers", "func_commits",
        "func_fix_commits", "code_churn", "commit_authors", "last_modified_days")},
        "recommendations": recommendations,
        "interpretation": "Observed evidence and heuristic contributions are not causal explanations of an LLM score."}
