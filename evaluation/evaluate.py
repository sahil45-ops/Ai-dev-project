"""
Evaluate — Phase 5: compare prioritisation methods against ground truth.

Methods (higher score = fix first):
  static_score        Phase 1 composite static score (primary baseline)
  complexity          raw cyclomatic complexity (the plan's baseline)
  context_heuristic   non-LLM equal-weight mix of static + context features
                      (tests whether an LLM is needed at all)
  llm                 LLM + repository context + RAG (the proposed method)
  llm_nocontext       same LLM, static metrics + code only (ablation)

Ground truths:
  defects   bug-fix commits touching the function *after* the cutoff
  changes   all non-sweeping commits touching it after the cutoff
  experts   mean 1–10 urgency score from human raters (sampled items)
"""

from __future__ import annotations

import math
from typing import Any

from evaluation.explanations import ADVANCED, BASE, HISTORY, explain_heuristic
from evaluation.metrics import (
    bootstrap_ci,
    kendall,
    ndcg_at_k,
    order_from_scores,
    precision_at_k,
    resampled_spearman,
    roc_auc,
    spearman,
    topk_overlap,
)

METHOD_LABELS = {
    "static_score": "Static analysis (composite score)",
    "complexity": "Static analysis (complexity only)",
    "context_heuristic": "Context heuristic (no LLM)",
    "llm": "LLM + RAG context (proposed)",
    "llm_nocontext": "LLM without context (ablation)",
    "context_no_history": "Context heuristic without Git history",
    "context_advanced": "Context heuristic with advanced Git history",
}

_HEURISTIC_FEATURES = (
    ("static_score", lambda x: x["static_score"]),
    ("callers", lambda x: math.log1p(x["callers"])),
    ("func_fix_commits", lambda x: math.log1p(x["func_fix_commits"])),
    ("func_commits", lambda x: math.log1p(x["func_commits"])),
    ("file_defects", lambda x: math.log1p(x["defect_commits"])),
    ("coverage_gap", lambda x: 100.0 - x["function_coverage"]),
)


def context_heuristic_scores(items: list[dict]) -> dict[str, float]:
    """Equal-weight mean of min–max normalised features."""
    return {k: v["score"] for k, v in explain_heuristic(items).items()}


def collect_methods(items: list[dict], llm: list[dict] | None,
                    llm_nocontext: list[dict] | None) -> dict[str, dict[str, Any]]:
    """Return {method: {"scores": {id: score}, "order": [ids]}}."""
    methods: dict[str, dict[str, Any]] = {}
    static = {x["id"]: x["static_score"] for x in items}
    methods["static_score"] = {
        "scores": static,
        "order": [x["id"] for x in sorted(items, key=lambda x: x["static_rank"])],
    }
    cx = {x["id"]: x["complexity"] for x in items}
    methods["complexity"] = {"scores": cx, "order": order_from_scores(cx)}
    ch = context_heuristic_scores(items)
    methods["context_heuristic"] = {"scores": ch, "order": order_from_scores(ch)}
    for name, features in (("context_no_history", BASE),
                           ("context_advanced", BASE + HISTORY + ADVANCED)):
        if name == "context_advanced" and not any(x.get("code_churn") is not None for x in items):
            continue
        sc = {k: v["score"] for k, v in explain_heuristic(items, features).items()}
        methods[name] = {"scores": sc, "order": order_from_scores(sc)}
    for name, results in (("llm", llm), ("llm_nocontext", llm_nocontext)):
        ok = [r for r in (results or []) if "error" not in r and "llm_rank" in r]
        if ok:
            methods[name] = {
                "scores": {r["id"]: r["priority_score"] for r in ok},
                "order": [r["id"] for r in sorted(ok, key=lambda r: r["llm_rank"])],
            }
    return methods


def _restrict(method: dict, ids: set[str]) -> dict:
    return {"scores": {k: v for k, v in method["scores"].items() if k in ids},
            "order": [i for i in method["order"] if i in ids]}


def _ranking_intervals(method, gains, labels, ks, n_boot=500):
    """Item bootstrap, retaining duplicate observations and the original tie order."""
    ids = sorted(gains)
    positions = {item: rank for rank, item in enumerate(method["order"])}
    def statistic(sample, metric):
        ordered = sorted(range(len(sample)), key=lambda j: (positions[sample[j]], j))
        scores = {str(j): method["scores"][item] for j, item in enumerate(sample)}
        sampled_gains = {str(j): gains[item] for j, item in enumerate(sample)}
        order = [str(j) for j in ordered]
        if metric == "auc_future_defect":
            return roc_auc(scores, {str(j): labels[item] for j, item in enumerate(sample)})
        if metric == "ndcg@10_fix_commits":
            return ndcg_at_k(order, sampled_gains, 10)
        k = int(metric.split("@")[1])
        relevant = {str(j) for j, item in enumerate(sample) if labels[item]}
        return precision_at_k(order, relevant, k)
    keys = ["auc_future_defect", "ndcg@10_fix_commits"] + [
        f"precision@{k}" for k in ks if k <= len(ids)]
    return {key + "_ci95": list(bootstrap_ci(ids, lambda s, key=key: statistic(s, key), n_boot=n_boot))
            for key in keys}


def evaluate_against_defects(methods: dict, ground_truth: dict, ks=(5, 10, 20)) -> dict:
    ids = set.intersection(*(set(m["scores"]) for m in methods.values())) & set(ground_truth)
    fixes = {i: float(ground_truth[i]["future_fix_commits"]) for i in ids}
    changes = {i: float(ground_truth[i]["future_commits"]) for i in ids}
    labels = {i: ground_truth[i]["future_defect"] for i in ids}
    relevant = {i for i in ids if labels[i]}
    out: dict[str, Any] = {"n_items": len(ids), "n_positive": len(relevant), "methods": {}}
    for name, method in methods.items():
        m = _restrict(method, ids)
        row = {
            "spearman_vs_fix_commits": spearman(m["scores"], fixes),
            "spearman_vs_change_commits": spearman(m["scores"], changes),
            "auc_future_defect": roc_auc(m["scores"], labels),
            "ndcg@10_fix_commits": ndcg_at_k(m["order"], fixes, 10),
            "ndcg@10_change_commits": ndcg_at_k(m["order"], changes, 10),
        }
        for k in ks:
            if k <= len(ids):
                row[f"precision@{k}"] = precision_at_k(m["order"], relevant, k)
        lo, hi = bootstrap_ci(sorted(ids), lambda s, sc=m["scores"]: resampled_spearman(sc, changes, s))
        row["spearman_vs_change_commits_ci95"] = [lo, hi]
        row.update(_ranking_intervals(m, fixes, labels, ks))
        out["methods"][name] = row
    return out


def evaluate_against_experts(methods: dict, expert_scores: dict[str, float],
                             ks=(5, 10)) -> dict:
    ids = set(expert_scores).intersection(*(set(m["scores"]) for m in methods.values()))
    if len(ids) < 3:
        return {"n_items": len(ids), "methods": {}, "note": "not enough expert-rated items"}
    expert = {i: expert_scores[i] for i in ids}
    expert_order = order_from_scores(expert)
    out: dict[str, Any] = {"n_items": len(ids), "methods": {}}
    for name, method in methods.items():
        m = _restrict(method, ids)
        row = {
            "spearman": spearman(m["scores"], expert),
            "kendall_tau": kendall(m["scores"], expert),
            "ndcg@10": ndcg_at_k(m["order"], expert, 10),
        }
        for k in ks:
            if k < len(ids):
                row[f"precision@{k}"] = topk_overlap(m["order"], expert_order, k)
        lo, hi = bootstrap_ci(sorted(ids), lambda s, sc=m["scores"]: resampled_spearman(sc, expert, s))
        row["spearman_ci95"] = [lo, hi]
        out["methods"][name] = row
    return out


def paired_difference(method_a: dict, method_b: dict, truth: dict[str, float]) -> dict:
    """Bootstrap CI of ρ(a, truth) − ρ(b, truth) on shared items."""
    ids = sorted(set(method_a["scores"]) & set(method_b["scores"]) & set(truth))

    def diff(sample: list[str]) -> float | None:
        ra = resampled_spearman(method_a["scores"], truth, sample)
        rb = resampled_spearman(method_b["scores"], truth, sample)
        return None if ra is None or rb is None else ra - rb

    shared_truth = {i: truth[i] for i in ids}
    point_a = spearman(method_a["scores"], shared_truth)
    point_b = spearman(method_b["scores"], shared_truth)
    lo, hi = bootstrap_ci(ids, diff)
    point = None if point_a is None or point_b is None else round(point_a - point_b, 4)
    return {"delta_spearman": point, "ci95": [lo, hi], "n_items": len(ids),
            "inference": "Exploratory paired item bootstrap; unadjusted for multiple comparisons.",
            "significant": bool(lo is not None and (lo > 0 or hi < 0))}
