"""Evaluate advanced pre-cutoff Git signals without modifying the original dataset."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from enricher.function_history import canonical_path
from enricher.git_enricher import build_file_context
from evaluation.evaluate import (
    collect_methods,
    evaluate_against_defects,
    paired_difference,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--output", default="reports/history_ablation.json")
    args = parser.parse_args()
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    items, truth, meta = read(args.items), read(args.truth), read(args.meta)
    contexts = build_file_context(args.repo, 730, datetime.fromisoformat(meta["as_of"]))
    if not contexts:
        raise SystemExit("No pre-cutoff Git context available; no ablation was reported.")
    churn = {}
    for path, context in contexts.items():
        key = canonical_path(path)
        churn[key] = churn.get(key, 0) + context["code_churn"]
    for item in items:
        item["code_churn"] = churn.get(item["file"], 0)
        item["recurring_bug_file"] = item.get("defect_commits", 0) >= 2
    methods = collect_methods(items, None, None)
    changes = {key: float(row["future_commits"]) for key, row in truth.items()}
    report = {"as_of": meta["as_of"], "history_window_days": 730,
              "evaluation": evaluate_against_defects(methods, truth),
              "paired": [{"comparison": f"{a} vs {b}", "truth": "changes",
                          **paired_difference(methods[a], methods[b], changes)}
                         for a, b in (("context_heuristic", "context_no_history"),
                                      ("context_advanced", "context_heuristic"))],
              "notes": ["Historical labels are proxies, not expert judgments.",
                        "Advanced features are added jointly; this does not isolate each feature's causal effect.",
                        "Contributor counts and recency use existing pre-cutoff dataset values.",
                        "Exploratory intervals are not adjusted for multiple comparisons."]}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    items_target = target.with_name(target.stem + "_items.json")
    items_target.write_text(json.dumps(items, indent=2), encoding="utf-8")
    if meta.get("repo"):
        repo_report = target.with_name(f"{meta['repo']}_history_ablation.json")
        repo_items = target.with_name(f"{meta['repo']}_history_ablation_items.json")
        if repo_report != target:
            repo_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if repo_items != items_target:
            repo_items.write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
