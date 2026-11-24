"""Compare models on frozen contexts. Real API calls require --execute.

Example: python run_model_benchmark.py --items data/requests_items.json
  --truth data/requests_ground_truth.json --models model-a model-b
  --retrieval data/requests_llm_rankings_retrieval.json --execute
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from itertools import pairwise
from pathlib import Path

import config  # noqa: F401 -- loads local provider configuration
from agents.agents import analysis_agent, priority_agent, refactor_agent, single_agent
from agents.prioritise import build_contexts, estimate_tokens, prioritise
from evaluation.evaluate import evaluate_against_defects, paired_difference
from evaluation.metrics import order_from_scores, spearman
from llm.client import LLMError, _validate, make_client


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class MeasuredClient:
    """Serial benchmark wrapper: measure validated responses and failed attempts."""
    def __init__(self, client):
        self.client, self.model, self.usage = client, client.model, client.usage
        self.latencies = []
        self.valid = 0
        self.failures = 0

    def complete_json(self, system, prompt, schema, max_tokens=16000):
        start = time.perf_counter()
        try:
            result = _validate(self.client.complete_json(system, prompt, schema, min(max_tokens, 2048)), schema)
            self.valid += 1
            return result
        except LLMError:
            self.failures += 1
            raise
        finally:
            self.latencies.append(time.perf_counter() - start)


def compare(items, truth, contexts, models, repeats, factory, prices=None, checkpoint=None,
            mode="chain", workers=1):
    if repeats < 1 or len({x["id"] for x in items}) != len(items):
        raise ValueError("Positive repeats and unique item IDs are required.")
    runs, methods = [], {}
    for model in models:
        for arm, prompts in contexts.items():
            for repeat in range(repeats):
                key = f"{model} | {arm} | {repeat + 1}"
                client = MeasuredClient(factory(model))
                start = time.perf_counter()
                results = prioritise(items, client, prompts, mode=mode, workers=workers)
                scores = {r["id"]: r["priority_score"] for r in results if "error" not in r}
                methods[key] = {"scores": scores, "order": order_from_scores(scores)}
                usage = client.usage
                rate = (prices or {}).get(model)
                cost = ((usage.input_tokens * rate[0] + usage.output_tokens * rate[1]) / 1e6
                        if rate is not None else None)
                runs.append({"key": key, "model": model, "arm": arm, "repeat": repeat + 1,
                             "workers": workers,
                             "seconds": time.perf_counter() - start,
                             "mean_call_seconds": statistics.mean(client.latencies) if client.latencies else None,
                             "calls_attempted": len(client.latencies),
                             "schema_acceptance_rate": client.valid / len(client.latencies) if client.latencies else None,
                             "call_failure_rate": client.failures / len(client.latencies) if client.latencies else None,
                             "item_failure_rate": 1 - len(scores) / len(items) if items else None,
                             "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                             "cached_calls": usage.cached_calls, "cost_usd": cost,
                             "pricing_per_million": rate, "results": results})
                if checkpoint:
                    checkpoint(runs)
    # Compare models on a matched intersection per arm, not the intersection of
    # every model and every arm (one failed arm must not erase all quality data).
    evaluation = {"n_items": 0, "n_positive": 0, "methods": {}, "method_n": {}, "method_positive": {}}
    all_shared = set()
    for arm in contexts:
        for repeat in range(repeats):
            keys = [f"{model} | {arm} | {repeat + 1}" for model in models]
            present = [key for key in keys if key in methods]
            if len(present) < 1:
                continue
            shared = set.intersection(*(set(methods[key]["scores"]) for key in present)) & set(truth)
            if not shared:
                continue
            all_shared.update(shared)
            aligned = {key: {"scores": {i: methods[key]["scores"][i] for i in shared},
                             "order": [i for i in methods[key]["order"] if i in shared]}
                       for key in present}
            part = evaluate_against_defects(aligned, truth)
            evaluation["methods"].update(part["methods"])
            evaluation["method_n"].update({key: part["n_items"] for key in present})
            evaluation["method_positive"].update({key: part["n_positive"] for key in present})
    evaluation["n_items"] = len(all_shared)
    evaluation["n_positive"] = sum(bool(truth.get(i, {}).get("future_defect")) for i in all_shared)
    summary = []
    metric_keys = ("auc_future_defect", "ndcg@10_fix_commits", "precision@10",
                   "spearman_vs_change_commits")
    for model in models:
        for arm in contexts:
            selected = [r for r in runs if r["model"] == model and r["arm"] == arm]
            scored = [evaluation.get("methods", {}).get(r["key"], {}) for r in selected]
            summary.append({
                "model": model, "arm": arm, "repeats": len(selected),
                "mean_metrics": {metric: (statistics.mean(vals) if vals else None)
                                 for metric in metric_keys
                                 for vals in [[row[metric] for row in scored if isinstance(row.get(metric), (int, float))]]},
                "mean_seconds": (statistics.mean(v) if (v := [r["seconds"] for r in selected if r.get("seconds") is not None]) else None),
                "mean_call_seconds": (statistics.mean(v) if (v := [r["mean_call_seconds"] for r in selected if r.get("mean_call_seconds") is not None]) else None),
                "mean_cost_usd": (statistics.mean(v) if (v := [r["cost_usd"] for r in selected if r.get("cost_usd") is not None]) else None),
                "mean_item_failure_rate": (statistics.mean(v) if (v := [r["item_failure_rate"] for r in selected if r.get("item_failure_rate") is not None]) else None),
                "mean_schema_acceptance_rate": (statistics.mean(v) if (v := [r["schema_acceptance_rate"] for r in selected if r.get("schema_acceptance_rate") is not None]) else None),
                "quality_n_items": evaluation["method_n"].get(selected[0]["key"], 0) if selected else 0,
                "quality_n_positives": evaluation["method_positive"].get(selected[0]["key"], 0) if selected else 0,
                "pilot_interpretation": "descriptive only; repeat and dataset size are insufficient for a general model winner",
            })
    consistency = []
    for model in models:
        for arm in contexts:
            selected = [r for r in runs if r["model"] == model and r["arm"] == arm]
            for a, b in pairwise(selected):
                sa, sb = methods[a["key"]]["scores"], methods[b["key"]]["scores"]
                shared = set(sa) & set(sb)
                consistency.append({"model": model, "arm": arm, "runs": [a["repeat"], b["repeat"]],
                                    "n_shared": len(shared), "spearman": spearman(sa, sb),
                                    "exact_score_agreement": sum(sa[i] == sb[i] for i in shared) / len(shared) if shared else None})
    paired = []
    future_changes = {key: float(row.get("future_commits", 0)) for key, row in truth.items()}
    for arm in contexts:
        for repeat in range(repeats):
            for first_index, first_model in enumerate(models):
                for second_model in models[first_index + 1:]:
                    first = f"{first_model} | {arm} | {repeat + 1}"
                    second = f"{second_model} | {arm} | {repeat + 1}"
                    if first in methods and second in methods:
                        paired.append({"arm": arm, "repeat": repeat + 1,
                                       "model_a": first_model, "model_b": second_model,
                                       "truth": "future change commits",
                                       **paired_difference(methods[first], methods[second], future_changes)})
    return {"runs": runs, "summary": summary, "evaluation": evaluation,
            "consistency": consistency, "paired_model_comparisons": paired,
            "notes": ["All quality metrics use the intersection of successful items across every run.",
                      "Inspect failure rates and shared sample size before comparing quality.",
                      "Schema acceptance includes transport/refusal failures; raw invalid-JSON rate is not exposed by every provider.",
                      "Costs use supplied per-million token rates; unknown prices are null. Provider retries may incur unreported usage.",
                      "Independent repeats use no disk cache. No best model is declared from incomplete evidence."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--items", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--retrieval", help="Frozen item-ID to retrieved-context JSON; required for the RAG arm")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent item calls; keep fixed across models and arms")
    p.add_argument("--mode", choices=("single", "chain"), default="chain")
    p.add_argument("--sample-size", type=int, help="Optional seeded random pilot sample; selection never uses labels")
    p.add_argument("--prices", help="JSON mapping model to [input, output] USD per million tokens")
    p.add_argument("--output", default="reports/model_benchmark.json")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    items, truth = read(args.items), read(args.truth)
    full_dataset_sha256 = digest(items)
    if args.sample_size:
        import random
        if args.sample_size < 10 or args.sample_size > len(items):
            p.error("Pilot sample must contain between 10 and all dataset items")
        items = sorted(random.Random(7).sample(sorted(items, key=lambda x: x["id"]), args.sample_size),
                       key=lambda x: x["id"])
    if not items or args.repeats < 1 or args.workers < 1:
        p.error("A nonempty dataset and positive repeat/worker counts are required")
    retrieval = read(args.retrieval) if args.retrieval else None
    if retrieval is not None and any(x["id"] not in retrieval for x in items):
        p.error("Frozen retrieval must include every dataset item")
    contexts = {"no_context": build_contexts(items, False, None, "pre-cutoff")}
    if retrieval is not None:
        contexts["rag_context"] = build_contexts(items, True, retrieval, "pre-cutoff")
    plan = {"dataset_sha256": digest(items), "truth_sha256": digest(truth),
            "full_dataset_sha256": full_dataset_sha256, "sample_ids": [x["id"] for x in items],
            "pilot": bool(args.sample_size), "mode": args.mode, "workers": args.workers,
            "max_output_tokens": 2048,
            "context_sha256": digest(contexts), "models": args.models, "repeats": args.repeats,
            "protocol_sha256": digest([(a.system_prompt(), a.schema) for a in
                                       ((single_agent,) if args.mode == "single" else
                                        (analysis_agent, priority_agent, refactor_agent))]),
            "token_estimates_per_model_per_repeat": {k: estimate_tokens(v, args.mode) for k, v in contexts.items()}}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    def save(value):
        target.write_text(json.dumps({**plan, **value}, indent=2), encoding="utf-8")
    if not args.execute:
        save({"status": "plan_only"})
        print(f"Plan saved to {target}; --execute runs real models.")
        return
    report = compare(items, truth, contexts, args.models, args.repeats,
                     lambda model: make_client(model, "medium", None),
                     read(args.prices) if args.prices else None,
                     lambda runs: save({"status": "running", "runs": runs}),
                     mode=args.mode, workers=args.workers)
    save({"status": "complete", **report})
    print(target)


if __name__ == "__main__":
    main()
