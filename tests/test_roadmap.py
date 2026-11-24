from datetime import datetime, timedelta, timezone

import pytest

from enricher.git_enricher import build_file_context
from evaluation.evaluate import _ranking_intervals
from evaluation.explanations import explain_heuristic
from evaluation.metrics import bootstrap_ci
from llm.client import LLMError, MockClient
from run_model_benchmark import MeasuredClient, compare
from run_scalability import measure
from tests.test_research_pipeline import commit, git


def test_unknown_coverage_is_not_a_gap_and_contributions_sum():
    items = [{"id": "a", "static_score": 2, "callers": 0, "coverage_known": False, "function_coverage": 0},
             {"id": "b", "static_score": 8, "callers": 4, "function_coverage": 80},
             {"id": "c", "static_score": 5, "callers": 2, "function_coverage": 20}]
    explained = explain_heuristic(items)
    gap = next(f for f in explained["a"]["factors"] if f["feature"] == "coverage_gap")
    assert not gap["available"]
    assert gap["contribution"] == 0
    for row in explained.values():
        assert row["score"] == pytest.approx(sum(f["contribution"] for f in row["factors"]), abs=1e-5)
    assert explain_heuristic([]) == {}


def test_intervals_keep_duplicates_and_degenerate_metrics_unknown():
    method = {"scores": {"a": 3, "b": 2, "c": 1}, "order": ["a", "b", "c"]}
    result = _ranking_intervals(method, {"a": 0, "b": 0, "c": 0},
                                {"a": False, "b": False, "c": False}, (2,), n_boot=20)
    assert result["auc_future_defect_ci95"] == [None, None]
    assert result["ndcg@10_fix_commits_ci95"] == [None, None]
    assert result["precision@2_ci95"] == [0, 0]
    assert bootstrap_ci([], lambda _: 1) == (None, None)


def test_git_churn_uses_cutoff_and_behavioral_fixes(tmp_path):
    git(tmp_path, "init", "-q")
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    file = tmp_path / "module.py"
    file.write_text("def f():\n    return 1\n", encoding="utf-8")
    commit(tmp_path, "Initial", start)
    file.write_text("def f():\n    return 2\n", encoding="utf-8")
    commit(tmp_path, "Fix crash", start + timedelta(days=1))
    file.write_text("def f():\n    return 3\n", encoding="utf-8")
    commit(tmp_path, "Fix docs", start + timedelta(days=2))
    cutoff = start + timedelta(days=3)
    before = build_file_context(str(tmp_path), 90, cutoff)["module.py"]
    file.write_text("def f():\n    return 4\n", encoding="utf-8")
    commit(tmp_path, "Fix another crash", start + timedelta(days=4))
    after = build_file_context(str(tmp_path), 90, cutoff)["module.py"]
    assert before == after
    assert after["code_churn"] == 6
    assert after["defect_commits"] == 1
    assert not after["recurring_bug_file"]


def test_model_wrapper_rejects_schema_and_records_failure():
    client = MeasuredClient(MockClient(lambda *args: {"x": "wrong"}))
    with pytest.raises(LLMError):
        client.complete_json("", "", {"type": "object", "properties": {"x": {"type": "integer"}}})
    assert client.failures == 1 and client.valid == 0
    assert len(client.latencies) == 1


def test_model_comparison_matches_inputs_and_reports_unknown_cost(monkeypatch):
    import run_model_benchmark as benchmark
    seen = []
    def rank(items, client, prompts, **kwargs):
        seen.append(prompts)
        return [{"id": x["id"], "priority_score": n, "llm_rank": n + 1}
                for n, x in enumerate(items)]
    monkeypatch.setattr(benchmark, "prioritise", rank)
    items = [{"id": str(i)} for i in range(12)]
    truth = {x["id"]: {"future_commits": i, "future_fix_commits": i % 2,
                       "future_defect": bool(i % 2)} for i, x in enumerate(items)}
    prompts = {"no_context": {x["id"]: "fixed prompt" for x in items}}
    report = compare(items, truth, prompts, ["a", "b"], 2,
                     lambda model: MockClient(lambda *args: {}, model))
    assert all(x == seen[0] for x in seen)
    assert report["evaluation"]["n_items"] == 12
    assert all(r["cost_usd"] is None for r in report["runs"])
    assert all(r["exact_score_agreement"] == 1 for r in report["consistency"])


def test_scalability_failed_path_and_timeout(tmp_path):
    entry = {"name": "missing", "source": str(tmp_path / "missing")}
    failed = measure(entry, seconds=20, memory_mb=1024, flags=[])
    assert failed["status"] == "failed"
    timed = measure(entry, seconds=0.00001, memory_mb=1024, flags=[])
    assert timed["status"] == "timeout"


def test_function_history_cutoff_has_no_overlap(tmp_path):
    from enricher.function_history import build_function_history
    git(tmp_path, "init", "-q")
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    cutoff = start + timedelta(days=1)
    path = tmp_path / "module.py"
    path.write_text("def f():\n    return 1\n", encoding="utf-8")
    commit(tmp_path, "Initial", start)
    path.write_text("def f():\n    return 2\n", encoding="utf-8")
    commit(tmp_path, "Fix crash", cutoff)
    pre = build_function_history(str(tmp_path), start, cutoff)["module.py::f"]
    post = build_function_history(str(tmp_path), cutoff, cutoff + timedelta(days=1))["module.py::f"]
    assert pre["commits"] == 1 and pre["fix_commits"] == 0
    assert post["commits"] == 1 and post["fix_commits"] == 1


def test_embeddings_are_bounded_by_ingestion_batch():
    from types import SimpleNamespace

    from rag.vector_store import DebtVectorStore
    batches, ids = [], []
    def encode(documents):
        batches.append(len(documents))
        return [[1.0]] * len(documents)
    store = DebtVectorStore.__new__(DebtVectorStore)
    store.embedder = SimpleNamespace(encode_batch=encode)
    store.collection = SimpleNamespace(upsert=lambda **kwargs: ids.extend(kwargs["ids"]))
    assert store.ingest_issues([{"function": "f"} for _ in range(7)], batch_size=3) == 7
    assert batches == [3, 3, 1] and len(set(ids)) == 7
