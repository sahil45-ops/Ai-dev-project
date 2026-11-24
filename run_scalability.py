"""Measure local Python repositories in isolated processes.

Manifest: [{"name":"small","source":"path/to/source","repo":"path/to/repo"}].
The largest successful sample is an observed bound, never a proven maximum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil


def worker(entry, output, static=False, history=False, rag=False):
    from dataset.duplication import find_duplicate_lines
    from dataset.function_index import index_functions, iter_python_files
    from enricher.callgraph_enricher import build_call_counts
    source = str(Path(entry["source"]).resolve())
    repo = str(Path(entry.get("repo", source)).resolve())
    result = {"stages_seconds": {}, "scope": {"static": static, "history": history, "rag": rag}}
    def save():
        Path(output).write_text(json.dumps(result), encoding="utf-8")
    def stage(name, fn):
        start = time.perf_counter()
        try:
            return fn()
        finally:
            result["stages_seconds"][name] = time.perf_counter() - start
            save()
    if not Path(source).is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    files = iter_python_files(source)
    fingerprint = hashlib.sha256()
    for path in files:
        fingerprint.update(os.path.relpath(path, source).encode())
        fingerprint.update(Path(path).read_bytes())
    result["source_sha256"] = fingerprint.hexdigest()
    lines = sum(len(Path(f).read_text(encoding="utf-8", errors="replace").splitlines()) for f in files)
    result.update(python_files=len(files), physical_lines=lines, kloc=lines / 1000)
    functions = stage("function_index", lambda: index_functions(repo, source))
    result["functions"] = len(functions)
    if static:
        from analyzer.analyzer import run_all
        issues, _ = stage("static_analysis", lambda: run_all(source))
        result["issues"] = len(issues)
    stage("duplication", lambda: find_duplicate_lines(
        {os.path.relpath(f, source): Path(f).read_text(encoding="utf-8") for f in files}))
    stage("call_graph", lambda: build_call_counts(source))
    if history:
        from enricher.function_history import build_function_history
        from enricher.git_enricher import build_file_context
        cutoff = datetime.fromisoformat(entry["as_of"])
        cutoff = cutoff.replace(tzinfo=timezone.utc) if cutoff.tzinfo is None else cutoff
        since = cutoff - timedelta(days=730)
        stage("file_history", lambda: build_file_context(repo, 730, cutoff))
        stage("function_history", lambda: build_function_history(repo, since, cutoff))
    if rag and functions:
        from rag.retriever import DebtRetriever
        from rag.vector_store import DebtVectorStore
        # The worker owns an ephemeral store; no persistent database handles to clean up.
        store = stage("rag_initialization", lambda: DebtVectorStore())
        stage("rag_index", lambda: store.ingest_issues(functions))
        retriever = DebtRetriever(store)
        queries = functions[:min(100, len(functions))]
        stage("rag_retrieval", lambda: [retriever.retrieve(x, top_k=min(5, len(functions))) for x in queries])
        result["rag_queries"] = len(queries)
        result["rag_mean_retrieval_seconds"] = result["stages_seconds"]["rag_retrieval"] / len(queries)
    result.update(status="complete", llm_tokens=None, llm_cost_usd=None)
    save()


def stop_tree(proc):
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
    except psutil.NoSuchProcess:
        pass
    proc.wait()


def measure(entry, seconds, memory_mb, flags):
    with tempfile.TemporaryDirectory(prefix="debt-benchmark-") as directory:
        spec, out, log = [Path(directory) / name for name in ("input.json", "output.json", "log.txt")]
        spec.write_text(json.dumps(entry), encoding="utf-8")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(spec), "--output", str(out), *flags]
        start, peak, status = time.perf_counter(), 0, None
        with log.open("w", encoding="utf-8") as stream:
            proc = subprocess.Popen(command, stdout=stream, stderr=stream,
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            try:
                while proc.poll() is None:
                    try:
                        parent = psutil.Process(proc.pid)
                        rss = 0
                        for process in [parent, *parent.children(recursive=True)]:
                            try:
                                rss += process.memory_info().rss
                            except psutil.NoSuchProcess:
                                pass
                        peak = max(peak, rss)
                    except psutil.NoSuchProcess:
                        pass
                    if time.perf_counter() - start > seconds:
                        status = "timeout"
                    elif peak > memory_mb * 1024 ** 2:
                        status = "memory_limit"
                    if status:
                        stop_tree(proc)
                        break
                    time.sleep(0.1)
            finally:
                if proc.poll() is None:
                    stop_tree(proc)
        partial = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        return {**partial, "name": entry["name"], "source": entry["source"],
                "status": status or ("complete" if proc.returncode == 0 else "failed"),
                "wall_seconds": time.perf_counter() - start, "peak_tree_rss_mb": peak / 1024 ** 2,
                "log_tail": log.read_text(encoding="utf-8", errors="replace")[-4000:]}


def synthetic_manifest(root):
    """Deterministic structural fixtures, not representative production repositories."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    entries = []
    for name, count in (("small", 5), ("medium", 50), ("large", 250)):
        folder = root / name
        folder.mkdir()
        for i in range(count):
            source = "\n".join(
                f"def function_{j}(value):\n    if value > {i+j}:\n        return value + {j}\n    return value - {i}\n"
                for j in range(20))
            (folder / f"module_{i}.py").write_text(source, encoding="utf-8")
        entries.append({"name": f"synthetic_{name}", "source": str(folder.resolve())})
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest")
    p.add_argument("--synthetic-root", help="Create deterministic fixtures in a NEW directory instead of reading a manifest")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--output", default="reports/scalability.json")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--memory-mb", type=float, default=4096)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--static", action="store_true")
    p.add_argument("--history", action="store_true")
    p.add_argument("--rag", action="store_true")
    args = p.parse_args()
    if args.worker:
        worker(json.loads(Path(args.worker).read_text(encoding="utf-8")), args.output,
               args.static, args.history, args.rag)
        return
    if bool(args.manifest) == bool(args.synthetic_root) or min(args.timeout, args.memory_mb, args.repeats) <= 0:
        p.error("Supply --manifest and positive resource budgets/repeats")
    entries = (synthetic_manifest(args.synthetic_root) if args.synthetic_root else
               json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    flags = [f"--{name}" for name in ("static", "history", "rag") if getattr(args, name)]
    results = []
    for entry in entries:
        for repeat in range(args.repeats):
            row = measure(entry, args.timeout, args.memory_mb, flags)
            row["repeat"] = repeat + 1
            results.append(row)
            print(f"{entry['name']} repeat {repeat + 1}: {row['status']}, {row['wall_seconds']:.2f}s")
    reliable = [e["name"] for e in entries if all(
        r["status"] == "complete" for r in results if r["name"] == e["name"])]
    successful = [r for r in results if r["name"] in reliable and r.get("kloc") is not None]
    report = {"environment": {"python": sys.version, "platform": platform.platform(),
                              "cpu_count": os.cpu_count(), "ram_mb": psutil.virtual_memory().total / 1024 ** 2},
              "budgets": {"seconds": args.timeout, "memory_mb": args.memory_mb},
              "enabled_optional_stages": flags, "runs": results,
              "largest_successful_kloc": max((r["kloc"] for r in successful), default=None),
              "maximum_supported_kloc": None,
              "limitations": ["Observed bound applies only to these repositories and enabled stages.",
                              "RSS is sampled every 100ms across the process tree; brief peaks can be missed.",
                              "RAG indexes all functions as a load probe; this is not ranking quality evaluation.",
                              "LLM usage and cost are measured separately by run_model_benchmark.py.",
                              "Analyzer wrappers may report empty findings on tool failures; inspect logs.",
                              "Repeated runs can benefit from OS/model caches; record cold and warm runs separately."]}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
