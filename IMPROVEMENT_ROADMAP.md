# Improvement roadmap: implementation and evidence

## 1. Evaluation and visualizations

The HTML dashboard compares Static Analysis, Context Heuristic, LLM + RAG, and
LLM without Context. Missing methods appear as unavailable. It includes ROC,
Precision@K, NDCG@10, Spearman correlation, and per-method top-ten charts linking
to evidence. Scores in top-ten charts use each method's own scale.

Run `python run_evaluation.py`, then `python run_dashboard.py`.
The underlying evaluation now includes reproducible item-bootstrap intervals for
AUC, defect NDCG@10, and Precision@K (500 resamples); Spearman uses 2,000.
Undefined metrics remain null. Paired Spearman comparisons are exploratory,
unadjusted for multiple testing. Correlated functions within files and the small
number of positives limit inference. Ranking intervals resample items with
replacement; duplicate observations retain the original ranking tie order.

The existing illustrative proxy section is retained and explicitly labeled
synthetic. It is not an experimental result and must not support research claims.

## 2. Scalability

`run_scalability.py` runs each repository in a separate process with a wall-time
budget and sampled process-tree RSS memory budget. It records Python physical
KLOC, file/function counts, per-stage latency, hardware, failures, and source
fingerprints. KLOC includes comments and blank lines; it is not executable SLOC.

```powershell
python run_scalability.py --synthetic-root targets/benchmark-fixtures --repeats 3 --output reports/scale.json
python run_scalability.py --manifest reports/scalability_manifest.json --static --rag --repeats 3
python run_scalability.py --manifest reports/scalability_history_manifest.json --history --repeats 3
```

The fixture directory must be new. Custom manifests contain objects with
`name`, `source`, and optional `repo`. History runs require an `as_of` timestamp
and a repository with reachable history. Keep snapshots fixed across methods.
Default structural stages are function indexing, duplication, and call graph.
Static analysis, history, and RAG are explicitly enabled options. RAG timing
includes a separate initialization stage, embedding/indexing, and up to 100
retrieval queries. Its load probe indexes all functions, not just debt candidates.
Model download/startup and OS cache effects must be reported separately.

Measured synthetic structural runs:

| Size | KLOC | Files | Functions | Median seconds (3 runs) | Peak RSS MB |
|---|---:|---:|---:|---:|---:|
| Small | 0.495 | 5 | 100 | 0.256 | 25.3 |
| Medium | 4.950 | 50 | 1,000 | 0.840 | 27.6 |
| Large | 24.750 | 250 | 5,000 | 4.099 | 37.2 |

With all four static analyzers enabled, a fresh run per fixture took 3.68, 4.63,
and 5.96 seconds, with sampled peak RSS of 80.8, 92.5, and 75.1 MB respectively.
Static analysis consumed 3.40, 3.94, and 3.78 seconds. An earlier cold/concurrent
run was substantially slower; treat single runs as variable, not capacity guarantees.
Raw evidence is in `reports/scalability_structural.json` and
`reports/scalability_static.json`.

The installed CPU embedding model materially changes RAG scale: indexing all
functions took 38.02, 59.33, and 120.02 seconds for the same sizes; 100-query
retrieval averaged 19, 28, and 35 milliseconds. Sampled process-tree RSS rose
to 626–695 MB, largely due to loading the embedding model. These are single
warm/first-load probes, not three-repeat reliability estimates. See
`reports/scalability_rag.json`.

The real Requests snapshot contained 5.426 KLOC, 18 Python files, and 235
functions. Indexing, file history, and function history took 0.28, 10.86, and
2.10 seconds; total elapsed time was 13.82 seconds and sampled RSS peaked at
121 MB. History traversal was the dominant enabled stage. See
`reports/scalability_history.json`.

These synthetic samples establish an observed bound for enabled stages only.
They do **not** establish a maximum supported repository size. To establish an
operating limit, repeat the complete pipeline on real small/medium/large
repositories, increase size until a resource budget fails, and report the last
size with all repetitions successful. Include history length, dependency shape,
embedding model, analyzer versions, and LLM shortlist size. Analyzer wrappers
currently sometimes return empty findings on tool errors, so logs require review.

Memory improvements: embeddings are generated one ingestion batch at a time;
function-history AST spans are retained only for the current commit. AUC uses
rank sums instead of enumerating every positive-negative pair. Remaining
bottlenecks include buffered Git diffs, AST parsing, external analyzers, full
function/source lists, and one LLM request per item (three for chain mode).

## 3. Model comparison

`run_model_benchmark.py` freezes source/context inputs across models and arms,
records dataset, labels, prompt and schema fingerprints, and disables disk caches
for independent repeats. It records token usage, elapsed time, schema acceptance,
call/item failures, supplied-price cost estimates, and repeated-score agreement.
Quality metrics compare models on the shared successfully ranked items within
each context arm and repeat. Failure rates and the matched sample size accompany
quality scores; a failed arm does not erase results from the other arm.

```powershell
python run_model_benchmark.py --items data/requests_items.json --truth data/requests_ground_truth.json --retrieval data/requests_llm_rankings_retrieval.json --models groq/openai/gpt-oss-20b groq/openai/gpt-oss-120b --mode single --workers 1 --sample-size 30 --repeats 1 --prices reports/benchmark_prices.json --output reports/model_benchmark_pilot.json --execute
```

Omit `--execute` to create a plan without API calls. Omit `--sample-size` for the
full dataset. Pilot selection uses seed 7 and never examines labels. Both models
must use the same mode; single-call and three-agent-chain results are separate
experiments. Output is capped at 2,048 tokens per call. Omit `--retrieval` to run
only the no-context arm; repository context is never mislabeled as RAG.
Calls run serially by default to reduce provider rate limiting. Use identical
concurrency for all models when explicitly changing `--workers`.

The measured 30-item pilot is in
[`reports/model_benchmark_analysis.md`](reports/model_benchmark_analysis.md).
On the no-context arm, the shared set contained 23 items (3 defect positives):
20B scored AUC 0.925 and NDCG@10 0.900 versus 0.900 and 0.590 for 120B. The
paired Spearman difference was +0.178 (95% interval [-0.115, 0.516]), so this
pilot does not establish a better model. The 20B RAG arm failed all 30 calls
with provider rate-limit errors; its RAG quality is unavailable. 120B's RAG
arm completed all 30 items, with AUC 0.900, NDCG@10 0.796, and a 447-second
runtime. Repeat with the rate limit resolved and multiple seeds before model
selection.

Prices were checked on 2026-10-27 in
[Groq's model documentation](https://console.groq.com/docs/models):
GPT-OSS 20B costs $0.075 input/$0.30 output per million tokens;
120B costs $0.15 input/$0.60 output. Both support
[structured outputs](https://console.groq.com/docs/structured-outputs).
Rates are explicitly supplied, not hardcoded as universal defaults.
Unknown costs remain null. Provider retries, cached billing, and reasoning-token
accounting may differ from estimates; provider billing is authoritative.
Schema acceptance measures successful validated calls, not raw JSON syntax alone.

Choose a model only after considering ranking quality, uncertainty, failure rate,
cost and latency together. A small pilot or partial run cannot establish a
general best model. Independent human expert ratings and more repositories remain
necessary for strong conclusions.

## 4. Git history and ablations

File enrichment now includes added/deleted lines, total churn, contributors,
recency and repeated bug-fix flags. File and function history use the same
behavioral-fix classifier, excluding documentation/CI-only fixes. Function
history enforces a half-open time window, preventing cutoff commits from appearing
in both context and ground truth. Commit-message classification remains heuristic.

The original context heuristic, its no-history ablation, and an advanced-history
variant are separate methods. Advanced history is shown only when churn has been
measured. The advanced variant jointly adds churn, authors, recency and recurring
fixes; contributor count and recency are hypotheses about risk, not causal facts.

On the 84-item Requests snapshot, adding available history to the basic context
heuristic changed Spearman vs. future changes by +0.050 (95% paired interval
−0.162 to +0.260), which is inconclusive. Adding advanced churn and history
features changed it by −0.157 (−0.256 to −0.067). This suggests the current
advanced recipe hurts this proxy ranking and should not be enabled as a default.
The interval is exploratory, item-level, unadjusted, and sensitive to correlated
functions and commit-message labels. See `reports/history_ablation.json`.

```powershell
python run_history_ablation.py --repo targets/requests --items data/requests_items.json --truth data/requests_ground_truth.json --meta data/requests_dataset_meta.json --output reports/history_ablation.json
```

This writes a separate enriched item copy and comparison report. The original
dataset and cached LLM rankings are preserved. Rebuild the full dataset and rerun
LLM experiments before claiming that these new signals improve LLM rankings.
File renames and authors with multiple identities still limit history accuracy.

## 5. Explainability

Item details show the existing LLM reason/recommendation, observed complexity,
coverage, caller counts, history and RAG neighbors. New panels show the exact
normalized contribution of each context-heuristic feature, plus recommendations
linked to observed evidence. They explicitly distinguish heuristic contributions
from LLM attribution; the system does not invent numerical weights for LLM scores.
Caller counts are syntactic call sites, not a precise distinct-dependent-function
graph. Missing coverage is marked unknown rather than asserted as zero coverage.

## Validation

The full existing suite plus initial regression tests passed (124 tests).
Additional targeted tests cover cutoff non-overlap and bounded embedding batches.
The dashboard JavaScript passed `node --check`. Visual browser verification was
blocked by the browser's local-file URL policy; open the generated HTML locally
for visual review. Benchmarks are measurements, not replacement correctness tests.
