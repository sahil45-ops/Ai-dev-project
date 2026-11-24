# LLM comparison pilot

## Protocol

This is a seeded 30-item pilot (seed 7), with one repeat and single-call
prioritization. The two Groq-hosted OpenAI GPT-OSS models received the same
frozen item data and, within each arm, identical prompts and retrieved context.
Quality metrics use the shared successfully ranked items for the two models in
that arm. This one-repeat sample is exploratory; it cannot establish a general
best model or stable cost/latency estimates.

## Results

| Model | Context | Shared items | Defect positives | ROC-AUC | NDCG@10 | Precision@10 | Spearman vs. future changes | Time | Cost | Item failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-OSS 20B | None | 23 | 3 | 0.925 | 0.900 | 0.30 | 0.421 | 154.6 s | $0.0039 | 23.3% |
| GPT-OSS 120B | None | 23 | 3 | 0.900 | 0.590 | 0.30 | 0.243 | 172.0 s | $0.0094 | 0% |
| GPT-OSS 20B | RAG | 0 | 0 | — | — | — | — | 3.0 s | $0.0000 | 100% |
| GPT-OSS 120B | RAG | 30 | 3 | 0.900 | 0.796 | 0.30 | 0.289 | 447.1 s | $0.0148 | 0% |

The no-context paired Spearman difference (20B minus 120B, against future
change commits) was +0.178, with a bootstrap 95% interval of [-0.115, 0.516]
(23 shared items). The interval includes zero. The RAG arm cannot be compared
because the 20B arm returned no valid items. The 20B RAG failures occurred
quickly and were recorded as call/schema failures; do not interpret its zero
reported cost as evidence that the arm was inexpensive to operate.

## Interpretation

On the available no-context sample, 20B had higher observed AUC, NDCG@10, and
Spearman, with equal Precision@10, but lower reliability and only a small
latency advantage. The paired interval is wide and does not support choosing
20B as the better model. In the RAG arm, 120B ranked all 30 items successfully,
but was substantially slower and cost more in this pilot. The practical result
is a reliability/quality/cost trade-off, not a declared winner.

Reported costs use the prices recorded in
[`benchmark_prices.json`](benchmark_prices.json). Prices may change; check
[Groq model documentation](https://console.groq.com/docs/models) before a new
run. Failure attempts may have incurred provider-side usage not reflected in
the client token totals. For a model selection decision, repeat the paired
experiment over multiple seeds, increase the sample, and resolve the 20B RAG
failure before comparing the RAG quality scores.

Full item-level outputs, token counts, validation results, timings, and the
matched-set bootstrap results are in
[`model_benchmark_pilot.json`](model_benchmark_pilot.json). The interrupted
four-worker attempt is kept separately in
[`model_benchmark_rate_limited_30.json`](model_benchmark_rate_limited_30.json)
and is excluded from these results.
