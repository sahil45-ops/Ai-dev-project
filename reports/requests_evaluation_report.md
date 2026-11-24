# Evaluation report — requests

_Generated 2026-11-24T16:37:01+00:00_

## Setup

- Snapshot: `72eccc8d` (last commit before 2024-01-01); context history from 2019-01-01.

- Ground-truth window: 2024-01-01 → 2026-10-06.

- Candidate debt items: 71 functions; 5 received a bug-fix commit after the cutoff.

- Methods compared: Static analysis (composite score), Static analysis (complexity only), Context heuristic (no LLM), Context heuristic without Git history, LLM + RAG context (proposed).

### Incomplete experiments

- llm_rankings: 71/84 successful items (13 errors); using partial results
- llm_rankings_nocontext: no successful items; excluded

## 1. Against expert rankings

_No expert ratings yet — collect them in the dashboard's Expert ranking tab._

## 2. Against post-cutoff defect history

Precision@K = share of a method's top-K items that received a bug-fix commit after the cutoff. AUC = probability that a defect-fixed item outranks a non-fixed one.

| Method | precision@5 | precision@10 | precision@20 | auc_future_defect | spearman_vs_fix_commits | ndcg@10_fix_commits |
|---|---|---|---|---|---|---|
| Static analysis (composite score) | 0.200 | 0.300 | 0.200 | 0.815 | 0.277 | 0.468 |
| Static analysis (complexity only) | 0.400 | 0.300 | 0.150 | 0.811 | 0.278 | 0.611 |
| Context heuristic (no LLM) | 0.600 | 0.300 | 0.250 | 0.912 | 0.366 | 0.667 |
| Context heuristic without Git history | 0.400 | 0.200 | 0.250 | 0.879 | 0.338 | 0.401 |
| LLM + RAG context (proposed) | 0.200 | 0.100 | 0.050 | 0.494 | -0.006 | 0.076 |

## 3. Against post-cutoff change-proneness

| Method | spearman_vs_change_commits | spearman_vs_change_commits_ci95 | ndcg@10_change_commits |
|---|---|---|---|
| Static analysis (composite score) | 0.413 | [0.179, 0.599] | 0.546 |
| Static analysis (complexity only) | 0.507 | [0.307, 0.660] | 0.616 |
| Context heuristic (no LLM) | 0.424 | [0.212, 0.594] | 0.549 |
| Context heuristic without Git history | 0.363 | [0.150, 0.549] | 0.301 |
| LLM + RAG context (proposed) | 0.139 | [-0.095, 0.350] | 0.221 |

## 4. Paired comparisons (bootstrap 95% CI of Δρ)

| Comparison | Ground truth | Δρ | 95% CI | Significant |
|---|---|---|---|---|

| llm vs static_score | changes | -0.275 | [-0.570, 0.035] | no |

| llm vs complexity | changes | -0.369 | [-0.637, -0.070] | yes |

| llm vs context_heuristic | changes | -0.285 | [-0.551, 0.005] | no |

| context_heuristic vs context_no_history | changes | 0.050 | [-0.162, 0.260] | no |

## 5. Refactoring + CI (Phase 6)

- Attempted: 5; succeeded (no new test failures and lower complexity): 0 → success rate 0%

- Test pass rate (no regressions): 0%

- Mean complexity reduction (target function): 0.670; (whole replaced block incl. helpers): 0.670

- Mean Ruff findings Δ per file: 0.330; duplicated-lines Δ: -2.670

## Research contribution statement

> LLM-based prioritisation of technical debt, augmented with repository context retrieved via RAG, achieves a Spearman correlation of 0.139 with post-cutoff change history — compared to 0.413 for static analysis alone — and a Precision@10 of 0.100 vs 0.300. LLM-generated refactoring patches pass CI validation at a rate of 0%, with an average cyclomatic complexity reduction of 0.670 points per refactored function.

## Threats to validity

- Defect labels come from commit-message keywords mapped to changed functions; some fixes are missed or mislabelled.

- Few post-cutoff fixes in a mature library make defect-based metrics noisy — read the confidence intervals, not just point estimates.

- Independent expert ratings are pending unless explicitly reported above; historical fixes are a proxy for debt urgency.

- One repository and one snapshot limit generalisation. The model may have seen public repository code during training.

- The context heuristic uses arbitrary equal weights; it is a sanity baseline, not a tuned model.

- LLM output can vary between runs; responses are cached so the reported run is reproducible from data/llm_cache.

- Matched comparisons require the same model, agent mode, item set and prompt rubric. Report incomplete runs explicitly.
