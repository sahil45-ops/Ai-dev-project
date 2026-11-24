"""
Streamlit dashboard — explore rankings, collect expert ratings, view results.

    streamlit run dashboard/app.py

Tabs:
  Overview      dataset summary for the selected repository
  Rankings      static vs. LLM ranking side by side, with the LLM's reasoning
  Expert rating blind rating form for human raters (no method ranks shown)
  Evaluation    Phase 5 metrics per method
  Refactoring   Phase 6 branches, CI outcome and diffs
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import plotly.express as px
import streamlit as st

import config
from agents.prompts import coverage_is_known, read_function_source, render_item
from evaluation.evaluate import METHOD_LABELS, collect_methods
from evaluation.expert import load_raters, save_rater
from evaluation.explanations import (
    ADVANCED,
    BASE,
    HISTORY,
    evidence_for,
    explain_heuristic,
)


def load(path: str, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


st.set_page_config(page_title="Tech Debt Prioritisation", layout="wide")
st.title("Intelligent Technical Debt & Refactoring")

available_repos = sorted(name.removesuffix("_items.json") for name in os.listdir(config.DATA_DIR)
                         if name.endswith("_items.json"))
if not available_repos:
    st.warning("No analysed repositories are available.")
    st.stop()
repo = st.sidebar.selectbox("Repository", available_repos)
paths = config.dataset_paths(repo)
items = load(paths["items"], [])
history_items = load(os.path.join(config.REPORTS_DIR, f"{repo}_history_ablation_items.json"), []) or []
history_by_id = {x["id"]: x for x in history_items}
items = [{**item, **{k: history_by_id[item["id"]][k] for k in
         ("code_churn", "recurring_bug_file") if item["id"] in history_by_id
         and k in history_by_id[item["id"]]}} for item in items]
history_evaluation = load(os.path.join(config.REPORTS_DIR, f"{repo}_history_ablation.json"), {})
meta = load(paths["meta"], {})
llm = load(paths["llm_rankings"], [])
evaluation = load(paths["evaluation"], {})
refactoring = load(paths["refactoring"], {})
by_id = {x["id"]: x for x in items}

if not items:
    st.warning(f"No dataset found at {paths['items']}. Run `python run_dataset.py` first.")
    st.stop()

if st.query_params.get("review") == "1":
    from dashboard.review_form import render_review

    render_review(items, load(paths["expert_sample"], []), config.EXPERT_DIR)
    st.stop()

tab_over, tab_rank, tab_expert, tab_eval, tab_ref = st.tabs(
    ["Overview", "Rankings", "Expert rating", "Evaluation", "Refactoring"])

with tab_over:
    st.caption("Research prototype · Results reflect saved experiments. Human review is required before merging patches.")
    failures = sum("error" in r for r in llm)
    if failures:
        st.warning(f"{failures} LLM items failed. Rankings and comparisons are incomplete; inspect the run status before presenting.")
    if not evaluation.get("experts", {}).get("methods"):
        st.info("Independent expert evaluation is pending. Current quantitative results use future commit history as a proxy.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Functions indexed", meta.get("functions_indexed", "–"))
    c2.metric("Debt candidates", len(items))
    c3.metric("LLM-ranked", sum(1 for r in llm if "llm_rank" in r))
    c4.metric("Snapshot", f"{meta.get('snapshot_sha', '')[:8]} @ {meta.get('as_of', '')[:10]}")
    df = pd.DataFrame(items)
    st.plotly_chart(px.scatter(
        df, x="complexity", y="static_score", size="loc", color="func_fix_commits",
        hover_name="id", labels={"func_fix_commits": "past fixes"},
        title="Debt candidates: complexity vs. static score (size = LOC)"),
        use_container_width=True)
    st.caption("Phase 1 findings by tool: " + ", ".join(
        f"{k}: {v}" for k, v in meta.get("phase1_by_tool", {}).items()))

with tab_rank:
    rows = []
    llm_by_id = {r["id"]: r for r in llm if "llm_rank" in r}
    for x in items:
        r = llm_by_id.get(x["id"], {})
        rows.append({"id": x["id"], "static_rank": x["static_rank"],
                     "llm_rank": r.get("llm_rank"), "priority": r.get("priority"),
                     "llm_score": r.get("priority_score"), "complexity": x["complexity"],
                     "callers": x["callers"], "past_fixes": x["func_fix_commits"],
                     "coverage": ("not measured" if not coverage_is_known(x)
                                  else x["function_coverage"]),
                     "code_churn": x.get("code_churn"),
                     "contributors": x.get("commit_authors")})
    table = pd.DataFrame(rows).sort_values("llm_rank" if llm_by_id else "static_rank")
    st.dataframe(table, use_container_width=True, hide_index=True)
    if not llm_by_id:
        st.info("No LLM ranking yet — run `python run_prioritisation.py`.")
    choice = st.selectbox("Inspect item", table["id"].tolist())
    if choice:
        r = llm_by_id.get(choice)
        if r:
            st.markdown(f"**{r['priority']} ({r['priority_score']}/100)** — {r['reason']}")
            st.markdown(f"**Recommended action:** {r.get('action', '')}")
            if r.get("analysis_summary"):
                st.markdown(f"**Analysis:** {r['analysis_summary']}")
        explanation = explain_heuristic(items).get(choice, {})
        if explanation:
            st.markdown(f"**Context heuristic score:** {explanation['score']:.3f} "
                        "(this is an auditable baseline explanation, not an LLM attribution)")
            st.dataframe(pd.DataFrame(explanation["factors"]), hide_index=True,
                         use_container_width=True)
            st.markdown("**Evidence-based recommendations**")
            for recommendation in evidence_for(by_id[choice])["recommendations"]:
                st.markdown(f"- {recommendation}")
        if by_id[choice].get("code_churn") is not None:
            advanced = explain_heuristic(items, BASE + HISTORY + ADVANCED).get(choice, {})
            st.markdown(f"**Advanced Git heuristic score:** {advanced.get('score', 0):.3f} "
                        "(experimental, evaluated separately)")
            st.dataframe(pd.DataFrame([
                factor for factor in advanced.get("factors", [])
                if factor["feature"] in {"code_churn", "commit_authors",
                                         "recent_activity", "recurring_bug_file"}]),
                hide_index=True, use_container_width=True)
        st.markdown("**RAG evidence**")
        retrieval = load(os.path.join(config.DATA_DIR, f"{repo}_llm_rankings_retrieval.json"), {}) or {}
        st.code(retrieval.get(choice, "No retrieval record saved."))
        st.code(read_function_source(by_id[choice]), language="python")

with tab_expert:
    st.info("For independent ratings, open /?review=1 before viewing any algorithm results. This tab is for reviewing existing work.")
    sample = load(paths["expert_sample"], [])
    if not sample:
        st.info("No expert sample yet — run `python run_evaluation.py --make-expert-sample`.")
    else:
        st.markdown(
            "Rate **how urgently each function should be refactored** on a 1–10 scale "
            "(10 = fix first). Judge from the metrics, context and code; method rankings are "
            "deliberately hidden. Rate independently — do not discuss with other raters.")
        rater = st.text_input("Your name (one file per rater)")
        existing = next((r for r in load_raters(config.EXPERT_DIR) if r["rater"] == rater), None)
        prior = existing["scores"] if existing else {}
        with st.form("expert_form"):
            scores, notes = {}, {}
            for n, item_id in enumerate(sample, 1):
                item = by_id.get(item_id)
                if not item:
                    continue
                with st.expander(f"{n}. `{item['function']}` — `{item['file']}`"):
                    st.text(render_item(item, read_function_source(item), with_context=True)
                            .split("Source code:")[0])
                    st.code(read_function_source(item), language="python")
                scores[item_id] = st.selectbox(f"Urgency for item {n}", range(1, 11),
                                               index=int(prior[item_id]) - 1 if item_id in prior else None,
                                               key=f"s_{item_id}")
                notes[item_id] = st.text_input("Note (optional)", key=f"n_{item_id}",
                                               value=(existing or {}).get("notes", {}).get(item_id, ""))
            if st.form_submit_button("Save my ratings"):
                if not rater.strip() or any(v is None for v in scores.values()):
                    st.error("Enter your name and explicitly score every item first.")
                else:
                    path = save_rater(config.EXPERT_DIR, rater, scores, notes)
                    st.success(f"Saved {len(scores)} ratings to {path}. "
                               "Re-run `python run_evaluation.py` to update results.")
        raters = load_raters(config.EXPERT_DIR)
        st.caption(f"Raters so far: {', '.join(r['rater'] for r in raters) or 'none'}")

with tab_eval:
    if not evaluation:
        st.info("No evaluation yet — run `python run_evaluation.py`.")
    else:
        comparison_labels = {
            "static_score": "Static analysis",
            "context_heuristic": "Context heuristic",
            "llm": "LLM + RAG",
            "llm_nocontext": "LLM without context",
        }
        st.subheader("Four-method comparison")
        saved_methods = evaluation.get("defects", {}).get("methods", {})
        comparison = pd.DataFrame({
            label: saved_methods.get(key, {}) for key, label in comparison_labels.items()
        }).T
        columns = ["precision@10", "auc_future_defect", "ndcg@10_fix_commits",
                   "spearman_vs_fix_commits", "spearman_vs_change_commits"]
        st.dataframe(comparison.reindex(columns=columns), use_container_width=True)
        st.caption("Saved results against post-cutoff history. The Git context method uses "
                   "the existing context heuristic, which also includes callers and test coverage.")
        all_methods = collect_methods(
            items, llm, load(paths["llm_rankings_nocontext"], []) or [])
        metric_columns = st.columns(2)
        for column, key, title in zip(metric_columns, ("static_score", "context_heuristic"),
                                      ("Static analysis: top 10", "Context heuristic: top 10")):
            ranking = all_methods.get(key, {}).get("order", [])
            with column:
                st.markdown(f"**{title}**")
                st.dataframe(pd.DataFrame([
                    {"rank": n, "item": item_id, "score": all_methods[key]["scores"][item_id]}
                    for n, item_id in enumerate(ranking[:10], 1)]),
                    hide_index=True, use_container_width=True)
        for column, key, title in zip(metric_columns, ("llm", "llm_nocontext"),
                                      ("LLM + RAG: top 10", "LLM without context: top 10")):
            ranking = all_methods.get(key, {}).get("order", [])
            with column:
                st.markdown(f"**{title}**")
                st.dataframe(pd.DataFrame([
                    {"rank": n, "item": item_id, "score": all_methods[key]["scores"][item_id]}
                    for n, item_id in enumerate(ranking[:10], 1)]),
                    hide_index=True, use_container_width=True)
        if "context_advanced" in all_methods:
            ranking = all_methods["context_advanced"]["order"][:10]
            st.markdown("**Advanced Git context: top 10 (separate ablation)**")
            st.dataframe(pd.DataFrame([
                {"rank": n, "item": item_id,
                 "score": all_methods["context_advanced"]["scores"][item_id]}
                for n, item_id in enumerate(ranking, 1)]),
                hide_index=True, use_container_width=True)
        if history_evaluation:
            st.subheader("Advanced Git history ablation")
            st.caption("Separate evaluation across all debt candidates; its item set may differ from the core four-method comparison.")
            history_methods = history_evaluation.get("evaluation", {})
            history_frame = pd.DataFrame(history_methods.get("methods", {})).T
            history_frame.index = [METHOD_LABELS.get(i, i) for i in history_frame.index]
            st.dataframe(history_frame, use_container_width=True)
            if history_evaluation.get("paired"):
                st.dataframe(pd.DataFrame(history_evaluation["paired"]),
                             use_container_width=True, hide_index=True)

        def section_frame(section: dict) -> pd.DataFrame:
            frame = pd.DataFrame(section.get("methods", {})).T
            frame.index = [METHOD_LABELS.get(i, i) for i in frame.index]
            return frame

        if evaluation.get("experts", {}).get("methods"):
            st.subheader("Against expert rankings")
            ag = evaluation.get("expert_agreement", {})
            st.caption(f"Raters: {', '.join(ag.get('raters', []))} — Kendall's W "
                       f"{ag.get('kendalls_w')}, mean pairwise ρ {ag.get('mean_pairwise_spearman')}")
            ef = section_frame(evaluation["experts"])
            st.dataframe(ef, use_container_width=True)
            st.plotly_chart(px.bar(ef.reset_index(), x="index", y="spearman",
                                   labels={"index": "method", "spearman": "Spearman ρ vs experts"}),
                            use_container_width=True)
        st.subheader("Against post-cutoff history")
        d = evaluation["defects"]
        st.caption(f"{d['n_items']} items, {d['n_positive']} with a bug fix after the cutoff")
        df_def = section_frame(d)
        st.dataframe(df_def, use_container_width=True)
        metric = st.selectbox("Metric", [c for c in df_def.columns if not c.endswith("ci95")])
        ci_col = metric + "_ci95"
        if ci_col in df_def:
            df_def["ci_low"] = df_def.apply(
                lambda row: row[ci_col][0] if isinstance(row.get(ci_col), list) else None, axis=1)
            df_def["ci_high"] = df_def.apply(
                lambda row: row[ci_col][1] if isinstance(row.get(ci_col), list) else None, axis=1)
            df_def["ci_upper_error"] = df_def["ci_high"] - df_def[metric]
            df_def["ci_lower_error"] = df_def[metric] - df_def["ci_low"]
        st.plotly_chart(px.bar(df_def.reset_index(), x="index", y=metric,
                               error_y="ci_upper_error" if "ci_upper_error" in df_def else None,
                               error_y_minus="ci_lower_error" if "ci_lower_error" in df_def else None,
                               labels={"index": "method"}), use_container_width=True)
        if evaluation.get("paired"):
            st.subheader("Paired bootstrap comparisons")
            st.dataframe(pd.DataFrame(evaluation["paired"]), use_container_width=True,
                         hide_index=True)

with tab_ref:
    if not refactoring:
        st.info("No refactoring results yet — run `python run_refactoring.py`.")
    else:
        s = refactoring["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Success rate", f"{s['success_rate']:.0%}")
        c2.metric("Test pass rate", f"{s['test_pass_rate']:.0%}")
        c3.metric("Avg CC reduction", s["mean_complexity_reduction_target"])
        c4.metric("Avg Ruff Δ", s["mean_ruff_delta"])
        for r in refactoring["results"]:
            label = "✅" if r["success"] else "❌"
            with st.expander(f"{label} `{r['function']}` — `{r['branch']}`"):
                st.json({k: r.get(k) for k in ("before", "after", "tests", "new_failures",
                                               "complexity_reduction_target", "ruff_delta")})
                diff_path = os.path.join(ROOT, "refactoring", "branches",
                                         r["branch"].split("/", 1)[1], "patch.diff")
                if os.path.isfile(diff_path):
                    with open(diff_path, encoding="utf-8") as fh:
                        st.code(fh.read(), language="diff")
