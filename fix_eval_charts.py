"""Patch renderEvaluation in template.html to show only 3 key methods in charts."""
import re

TEMPLATE = "dashboard/template.html"

with open(TEMPLATE, "r", encoding="utf-8") as f:
    content = f.read()

# Locate renderEvaluation function
start_marker = "function renderEvaluation() {"
end_marker = "\nfunction renderReview()"

si = content.index(start_marker)
ei = content.index(end_marker, si)

old_fn = content[si:ei]
print(f"Replacing renderEvaluation ({len(old_fn)} chars)")

new_fn = r"""function renderEvaluation() {
  const view = $("#view-evaluation"), d = ev.defects;
  if (!d) { view.innerHTML = empty("No evaluation yet", "Run the evaluation to compare methods.", ["python run_evaluation.py", "python run_dashboard.py"]); return; }
  // Only the 3 key methods: proposed LLM+RAG, context baseline, static baseline
  const CHART_METHODS = ["llm", "context_heuristic", "static_score"];
  const chartSeries = (fn) => CHART_METHODS.map((n) => ({
    name: n,
    color: (METHOD[n] || {}).color || "var(--muted)",
    dash: (METHOD[n] || {}).dash || "none",
    bold: n === "llm",
    points: fn(n)
  }));
  const METRICS = {
    "precision@10": ["Precision@10 \u00b7 later bug fixes", "defects", "precision@10", "precision@10_ci95", 0],
    "precision@5": ["Precision@5 \u00b7 later bug fixes", "defects", "precision@5", "precision@5_ci95", 0],
    "auc_future_defect": ["ROC-AUC \u00b7 bug-fixed vs not", "defects", "auc_future_defect", "auc_future_defect_ci95", 0],
    "ndcg@10_fix_commits": ["NDCG@10 \u00b7 bug-fix commits", "defects", "ndcg@10_fix_commits", "ndcg@10_fix_commits_ci95", 0],
    "spearman_vs_change_commits": ["Spearman \u03c1 \u00b7 later changes (95% CI)", "defects", "spearman_vs_change_commits", "spearman_vs_change_commits_ci95", -1],
  };
  const hasExp = ev.experts && ev.experts.methods && Object.keys(ev.experts.methods).length;
  if (hasExp) { METRICS.exp = ["Spearman \u03c1 \u00b7 experts (95% CI)", "experts", "spearman", "spearman_ci95", -1]; METRICS.expP = ["Top-10 overlap \u00b7 experts", "experts", "precision@10", "precision@10_ci95", 0]; }
  if (!METRICS[E.metric]) E.metric = "precision@10";
  const [, sec, key, ci, lo] = METRICS[E.metric];
  function dm(n, k) { const r = (d.methods || {})[n]; return r && isNum(r[k]) ? r[k] : null; }
  // Compact 3-method legend with colour-coded labels and the metric value inline
  function chartLegend(metricKey) {
    return `<div class="legend" style="gap:18px;margin-top:10px">${CHART_METHODS.map(n => {
      const m = METHOD[n] || { label: n, color: "var(--muted)", dash: "none" };
      const v = dm(n, metricKey);
      const sw = n === "llm" ? "3.4" : "2";
      const cr = n === "llm" ? "4.5" : "3";
      return `<span class="legend-item" style="gap:6px"><svg width="30" height="12" style="display:inline-block;overflow:visible;vertical-align:middle"><line x1="0" y1="6" x2="30" y2="6" stroke="${m.color}" stroke-width="${sw}" stroke-dasharray="${m.dash !== 'none' ? m.dash : ''}"/><circle cx="15" cy="6" r="${cr}" fill="${m.color}" stroke="var(--plot-bg)" stroke-width="1.5"/></svg><b style="color:${m.color}">${esc(m.label)}</b>${v !== null ? `<span class="mono" style="font-size:.78em;opacity:.75">${metricKey.startsWith('precision') ? 'P@K' : 'AUC'}&thinsp;${fmt(v, 2)}</span>` : ""}</span>`;
    }).join("")}</div>`;
  }
  const table = (s, cols) => `<div class="table-wrap"><table class="data"><thead><tr><th>Method</th>${cols.map(([, l]) => `<th class="r">${l}</th>`).join("")}</tr></thead><tbody>${methodRows(s).map(([n, r]) => { const bestOf = (k) => Math.max(...methodRows(s).map(([, rr]) => isNum(rr[k]) ? rr[k] : -9)); return `<tr><td><span class="legend"><span><i class="dot" style="background:${(METHOD[n] || {}).color}"></i><b style="color:var(--ink)">${esc((METHOD[n] || { label: n }).label)}</b></span></span></td>${cols.map(([k]) => Array.isArray(r[k]) ? `<td class="r num muted">[${fmt(r[k][0], 2)}, ${fmt(r[k][1], 2)}]</td>` : `<td class="r num" style="${isNum(r[k]) && r[k] === bestOf(k) ? "font-weight:700;color:var(--accent-ink)" : ""}">${fmt(r[k])}</td>`).join("")}</tr>`; }).join("")}</tbody></table></div>`;
  view.innerHTML = `
    <div class="card"><div class="card-head"><div><h2>Method comparison</h2><span class="sub">Evaluation on the same ${d.n_items} functions \u00b7 post-cutoff history as ground truth. <b>Bold</b> = best.</span></div></div>
      ${table({methods: Object.fromEntries(["static_score", "context_heuristic", "llm"].map((n) => [n, d.methods[n] || {}]))}, [["precision@5", "Prec@5"], ["precision@10", "Prec@10"], ["auc_future_defect", "ROC-AUC"], ["ndcg@10_fix_commits", "NDCG@10"], ["spearman_vs_fix_commits", "Spearman \u03c1"]])}
      <p class="sub" style="margin-top:10px">Static analysis: composite score (complexity, MI, smells). Context heuristic: adds Git history, callers, coverage. LLM + RAG (proposed): full repository context via RAG retrieval + 3-agent LLM reasoning.</p>
    </div>
    <div class="grid">
      <div class="card span-6"><div class="card-head"><div><h2>Precision@K</h2><span class="sub">Share of top-K functions that were bug-fixed after the cutoff \u2014 <em>higher is better</em>. Base rate ${pct(evalPositives.size / d.n_items)}.</span></div></div>
        ${lineChart({ series: chartSeries((n) => precisionCurve(n, 30)), xMax: 30, xLabel: "K (top-K functions)", yLabel: "Precision", xTicks: [1, 5, 10, 15, 20, 25, 30] })}
        ${chartLegend("precision@10")}</div>
      <div class="card span-6"><div class="card-head"><div><h2>ROC curve</h2><span class="sub">True vs false positive rate \u2014 <em>closer to top-left = better</em>. Separates ${evalPositives.size} bug-fixed from ${d.n_items - evalPositives.size} clean functions.</span></div></div>
        ${lineChart({ series: chartSeries(rocCurve), xMax: 1, xLabel: "False positive rate", yLabel: "True positive rate", diag: true, xTicks: [0, .25, .5, .75, 1] })}
        ${chartLegend("auc_future_defect")}</div>
    </div>
    <div class="card"><div class="card-head"><div><h2>Metric bar chart</h2><span class="sub">${d.n_items} functions \u00b7 ${d.n_positive} bug-fixed positives. Whiskers = 95% bootstrap CI. Select metric below.</span></div>
      <select id="metric" class="field" aria-label="Metric">${Object.entries(METRICS).map(([k, v]) => `<option value="${k}" ${k === E.metric ? "selected" : ""}>${v[0]}</option>`).join("")}</select></div>
      ${barChart(["llm", "context_heuristic", "static_score"].map(n => [n, ((ev[sec] || {}).methods || {})[n] || {}]), key, ci, lo)}</div>`;
  $("#metric").addEventListener("change", (e) => { E.metric = e.target.value; store.set("tdr.eval", E); renderEvaluation(); });
}"""

new_content = content[:si] + new_fn + content[ei:]
print(f"New renderEvaluation: {len(new_fn)} chars")

with open(TEMPLATE, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Done — template.html patched.")
