"""Generate docs/MODEL_CARD.md and the README results block from the result files. Numbers are never typed by hand."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.config import resolve_path
from pipeline.notebook_support import seed_summary

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
FRAUD_COLS = ["recall_at_fpr", "realised_fpr", "cost", "cost_flag_none", "brier", "ece"]   # plus every alert-budget column


def load_tables(out_dir, reports_dir) -> dict:
    out_dir, reports_dir = Path(out_dir), Path(reports_dir)
    tables = {"fraud": pd.read_csv(out_dir / "results_fraud.csv"), "account": pd.read_csv(out_dir / "results_account.csv")}
    if (out_dir / "drift_summary.csv").exists():
        tables["drift_summary"] = pd.read_csv(out_dir / "drift_summary.csv")
    for name, file in (("weekly", "results_weekly.csv"), ("calibration", "calibration.csv")):
        if (out_dir / file).exists():
            tables[name] = pd.read_csv(out_dir / file)
    if (out_dir / "latency.csv").exists():
        tables["latency"] = pd.read_csv(out_dir / "latency.csv")
        tables["latency_meta"] = json.loads((out_dir / "latency_meta.json").read_text(encoding="utf-8"))
    if (out_dir / "results_sampling_rule.csv").exists():
        tables["sampling_rule"] = pd.read_csv(out_dir / "results_sampling_rule.csv")
    if (reports_dir / "explain_metrics.json").exists():
        tables["explain"] = json.loads((reports_dir / "explain_metrics.json").read_text(encoding="utf-8"))
    return tables


def check_single_config(tables: dict) -> str:
    hashes = set()
    for name, t in tables.items():
        if name == "latency":                                # this table carries its hash in latency_meta
            continue
        if isinstance(t, pd.DataFrame):
            if "config_hash" not in t:
                raise ValueError(f"table {name} has no config_hash column: re-run the step that wrote it")
            hashes |= set(t["config_hash"].unique())
        else:
            hashes.add(t["config_hash"])
    if len(hashes) != 1:
        raise ValueError(f"mixed config hashes {sorted(hashes)}: re-run all steps with one configuration")
    return hashes.pop()


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def _fraud_columns(fraud: pd.DataFrame) -> list:
    budgets = sorted((c for c in fraud.columns if "@" in c), key=lambda c: (float(c.split("@")[1]), c.split("@")[0]))
    return [c for c in FRAUD_COLS[:2] if c in fraud] + budgets + [c for c in FRAUD_COLS[2:] if c in fraud]


def results_markdown(fraud: pd.DataFrame, account: pd.DataFrame) -> str:
    f = seed_summary(fraud, ["pr_auc"])
    others = fraud.groupby("model")[_fraud_columns(fraud)].mean().reset_index()
    f = f.merge(others, on="model")
    f["pr_auc (mean, min-max over seeds)"] = [f"{m:.3f} ({lo:.3f}-{hi:.3f})" for m, lo, hi in
                                              zip(f["pr_auc_mean"], f["pr_auc_min"], f["pr_auc_max"])]
    cols = ["model", "pr_auc (mean, min-max over seeds)"]
    if {"pr_auc_lo", "pr_auc_hi"} <= set(fraud.columns):
        ci = fraud.groupby("model")[["pr_auc_lo", "pr_auc_hi"]].mean()
        f["bootstrap 95% CI (mean of per-seed bounds)"] = [f"{ci.loc[m, 'pr_auc_lo']:.3f}-{ci.loc[m, 'pr_auc_hi']:.3f}"
                                                           for m in f["model"]]
        cols.append("bootstrap 95% CI (mean of per-seed bounds)")
    f = f[cols + [c for c in others.columns if c != "model"]]
    a = seed_summary(account, ["pr_auc"])[["model", "pr_auc_mean", "pr_auc_min", "pr_auc_max"]]
    return ("Transaction fraud, test period:\n\n" + _table(f) + "\n\nAccount risk (proxy label), test users:\n\n" + _table(a))


def _mean_range(table, model, col="pr_auc"):
    v = table.loc[table["model"] == model, col]
    return (float(v.mean()), float(v.max() - v.min())) if len(v) else None


def _compare(table, a, b, what):
    """One computed comparison of two models' mean PR-AUC; a gap inside the seed range is not presented as an effect."""
    ma, mb = _mean_range(table, a), _mean_range(table, b)
    if ma is None or mb is None:
        return None
    gap = ma[0] - mb[0]
    note = " (within the seed range, so not an effect this run can claim)" if abs(gap) <= max(ma[1], mb[1]) else ""
    return f"- {what}: {a} versus {b}, {gap:+.3f} PR-AUC{note}."


def findings_markdown(fraud: pd.DataFrame, account=None, drift=None) -> str:
    """Every number and every direction below is computed from the result tables; nothing is typed by hand."""
    means = {m: float(g["pr_auc"].mean()) for m, g in fraud.groupby("model")}
    lines = []
    best = max(means, key=means.get)
    lines.append(f"- Highest mean test PR-AUC: {best} ({means[best]:.3f}).")
    graph = [m for m in ("fraud_only", "multitask") if m in means]
    tabular = [m for m in ("naive_tabular", "history_tabular", "history_stable_tabular") if m in means]
    if graph and tabular:
        g, t = max(graph, key=means.get), max(tabular, key=means.get)
        line = _compare(fraud, g, t, "Best graph model against best tabular model")
        lines.append(f"- The best graph model is {g} ({means[g]:.3f}) and the best tabular model is {t} ({means[t]:.3f}).")
        lines.append(line)
    for a, b, what in (("fraud_only", "no_graph", "Graph structure"), ("multitask", "fraud_only", "The second (account) head"),
                       ("history_tabular", "naive_tabular", "History features as specified"),
                       ("history_stable_tabular", "naive_tabular", "History features without cumulative counts")):
        line = _compare(fraud, a, b, what)
        if line:
            lines.append(line)
    if "ece" in fraud:
        ece = {m: float(g["ece"].mean()) for m, g in fraud.groupby("model")}
        lo, hi = min(ece, key=ece.get), max(ece, key=ece.get)
        lines.append(f"- Calibration (ECE, lower is better) runs from {ece[lo]:.3f} ({lo}) to {ece[hi]:.3f} ({hi}).")
    if account is not None and len(account):
        acc = {m: float(g["pr_auc"].mean()) for m, g in account.groupby("model")}
        top = max(acc, key=acc.get)
        lines.append(f"- Account risk: highest mean PR-AUC is {top} ({acc[top]:.3f}); "
                     + ", ".join(f"{m} {v:.3f}" for m, v in sorted(acc.items(), key=lambda kv: -kv[1]) if m != top) + ".")
    if drift is not None and len(drift):
        for kind, g in drift.groupby("kind"):
            s = g.set_index("schedule")
            if {"static", "retrain"} <= set(s.index):
                lines.append(f"- Drift, {kind}: mean weekly PR-AUC {s.loc['static', 'mean_pr_auc']:.3f} when trained once and "
                             f"{s.loc['retrain', 'mean_pr_auc']:.3f} with periodic retraining; trend per week "
                             f"{s.loc['static', 'slope_per_week']:+.4f} (static) and {s.loc['retrain', 'slope_per_week']:+.4f} (retrained).")
    lines.append("- Whether dropping the cumulative counts explains any gap between history_tabular and history_stable_tabular is a "
                 "hypothesis: such counts only grow, so their distribution can differ between the train and test periods, but this "
                 "pipeline does not test that directly. Likewise, any reason given for a graph model doing better or worse than "
                 "a tabular one is a hypothesis unless an ablation above tested it.")
    return "\n".join(lines)


def build_model_card(out_dir, reports_dir, fixture=False) -> str:
    tables = load_tables(out_dir, reports_dir)
    cfg_hash = check_single_config(tables)
    fraud = tables["fraud"]
    flagged = [name for name, t in tables.items()
               if (bool(t["fixture"].any()) if isinstance(t, pd.DataFrame) and "fixture" in t else
                   (isinstance(t, dict) and bool(t.get("fixture"))))]
    if flagged and not fixture:
        raise ValueError(f"these results are fixture results (generated data): {flagged}; re-run on real data or build the "
                         "card with fixture=True")
    parts = ["# Model card: unified risk engine\n"]
    if fixture:
        parts.append("> **FIXTURE RESULTS: generated data, not findings.** Nothing below describes real fraud.\n")
    parts.append(f"Configuration hash: `{cfg_hash}`. Seeds: {sorted(fraud['seed'].unique().tolist())}. "
                 f"Test transactions: {int(fraud['n_test'].iloc[0])}; fraud prevalence {fraud['prevalence'].iloc[0]:.3f}.\n")
    parts.append("## Intended use\n\nA research prototype that ranks transactions by fraud risk and users by an account-risk proxy, "
                 "to study whether a temporal graph model adds value over strong tabular baselines. Not for production decisions, "
                 "credit, or any decision about a person.\n")
    parts.append("## Data\n\nIEEE-CIS Fraud Detection (Kaggle; Vesta Corporation): card-not-present e-commerce transactions with "
                 "anonymised features. Users are an approximation built "
                 "from card and address fields, so \"user\" is a proxy for a real account holder. Test-set size and prevalence "
                 "are given above.\n")
    parts.append("## Leakage controls\n\n- Chronological split: train, validation, test, separated by an embargo so late labels cannot cross a boundary.\n"
                 "- Users are split into disjoint train, validation and test groups; the account label uses history up to the end of training only.\n"
                 "- Feature statistics are fitted on the train period only; history features are strictly causal.\n"
                 "- Neighbour sampling only reaches transactions that arrived no later than the seed (equal timestamps are ordered by transaction ID).\n"
                 "- The history-aware baseline gets hand-built history features (counts, amounts and recency per user, email domain and device, from strictly earlier transactions); the graph model gets the same per-transaction features and has to learn history through message passing.\n")
    parts.append("## Results\n\n" + results_markdown(fraud, tables["account"]) + "\n")
    parts.append("## Findings\n\n" + findings_markdown(fraud, tables["account"], tables.get("drift_summary")) + "\n")
    if "pr_auc_boot_kept" in fraud:
        parts.append(f"Bootstrap resamples without both classes are dropped; at least {fraud['pr_auc_boot_kept'].min():.2f} of the "
                     "resamples were kept for every row.\n")
    parts.append("Cost uses an assumed review cost per false alert; compare it with `cost_flag_none`, the cost of flagging nothing. "
                 "Differences smaller than the bootstrap interval or the range over seeds are not evidence of a better model.\n")
    parts.append("## Drift\n\n" + (_table(tables["drift_summary"].drop(columns=["config_hash", "fixture"], errors="ignore"))
                                   if "drift_summary" in tables else "The drift study was not run.") + "\n")
    e = tables.get("explain")
    parts.append("## Explanations\n\n" + (
        f"Mode `{e['mode']}`. Stability (top-k overlap across runs): {e['stability']:.2f}. Deleting the top-k features of the reported ({e.get('fidelity_method', e['mode'])}) importances moves the "
        f"logit by {e['fidelity_top']:.2f} on average, against {e['fidelity_random']:.2f} for random features"
        + (f"; PyG fidelity+ {e['pyg_fidelity_plus']:.2f}, fidelity- {e['pyg_fidelity_minus']:.2f}, characterization "
           f"{e['pyg_characterization']:.2f}" if e.get("pyg_fidelity_plus") is not None else "")
        + ". Features are anonymised; explanations describe the model and are not causal."
        if e else "Explanations have not been generated.") + "\n")
    lat = tables.get("latency")
    serving = (latency_markdown(lat, tables["latency_meta"]) if lat is not None else "The latency benchmark was not run.")
    rules = tables.get("sampling_rule")
    if rules is not None:
        serving += ("\n\nTest PR-AUC of the served model under each neighbour rule (the headline results use `uniform`, serving uses "
                    "`last`; `uniform` samples at random, so its number is a single draw):\n\n"
                    + _table(rules[["rule", "pr_auc"]]))
    parts.append("## Serving and latency\n\n" + serving + "\n")
    parts.append("## Limitations\n\n- One dataset, one split in time; no claim about other portfolios or later periods.\n"
                 "- The account label is a proxy derived from fraud history, not a ground-truth account-takeover label.\n"
                 "- Anonymised features limit what an explanation can say.\n- Models trained with focal loss are typically poorly "
                 "calibrated (see the ECE and Brier columns), so their scores rank transactions but are not probabilities; thresholds "
                 "are chosen on the validation period.\n- Three seeds give a range, not a confidence interval.\n"
                 "- Label-based history features (a user's past confirmed frauds) are excluded from the baseline, so a production "
                 "system could do better.\n- Nothing here is measured on real data unless the Kaggle files were supplied: check the "
                 "banner at the top of this card.\n"
                 "- The review cost is an assumption.\n- Any explanation of *why* a model wins is a hypothesis unless an ablation tested it.\n")
    return "\n".join(parts)


def splice_block(text: str, name: str, block: str) -> str:
    start, end = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    if start not in text or end not in text:
        raise ValueError(f"missing the {start} / {end} markers ({name} block)")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{block}\n{end}{tail}"


def splice_readme(readme: str, block: str) -> str:
    return splice_block(readme, "RESULTS", block)


def latency_markdown(df: pd.DataFrame, meta: dict) -> str:
    cols = ["model", "device", "mode", "batch_size", "concurrency", "p50_ms", "p95_ms", "p99_ms", "throughput_rps"]
    d = df[df["mode"].isin(["inprocess", "http"])][cols]
    p = meta["parity"]
    parity = (f"Parity check passed: online and offline scores agree (max absolute difference {p['max_abs_diff']:.1e} over {p['n_rows']} transactions)."
              if p["checked"] else f"Parity with the offline scores was not checked ({p['reason']}).")
    d = d.assign(**{c: d[c].astype(float).map("{:.2f}".format) for c in cols[5:]})
    gpu = meta["hardware"].get("gpu") or "none"
    return (_table(d) + f"\n\n{parity} Latencies are per call, in milliseconds, measured on one laptop (GPU: {gpu}; the "
            f"'cpu' rows run on the CPU). The {meta['budget_ms']:g} ms budget is illustrative, not a requirement. One machine "
            "replaying historical data is not a production claim.")


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate the model card and update the README results block.")
    p.add_argument("--out", default="plots")
    p.add_argument("--reports", default="reports")
    p.add_argument("--card", default="docs/MODEL_CARD.md")
    p.add_argument("--readme", default=None, help="README path to update; omit to skip")
    p.add_argument("--fixture", action="store_true", help="the tables come from generated data")
    a = p.parse_args(argv)
    card = build_model_card(resolve_path(a.out), resolve_path(a.reports), fixture=a.fixture)
    card_path = resolve_path(a.card)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card, encoding="utf-8")
    print("wrote", card_path)
    if a.readme:
        tables = load_tables(resolve_path(a.out), resolve_path(a.reports))
        banner = "> **FIXTURE RESULTS: not findings.**\n\n" if a.fixture else ""
        path = resolve_path(a.readme)
        path.write_text(splice_readme(path.read_text(encoding="utf-8"),
                                      banner + results_markdown(tables["fraud"], tables["account"])), encoding="utf-8")
        if "<!-- FINDINGS:START -->" in path.read_text(encoding="utf-8"):
            path.write_text(splice_block(path.read_text(encoding="utf-8"), "FINDINGS",
                                         findings_markdown(tables["fraud"], tables["account"], tables.get("drift_summary"))),
                            encoding="utf-8")
        if "latency" in tables and "<!-- LATENCY:START -->" in path.read_text(encoding="utf-8"):
            path.write_text(splice_block(path.read_text(encoding="utf-8"), "LATENCY",
                                         latency_markdown(tables["latency"], tables["latency_meta"])), encoding="utf-8")
        print("updated", path)


if __name__ == "__main__":
    main()
