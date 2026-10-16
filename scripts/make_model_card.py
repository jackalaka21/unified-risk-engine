"""Generate docs/MODEL_CARD.md and the README results block from the result files. Numbers are never typed by hand."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.config import resolve_path
from pipeline.notebook_support import seed_summary

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
FRAUD_COLS = ["pr_auc", "recall_at_fpr", "recall@0.01", "amount_recall@0.01", "cost", "brier", "ece"]


def load_tables(out_dir, reports_dir) -> dict:
    out_dir, reports_dir = Path(out_dir), Path(reports_dir)
    tables = {"fraud": pd.read_csv(out_dir / "results_fraud.csv"), "account": pd.read_csv(out_dir / "results_account.csv")}
    if (out_dir / "drift_summary.csv").exists():
        tables["drift_summary"] = pd.read_csv(out_dir / "drift_summary.csv")
    if (reports_dir / "explain_metrics.json").exists():
        tables["explain"] = json.loads((reports_dir / "explain_metrics.json").read_text(encoding="utf-8"))
    return tables


def check_single_config(tables: dict) -> str:
    hashes = set()
    for name, t in tables.items():
        hashes |= set(t["config_hash"].unique()) if isinstance(t, pd.DataFrame) else {t["config_hash"]}
    if len(hashes) != 1:
        raise ValueError(f"mixed config hashes {sorted(hashes)}: re-run all steps with one configuration")
    return hashes.pop()


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def results_markdown(fraud: pd.DataFrame, account: pd.DataFrame) -> str:
    f = seed_summary(fraud, ["pr_auc"])
    others = fraud.groupby("model")[[c for c in FRAUD_COLS if c != "pr_auc" and c in fraud.columns]].mean().reset_index()
    f = f.merge(others, on="model")
    f["pr_auc (mean, min-max over seeds)"] = [f"{m:.3f} ({lo:.3f}-{hi:.3f})" for m, lo, hi in
                                              zip(f["pr_auc_mean"], f["pr_auc_min"], f["pr_auc_max"])]
    f = f[["model", "pr_auc (mean, min-max over seeds)"] + [c for c in others.columns if c != "model"]]
    a = seed_summary(account, ["pr_auc"])[["model", "pr_auc_mean", "pr_auc_min", "pr_auc_max"]]
    return ("Transaction fraud, test period:\n\n" + _table(f) + "\n\nAccount risk (proxy label), test users:\n\n" + _table(a))


def build_model_card(out_dir, reports_dir, fixture=False) -> str:
    tables = load_tables(out_dir, reports_dir)
    cfg_hash = check_single_config(tables)
    fraud = tables["fraud"]
    parts = ["# Model card: unified risk engine\n"]
    if fixture:
        parts.append("> **FIXTURE RESULTS: generated data, not findings.** Nothing below describes real fraud.\n")
    parts.append(f"Configuration hash: `{cfg_hash}`. Seeds: {sorted(fraud['seed'].unique().tolist())}. "
                 f"Test transactions: {int(fraud['n_test'].iloc[0])}; fraud prevalence {fraud['prevalence'].iloc[0]:.3f}.\n")
    parts.append("## Intended use\n\nA research prototype that ranks transactions by fraud risk and users by an account-risk proxy, "
                 "to study whether a temporal graph model adds value over strong tabular baselines. Not for production decisions, "
                 "credit, or any decision about a person.\n")
    parts.append("## Data\n\nIEEE-CIS Fraud Detection (Kaggle; Vesta Corporation): card-not-present e-commerce transactions with "
                 "anonymised features, about 590k training transactions over about six months. Users are an approximation built "
                 "from card and address fields, so \"user\" is a proxy for a real account holder.\n")
    parts.append("## Leakage controls\n\n- Chronological split: train, validation, test, separated by an embargo so late labels cannot cross a boundary.\n"
                 "- Users are split into disjoint train, validation and test groups; the account label uses history up to the end of training only.\n"
                 "- Feature statistics are fitted on the train period only; history features are strictly causal.\n"
                 "- Neighbour sampling only reaches transactions at or before the seed transaction's time.\n"
                 "- Baselines get the same history features as the graph model.\n")
    parts.append("## Results\n\n" + results_markdown(fraud, tables["account"]) + "\n")
    parts.append("Cost uses an assumed review cost per false alert; compare it with `cost_flag_none`, the cost of flagging nothing. "
                 "Differences smaller than the bootstrap interval or the range over seeds are not evidence of a better model.\n")
    parts.append("## Drift\n\n" + (_table(tables["drift_summary"].drop(columns=["config_hash"]))
                                   if "drift_summary" in tables else "The drift study was not run.") + "\n")
    e = tables.get("explain")
    parts.append("## Explanations\n\n" + (
        f"Mode `{e['mode']}`. Stability (top-k overlap across runs): {e['stability']:.2f}. Deleting the top-k features moves the "
        f"logit by {e['fidelity_top']:.2f} on average, against {e['fidelity_random']:.2f} for random features"
        + (f"; PyG fidelity+ {e['pyg_fidelity_plus']:.2f}, fidelity- {e['pyg_fidelity_minus']:.2f}, characterization "
           f"{e['pyg_characterization']:.2f}" if e.get("pyg_fidelity_plus") is not None else "")
        + ". Features are anonymised; explanations describe the model and are not causal."
        if e else "Explanations have not been generated.") + "\n")
    parts.append("## Limitations\n\n- One dataset, one split in time; no claim about other portfolios or later periods.\n"
                 "- The account label is a proxy derived from fraud history, not a ground-truth account-takeover label.\n"
                 "- Anonymised features limit what an explanation can say.\n- Three seeds give a range, not a confidence interval.\n"
                 "- The review cost is an assumption.\n- Any explanation of *why* a model wins is a hypothesis unless an ablation tested it.\n")
    return "\n".join(parts)


def splice_readme(readme: str, block: str) -> str:
    if START not in readme or END not in readme:
        raise ValueError(f"README is missing the {START} / {END} markers (RESULTS block)")
    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{block}\n{END}{tail}"


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate the model card and update the README results block.")
    p.add_argument("--out", default="plots")
    p.add_argument("--reports", default="reports")
    p.add_argument("--card", default="docs/MODEL_CARD.md")
    p.add_argument("--readme", default=None, help="README path to update; omit to skip")
    p.add_argument("--fixture", action="store_true")
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
        print("updated", path)


if __name__ == "__main__":
    main()
