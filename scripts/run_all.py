"""One command from raw data to evaluation tables, drift study, explanations and the model card."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.config import config_hash, load_config, resolve_device, resolve_path
from pipeline.prepare import prepare
from scripts.drift_study import drift_study, summarize_drift
from scripts.evaluate import evaluate_all
from scripts.explain import explain_report
from scripts.make_model_card import build_model_card
from scripts.train import default_spec, train_variant


CHECKPOINT_NAME = "multitask_seed0.pt"          # the checkpoint the explanations, the export and the benchmark read


def run_all(cfg, out_dir, reports_dir, models_dir, df=None,
            variants=("multitask", "fraud_only", "account_only", "no_graph", "fraud_only_history",
                                      "multitask_history"), seeds=None,
            drift_kinds=("history_tabular", "history_stable_tabular", "gnn_fraud_only"), device=None, fixture=False, log=print):
    out_dir, reports_dir, models_dir = Path(out_dir), Path(reports_dir), Path(models_dir)
    device = resolve_device(device or cfg["train"]["device"])
    seeds = cfg["train"]["seeds"] if seeds is None else seeds
    if fixture:
        log("FIXTURE RUN: generated data, results are not findings")
    prepared = prepare(cfg, df=df)
    paths = evaluate_all(prepared, out_dir, variants=variants, seeds=seeds, device=device, log=log, fixture=fixture,
                         stacking=True)
    h = config_hash(cfg)
    drift = drift_study(prepared, seed=cfg["drift"]["seed"], kinds=drift_kinds, device=device, log=log)
    paths["drift_weekly"], paths["drift_summary"] = out_dir / "drift_weekly.csv", out_dir / "drift_summary.csv"
    drift.assign(config_hash=h, fixture=fixture).to_csv(paths["drift_weekly"], index=False)
    summarize_drift(drift).assign(config_hash=h, fixture=fixture).to_csv(paths["drift_summary"], index=False)
    ckpt = models_dir / CHECKPOINT_NAME  # one extra run so the explanations have a checkpoint to load
    train_variant(prepared, default_spec(prepared, "multitask", seeds[0]), device, ckpt)
    metrics = explain_report(prepared, ckpt, reports_dir, device)
    metrics.update(config_hash=h, mode=cfg["explain"]["mode"], fixture=fixture)
    (reports_dir / "explain_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    paths["explanations"] = reports_dir / "explanations.md"
    paths["card"] = out_dir / "MODEL_CARD.md"
    paths["card"].write_text(build_model_card(out_dir, reports_dir, fixture=fixture), encoding="utf-8")
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(description="Run the whole pipeline.")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--reports", default=None)
    p.add_argument("--models", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--fixture", action="store_true", help="use generated data; results are not findings")
    a = p.parse_args(argv)
    if a.fixture:
        from tests.fixtures import fixture_overrides, make_fixture

        cfg, df = load_config(a.config, overrides=fixture_overrides()), make_fixture()[0]
        base = "outputs/fixture"
    else:
        cfg, df, base = load_config(a.config), None, None
    out = resolve_path(a.out or (f"{base}/plots" if base else "plots"))
    reports = resolve_path(a.reports or (f"{base}/reports" if base else "reports"))
    models = resolve_path(a.models or (f"{base}/models" if base else "models"))
    paths = run_all(cfg, out, reports, models, df=df, seeds=a.seeds, fixture=a.fixture)
    print({k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
