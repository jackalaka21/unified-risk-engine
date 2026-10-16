import pandas as pd
import pytest

from pipeline.config import load_config
from scripts.run_all import run_all
from tests.fixtures import fixture_overrides, make_fixture


@pytest.mark.slow
def test_run_all_produces_every_artifact_with_one_config_hash(tmp_path):
    cfg = load_config(overrides=fixture_overrides())
    cfg["explain"]["mode"] = "occlusion"
    paths = run_all(cfg, tmp_path / "plots", tmp_path / "reports", tmp_path / "models", df=make_fixture()[0],
                    variants=("multitask", "fraud_only"), seeds=[0], drift_kinds=("history_tabular",),
                    device="cpu", fixture=True, log=lambda *_: None)
    for key in ("fraud", "account", "weekly", "calibration", "drift_weekly", "drift_summary", "explanations", "card"):
        assert paths[key].exists(), key
    hashes = {pd.read_csv(paths[k])["config_hash"].iloc[0] for k in ("fraud", "account", "drift_summary")}
    assert len(hashes) == 1
    assert "FIXTURE RESULTS" in paths["card"].read_text(encoding="utf-8")
    assert (tmp_path / "models" / "multitask_seed0.pt").exists()
