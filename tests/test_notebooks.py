import nbformat
import pytest

from pipeline.config import PROJECT_ROOT, load_config
from pipeline.prepare import prepare
from scripts.build_notebooks import NOTEBOOKS, build
from scripts.drift_study import drift_study, summarize_drift
from scripts.evaluate import evaluate_all
from scripts.explain import explain_report
from scripts.train import default_spec, train_variant
from tests.fixtures import fixture_overrides, make_fixture


def test_every_notebook_cell_kind_is_valid():
    assert len(NOTEBOOKS) == 5
    for cells in NOTEBOOKS.values():
        assert cells and {kind for kind, _ in cells} <= {"md", "code"}


def test_build_writes_valid_notebooks_with_the_project_kernel(tmp_path):
    for path in build(tmp_path):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        assert nb.metadata["kernelspec"]["name"] == "unified_risk_engine"


@pytest.mark.slow
def test_notebooks_execute_on_fixture_results(tmp_path, monkeypatch):
    out, reports = tmp_path / "plots", tmp_path / "reports"
    cfg = load_config(overrides=fixture_overrides())
    cfg["explain"]["mode"] = "occlusion"
    prep = prepare(cfg, df=make_fixture()[0])
    evaluate_all(prep, out, variants=("multitask", "fraud_only"), seeds=[0], device="cpu", log=lambda *_: None)
    drift = drift_study(prep, kinds=("history_tabular",), log=lambda *_: None)
    drift.to_csv(out / "drift_weekly.csv", index=False)
    summarize_drift(drift).to_csv(out / "drift_summary.csv", index=False)
    train_variant(prep, default_spec(prep, "multitask", 0), "cpu", tmp_path / "m.pt")
    explain_report(prep, tmp_path / "m.pt", reports, device="cpu")
    from scripts.benchmark_latency import run_benchmark

    train_variant(prep, default_spec(prep, "fraud_only", 0), "cpu", tmp_path / "f.pt")
    run_benchmark(prep, tmp_path / "f.pt", out, devices=["cpu"], batch_sizes=(1, 4), n_requests=40, include_http=False,
                  skip_parity=True, fixture=True, log=lambda *_: None)
    from nbclient import NotebookClient  # not installed in CI, which skips this slow test

    monkeypatch.setenv("URE_FIXTURE", "1")
    monkeypatch.setenv("URE_OUT_DIR", str(out))
    monkeypatch.setenv("URE_REPORTS_DIR", str(reports))
    for path in build(tmp_path / "nb"):
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(PROJECT_ROOT)}}).execute()
