import yaml

from pipeline.config import PROJECT_ROOT
from scripts.make_ci_requirements import ci_requirements


def test_ci_requirements_drop_local_builds_and_gpu_only_packages():
    lines = ["torch==2.14.1+cu130", "pyg-lib==0.9.0+pt214cu130", "pandas==2.3.1", "jupyterlab==4.4.0",
             "ipykernel==6.30.0", "numpy==2.3.0", "# a comment"]
    out = ci_requirements(lines, extra=["torch==2.14.1"])
    assert "torch==2.14.1" in out and "pandas==2.3.1" in out and "numpy==2.3.0" in out
    assert not any("cu130" in l or "jupyterlab" in l or "ipykernel" in l for l in out)
    assert not any(l.startswith("#") for l in out)


def test_workflow_runs_the_cpu_suite_without_secrets():
    wf = yaml.safe_load((PROJECT_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8"))
    text = (PROJECT_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    steps = " ".join(str(s.get("run", "")) for s in wf["jobs"]["test"]["steps"])
    assert 'pytest -m "not gpu and not slow"' in steps and "requirements-ci.txt" in steps
    assert "secrets." not in text and "KAGGLE" not in text
    assert "cu130" not in (PROJECT_ROOT / "requirements-ci.txt").read_text(encoding="utf-8")
