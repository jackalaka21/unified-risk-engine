"""Derive requirements-ci.txt (CPU-only, no notebooks) from the pinned requirements.txt."""
from __future__ import annotations

import re
from pathlib import Path

DROP = ("torch", "pyg-lib", "pyg_lib", "jupyterlab", "ipykernel", "nbconvert")  # torch and pyg-lib are re-added for CPU


def ci_requirements(lines, extra=()):
    out = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[=<>~ ]", line, maxsplit=1)[0].lower()
        if name in DROP:
            continue
        out.append(re.sub(r"\+[A-Za-z0-9.]+", "", line))
    return out + list(extra)


SERVING_KEEP = {"numpy", "pandas", "pyyaml", "torch-geometric", "fastapi", "uvicorn"}   # pydantic comes with fastapi


def serving_requirements(lines):
    out = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if re.split(r"[=<>~ ]", line, maxsplit=1)[0].lower() in SERVING_KEEP:
            out.append(re.sub(r"\+[A-Za-z0-9.]+", "", line))
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    lines = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    extra = ["--extra-index-url https://download.pytorch.org/whl/cpu", "torch==2.14.1",
             "--find-links https://data.pyg.org/whl/torch-2.14.0+cpu.html", "pyg-lib==0.9.0+pt214cpu"]
    body = ci_requirements(lines, extra=[])
    (root / "requirements-ci.txt").write_text("\n".join(extra + body) + "\n", encoding="utf-8")
    print((root / "requirements-ci.txt").read_text(encoding="utf-8"))
    (root / "requirements-serving.txt").write_text("\n".join(serving_requirements(lines)) + "\n", encoding="utf-8")
