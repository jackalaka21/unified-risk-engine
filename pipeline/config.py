"""Configuration loading, validation, hashing and device resolution."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
VARIANTS = ("multitask", "fraud_only", "account_only", "no_graph")
CONVS = ("gatv2", "sage")


def resolve_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _merge(base: dict, over: dict) -> dict:
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def validate(cfg: dict) -> None:
    s = cfg["splits"]
    if not 0 < s["train_frac"] < s["val_end_frac"] < 1:
        raise ValueError("splits.train_frac must be below splits.val_end_frac, both within (0, 1)")
    if s["embargo_days"] < 0:
        raise ValueError("splits.embargo_days must be >= 0")
    if len(s["user_fractions"]) != 3 or abs(sum(s["user_fractions"]) - 1.0) > 1e-6:
        raise ValueError("splits.user_fractions must be three numbers that sum to 1")
    if cfg["train"]["variant"] not in VARIANTS:
        raise ValueError(f"train.variant must be one of {VARIANTS}")
    if cfg["model"]["conv"] not in CONVS:
        raise ValueError(f"model.conv must be one of {CONVS}")
    nn = cfg["train"]["num_neighbors"]
    if not nn or not all(isinstance(n, int) and n > 0 for n in nn):
        raise ValueError("train.num_neighbors must be a list of positive integers")
    if cfg["model"]["layers"] != len(nn):
        raise ValueError("model.layers must equal len(train.num_neighbors)")
    if not 0 < cfg["train"]["focal_alpha"] < 1:
        raise ValueError("train.focal_alpha must be within (0, 1)")


def load_config(path=None, overrides: dict | None = None) -> dict:
    path = resolve_path(path) if path else DEFAULT_CONFIG
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if os.environ.get("DATA_DIR"):
        cfg["data"]["dir"] = os.environ["DATA_DIR"]          # the spec's override; explicit `overrides` still win
    if overrides:
        _merge(cfg, copy.deepcopy(overrides))
    validate(cfg)
    return cfg


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Config requests device '{requested}' but CUDA is not available. "
            "Set `train.device: auto` (falls back to CPU) or `cpu`."
        )
    return requested


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]
