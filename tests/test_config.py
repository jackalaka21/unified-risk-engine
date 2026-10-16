import pytest
import torch

from pipeline.config import config_hash, load_config, resolve_device


def test_default_config_loads_with_spec_values():
    cfg = load_config()
    assert cfg["splits"]["train_frac"] == 0.70 and cfg["splits"]["val_end_frac"] == 0.85
    assert cfg["splits"]["embargo_days"] == 7
    assert cfg["account"]["min_transactions"] == 3 and cfg["account"]["fraud_rate_threshold"] == 0.2
    assert cfg["model"]["hidden"] == 64 and cfg["model"]["heads"] == 4 and cfg["model"]["layers"] == 2
    assert cfg["train"]["num_neighbors"] == [15, 10] and cfg["train"]["focal_alpha"] == 0.75
    assert cfg["train"]["seeds"] == [0, 1, 2]


def test_overrides_merge_deeply():
    cfg = load_config(overrides={"train": {"epochs": 1}, "seed": 5})
    assert cfg["train"]["epochs"] == 1 and cfg["train"]["patience"] == 3 and cfg["seed"] == 5


def test_data_dir_environment_variable_overrides_the_config_but_not_explicit_overrides(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "somewhere")
    assert load_config()["data"]["dir"] == "somewhere"
    assert load_config(overrides={"data": {"dir": "explicit"}})["data"]["dir"] == "explicit"


@pytest.mark.parametrize("overrides, message", [
    ({"splits": {"train_frac": 0.9, "val_end_frac": 0.8}}, "train_frac"),
    ({"splits": {"user_fractions": [0.5, 0.2, 0.1]}}, "user_fractions"),
    ({"train": {"variant": "bogus"}}, "variant"),
    ({"model": {"conv": "gcn"}}, "conv"),
    ({"model": {"layers": 3}}, "layers"),
    ({"splits": {"embargo_days": -1}}, "embargo_days"),
])
def test_invalid_configs_are_rejected_with_a_named_key(overrides, message):
    with pytest.raises(ValueError, match=message):
        load_config(overrides=overrides)


def test_config_hash_is_stable_and_sensitive():
    a, b = load_config(), load_config()
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(load_config(overrides={"seed": 1}))


def test_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu" and resolve_device("cpu") == "cpu"


def test_device_cuda_unavailable_message(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="device: auto"):
        resolve_device("cuda")


def test_fixture_overrides_disable_temporal_sampling_without_pyg_lib(monkeypatch):
    import importlib.util

    from tests.fixtures import fixture_overrides

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert fixture_overrides()["train"]["temporal"] is False
