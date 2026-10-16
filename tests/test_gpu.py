import pytest
import torch

from pipeline.config import load_config
from pipeline.prepare import prepare
from scripts.train import default_spec, predict_scores, train_variant
from tests.fixtures import fixture_overrides, make_fixture

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_one_epoch_on_cuda_matches_cpu_scores_closely():
    prep = prepare(load_config(overrides=fixture_overrides()), df=make_fixture()[0])
    spec = default_spec(prep, "fraud_only", 0)
    cpu = train_variant(prep, spec, "cpu")
    gpu = train_variant(prep, spec, "cuda")
    idx = spec.fraud_val_idx[:64]
    a = predict_scores(prep, cpu.model, cpu.heads, "fraud", idx, "cpu")
    b = predict_scores(prep, gpu.model, gpu.heads, "fraud", idx, "cuda")
    assert abs(a.mean() - b.mean()) < 0.1 and ((b >= 0) & (b <= 1)).all()
