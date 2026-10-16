from scripts.spike_serving import time_index, time_stock_loader
from tests.serving_helpers import HAS_TEMPORAL, last_rule_prepared

import numpy as np
import pytest


@pytest.mark.skipif(not HAS_TEMPORAL, reason="the stock loader path needs pyg-lib")
def test_both_timers_return_one_measurement_per_request():
    prep = last_rule_prepared()
    rows = np.arange(len(prep.df) // 2, len(prep.df) // 2 + 5)
    assert len(time_stock_loader(prep, rows)) == 5 and len(time_index(prep, rows)) == 5
