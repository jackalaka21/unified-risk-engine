
from scripts.sampling_rule import sampling_rule_table
from scripts.train import default_spec, train_variant
from tests.serving_helpers import last_rule_prepared


def test_table_has_both_rules_with_finite_scores_and_leaves_the_config_alone(tmp_path):
    prep = last_rule_prepared()
    train_variant(prep, default_spec(prep, "fraud_only", 0), "cpu", tmp_path / "m.pt")
    before = prep.cfg["train"]["temporal_strategy"]
    table = sampling_rule_table(prep, tmp_path / "m.pt")
    assert list(table["rule"]) == ["uniform", "last"] and table["pr_auc"].notna().all()
    assert table["config_hash"].nunique() == 1 and prep.cfg["train"]["temporal_strategy"] == before
