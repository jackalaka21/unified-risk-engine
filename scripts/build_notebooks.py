"""Generate the notebooks from the cell lists below, so their source is reviewable as plain text."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

KERNEL = {"name": "unified_risk_engine", "display_name": "Python (unified_risk_engine)", "language": "python"}
SETUP = "%matplotlib inline\nfrom pipeline.notebook_support import *\nimport pandas as pd\nimport numpy as np\npd.set_option('display.width', 160)"

NOTEBOOKS = {
    "01_data_and_graph.ipynb": [
        ("md", "# 1. Data, splits and graph\n\nWhat the data looks like, how it is split in time, and the graph that is built from it. "
               "If this says *fixture*, the numbers describe generated data and are **not findings**."),
        ("code", SETUP + "\nprep = get_prepared()\ncfg, df = prep.cfg, prep.df\nprint('FIXTURE DATA - not real results' if fixture_mode() else 'IEEE-CIS')"),
        ("code", "days = (df['TransactionDT'].max() - df['TransactionDT'].min()) / 86400\n"
                 "pd.Series({'transactions': len(df), 'users': df['user'].nunique(), 'days': round(days, 1),\n"
                 "           'fraud rate': round(df['isFraud'].mean(), 4)})"),
        ("md", "## Time periods\n\nTrain, validation and test are consecutive in time, separated by an embargo so that labels "
               "that arrive late cannot leak across a boundary."),
        ("code", "from pipeline.splits import DAY, period_of\n"
                 "p = prep.periods\nperiod = pd.Series(period_of(df['TransactionDT'].to_numpy(), p), name='period')\n"
                 "table = df.groupby(period).agg(rows=('isFraud', 'size'), fraud_rate=('isFraud', 'mean'))\ntable"),
        ("code", "gaps = {'train->validation (days)': (p.val_start - p.t_train) / DAY, 'validation->test (days)': (p.test_start - p.val_end) / DAY}\n"
                 "assert all(abs(g - p.embargo / DAY) < 1e-6 for g in gaps.values()), gaps\ngaps"),
        ("md", "## Account-risk population\n\nA proxy label from fraud history up to the end of training, for users with enough history."),
        ("code", "acc = prep.account\npd.DataFrame({'users': acc.groupby('split').size(), 'eligible': acc[acc['eligible']].groupby('split').size(),\n"
                 "              'positives': acc[acc['eligible']].groupby('split')['label'].sum()})"),
        ("md", "## Threshold choice\n\nThe account label's fraud-rate threshold is chosen by how many positives it leaves, never by model "
               "performance. The configured value is shown against alternatives."),
        ("code", "rates = acc[acc['eligible']]['fraud_rate']\n"
                 "print('configured threshold:', cfg['account']['fraud_rate_threshold'])\n"
                 "pd.DataFrame({t: {'positives': int((rates >= t).sum()), 'prevalence': round(float((rates >= t).mean()), 4)}\n"
                 "              for t in (0.1, 0.2, 0.3, 0.5)}).T"),
        ("md", "## Leakage checks\n\nTrain-period fraud rows come only from train users and end by the end of the training period."),
        ("code", "tr = prep.graph['transaction'].train_mask.numpy()\n"
                 "assert (df.loc[tr, 'TransactionDT'] <= p.t_train).all()\n"
                 "assert (prep.user_split.reindex(df.loc[tr, 'user']).to_numpy() == 'train').all()\n"
                 "print('leakage checks passed')"),
        ("md", "## Graph\n\nNode and edge counts of the heterogeneous graph (transaction, user, email_domain, device)."),
        ("code", "g = prep.graph\ndisplay(pd.DataFrame({'nodes': {nt: g[nt].num_nodes for nt in g.node_types}}))\n"
                 "display(pd.DataFrame({'edges': {str(et): g[et].edge_index.shape[1] for et in g.edge_types}}))"),
    ],
    "02_train_and_compare.ipynb": [
        ("md", "# 2. Results\n\nAll models on the same test period. Differences smaller than the bootstrap interval or the "
               "range over seeds are not claims; three seeds give a range, not a confidence interval."),
        ("code", SETUP + "\nfraud = load_results('results_fraud.csv')\naccount = load_results('results_account.csv')\n"
                 "print('config hash:', fraud['config_hash'].unique())"),
        ("code", "metrics = ['pr_auc', 'roc_auc', 'recall_at_fpr', 'cost', 'brier', 'ece']\nseed_summary(fraud, metrics).round(3)"),
        ("code", "plot_pr_auc(fraud);"),
        ("md", "## Alert budgets\n\nRecall, precision and the share of fraud *value* caught when only the top 0.5%, 1% or 5% of "
               "transactions can be reviewed."),
        ("code", "cols = [c for c in fraud.columns if c.split('@')[0] in ('recall', 'precision', 'amount_recall') and '@' in c]\n"
                 "fraud.groupby('model')[cols].mean().round(3)"),
        ("md", "## Cost\n\nExpected cost uses an assumed review cost per false alert (see the config); compare each model with flagging nothing."),
        ("code", "fraud.groupby('model')[['cost', 'cost_flag_none', 'cost_flag_all']].mean().round(0)"),
        ("md", "## Account-risk head and ablations"),
        ("code", "seed_summary(account, ['pr_auc', 'roc_auc']).round(3)"),
        ("md", "## Over time and calibration"),
        ("code", "plot_weekly(load_results('results_weekly.csv'));"),
        ("code", "plot_calibration(load_results('calibration.csv'));"),
    ],
    "03_drift_and_retraining.ipynb": [
        ("md", "# 3. Drift and retraining\n\nA model trained once decays as fraud patterns change. The *retrain* schedule refits the "
               "model on all labels known at each cut-off (older than the embargo)."),
        ("code", SETUP + "\ndrift = load_results('drift_weekly.csv')\nsummary = load_results('drift_summary.csv')\nsummary.round(4)"),
        ("code", "plot_drift(drift);"),
        ("md", "A negative `slope_per_week` for *static* means the frozen model gets worse each week. If *retrain* is not better, "
               "then on this data retraining did not help; report that as it is."),
    ],
    "04_explainability.ipynb": [
        ("md", "# 4. Explainability\n\nWhich inputs drive the model's score for individual transactions, and how much to trust "
               "those explanations. Features are anonymised, so names are column names, not business meaning; explanations "
               "describe the model, not causes."),
        ("code", SETUP + "\nfrom IPython.display import Image, Markdown, display\nr = reports_dir()\n"
                 "display(Markdown((r / 'explanations.md').read_text(encoding='utf-8')))"),
        ("code", "display(Image(filename=str(r / 'feature_importance.png')))"),
    ],
    "05_serving_latency.ipynb": [
        ("md", "# 5. Serving and latency\n\nThe model scored one transaction at a time by a service that keeps a time-sorted "
               "neighbour index. Numbers come from one machine replaying historical data; they are not a production claim, and "
               "the 100 ms budget is an illustrative assumption."),
        ("code", SETUP + "\nimport json\nlat = load_results('latency.csv')\nmeta = json.loads((out_dir() / 'latency_meta.json').read_text())\n"
                 "print('FIXTURE RESULTS - not findings' if meta['fixture'] else 'real data')\nprint('hardware:', meta['hardware'])\n"
                 "print('parity with the offline scores:', meta['parity'])"),
        ("code", "cols = ['model', 'device', 'mode', 'batch_size', 'concurrency', 'p50_ms', 'p95_ms', 'p99_ms', 'throughput_rps', 'p99_within_budget']\n"
                 "aggregate_latency(lat)[[c for c in cols if c not in ('p99_within_budget',)] + ['p50_min_ms', 'p50_max_ms', 'repeats']].round(2)"),
        ("code", "plot_latency(lat, meta['budget_ms']);"),
        ("md", "## Where the time goes (batch size 1)"),
        ("code", "plot_breakdown(lat);"),
        ("md", "## Stock sampler versus the custom index\n\nThe `stock_neighbor_loader` row is the cost of sampling the neighbourhood "
               "alone with PyG's loader built per request; compare it with the `neighbour_lookup` share above."),
        ("code", "aggregate_latency(lat[lat['mode'] == 'lookup_only'])[['model', 'p50_ms', 'p95_ms', 'p99_ms', 'repeats']].round(2)"),
        ("md", "Index size and the concurrency caveat: the metadata printed above lists the indexed transactions and the serialised "
               "index size. Scoring is serialised by one lock, so extra clients do not add throughput; the multi-client row shows "
               "the queueing cost."),
    ],
}


def build(out_dir="notebooks") -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, cells in NOTEBOOKS.items():
        nb = new_notebook(cells=[new_markdown_cell(s) if kind == "md" else new_code_cell(s) for kind, s in cells],
                          metadata={"kernelspec": KERNEL})
        nbformat.validate(nb)
        path = out_dir / name
        nbformat.write(nb, path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in build():
        print("wrote", p)
