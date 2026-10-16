# Serving spike

> Fixture data: indicative only.

Per-request cost of building the sampled neighbourhood, milliseconds, one machine, CPU:

| method | p50 | p95 |
|---|---|---|
| stock `NeighborLoader`, batch size 1 (loader built per request) | 9.40 | 10.60 |
| custom time-sorted index | 0.063 | 0.085 |

Decision: serve with the custom index regardless of the gap. The serving image then needs no sampler dependency (pyg-lib), the index is append-only so new transactions become neighbours immediately, and the parity tests pin it to the stock sampler's `"last"` rule.
