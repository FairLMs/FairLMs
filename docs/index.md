# FairLMs

A Python library for measuring bias and fairness in language models.

Thirty-three published bias metrics, re-implemented behind one interface. Every
metric exposes `compute(model, data) -> MetricResult`, follows scikit-learn's
estimator conventions — configuration in `__init__`, data at `compute`, working
`get_params` / `set_params` — and accepts any model through the same adapter
layer, so the same metric runs across checkpoints and architectures without
per-metric loading code.

Alongside the metrics, a separate dataset-first diagnostics layer audits
evaluation data and score tables directly. Evidence that cannot support a
measurement is reported as `blocked` or `not_applicable`, **never as a measured
zero** — the distinction between "no disparity" and "you have not given me what
I need to say anything" is preserved in the report rather than collapsed into a
number.

<div class="grid cards" markdown>

- **[Install](install.md)** — `pip install fairlms`
- **[Quickstart](quickstart.md)** — first metric in ten lines
- **[Registry](registry/metrics.md)** — every metric, loader and diagnostic
- **[API reference](api/metrics.md)** — full generated reference

</div>

## Coverage at a glance

| | Encoder-only | Decoder-only | Encoder-decoder |
|---|---|---|---|
| **Intrinsic** | 11 metrics | 4 | 4 |
| **Extrinsic** | 3 | 7 | 4 |

Six benchmark loaders (CrowS-Pairs, BBQ, StereoSet, Bias in Bios, WinoBias,
XNLI) fill the same `data` argument you can pass by hand, plus eight bundled
WEAT / SEAT word sets.

Five dataset diagnostics: axis representativeness (`b_rep`) and four scoring
instrument audits (mean gap, rate gap, Wasserstein-1 gap, counterfactual
sensitivity).

!!! note "Status"
    MIT licensed. Declared support: Python 3.10–3.13. CI runs the test suite on
    Python 3.10 and 3.13 on Linux; see the repository CI for the current test
    count and coverage.
