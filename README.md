# fairllms

[![PyPI](https://img.shields.io/pypi/v/fairllms)](https://pypi.org/project/fairllms/)
[![Python](https://img.shields.io/pypi/pyversions/fairllms)](https://pypi.org/project/fairllms/)
[![Tests](https://github.com/michaellarionov/JMLR_Library/actions/workflows/test.yml/badge.svg)](https://github.com/michaellarionov/JMLR_Library/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Fairness definitions and bias metrics for large language models — a companion library for studying and evaluating bias in LMs.

The long-term goal is a **stable, sklearn-style API**: import a metric, call `compute(...)`, and get a result — without caring where the implementation lives.

> **Status:** All 33 metrics share one contract — configuration in the
> constructor, data as a validated container passed to `compute(model, data)`.
> Leaf `main.py` files and `examples/` are short public-API demos.
> Implementation math still lives under `fairllms/definition/`.

## Install

```bash
pip install fairllms
```

That's all most users need — [`fairllms` is on PyPI](https://pypi.org/project/fairllms/),
so installing does not require access to this repository.

### Optional extras

```bash
pip install "fairllms[openai]"   # API-backed decoder metrics (CR, CTF, BA)
pip install "fairllms[dev]"      # pytest
pip install "fairllms[all]"      # openai + common extras
```

### From source

To track unreleased work, or to develop the library:

```bash
git clone https://github.com/michaellarionov/JMLR_Library.git && cd JMLR_Library && pip install -e ".[dev]"
```

An editable install means your edits take effect immediately, with no reinstall.
You can also install a specific commit or tag directly:

```bash
pip install "git+https://github.com/michaellarionov/JMLR_Library.git@v0.2.0"
```

Prefer a tag over `@main` if you do this: `main` tracks unreleased work, so any
push can change behaviour under you. Tags don't move.

The import name is `fairllms` regardless of the repository name.

### Requirements

Python ≥ 3.10 — the floor for `torch`, `transformers` and `datasets`.

Installed automatically: `torch`, `transformers`, `datasets`, `numpy`, `pandas`,
`scipy`, `scikit-learn`, `wordfreq`, `nltk`. The last three are needed at import
time by `counterfactual_auc` / `normalized_position_distance`,
`lexical_frequency_proportion`, and `morphological_choice_divergence`
respectively.

## Versioning

Semantic versioning, currently pre-1.0 — the public API may still change between
minor versions. The version is single-sourced in
[`fairllms/_version.py`](fairllms/_version.py); `pyproject.toml` reads it via
`[tool.setuptools.dynamic]`, so bump that one value.

To cut a release, bump `_version.py`, then tag. Pushing a `v*` tag triggers
[`.github/workflows/publish.yml`](.github/workflows/publish.yml), which builds,
validates with `twine check --strict`, checks the tag matches the version, and
uploads to PyPI via Trusted Publishing (OIDC — no API token is stored anywhere):

```bash
git tag -a "v$(python scripts/package_version.py)" -m "Release" && git push origin --tags
```

Users then upgrade with `pip install --upgrade fairllms`.

### Breaking changes

| Version | Change | Migration |
|---------|--------|-----------|
| 0.2.0 | Package renamed `fairLLMs` → `fairllms` (PEP 8) | `import fairllms` |
| 0.2.0 | Metric config is keyword-only | `WEAT(pooling="cls")`, not `WEAT("cls")` |

Metric *calling* conventions were **not** broken in 0.2.0: the previous keyword
style (`compute(model=m, T1_terms=[...])`) still works and emits a
`DeprecationWarning` naming its replacement. Those shims are scheduled for
removal in a later release, so migrate when convenient.

## Quick start

```python
from fairllms.metrics import CrowSPairsScore, list_metrics
from fairllms.datasets import CrowSPairs
from fairllms.models import HuggingFaceModel

model = HuggingFaceModel("bert-base-uncased", task="mlm")
result = CrowSPairsScore().compute(model, CrowSPairs(n_max=50))
print(result.score, result.by_category)

# Discover metrics
print(list_metrics())
```

### The contract

Every metric follows scikit-learn's estimator conventions:

```python
Metric(**config).compute(model, data) -> MetricResult
```

* **Configuration** goes in the constructor, keyword-only, and is introspectable
  via `get_params()` / `set_params()` — so metrics can be cloned or swept.
* **Data** is the second positional argument: a `FairnessDataset`, a plain
  sequence, or a typed container from `fairllms.metrics.data` for metrics that
  need several labelled sets.
* **Unknown keywords raise `TypeError`** instead of silently using a default.

```python
from fairllms.metrics import WEAT, SEAT, WordSets

words = WordSets(target_1=t1, target_2=t2, attribute_1=a1, attribute_2=a2)
WEAT(n_samples=10_000).compute(model, words)
SEAT(pooling="cls").compute(model, words)      # same data, different metric

WEAT(pooling="cls").get_params()               # {'n_samples': 10000, 'pooling': 'cls'}
```

Containers validate at construction, so mistakes fail immediately:

```python
>>> ContextSets(["a sentence"], ...)
TypeError: target_1 must be a mapping of term -> list of context sentences, got
list. (A flat list of sentences is not accepted; CEAT samples contexts per term,
so terms must be keyed.)
```

### Metrics that need no model

Five metrics score predictions you already have. They also exist as plain
functions, mirroring `sklearn.metrics`:

```python
from fairllms.metrics import equal_opportunity_gap, accuracy_disparity

equal_opportunity_gap(y_true, y_pred, groups, g1="A", g2="B")   # -> float
accuracy_disparity(scores_stereotype, scores_counter)           # -> float
```

Also available: `inference_bias_score`, `fair_inference_score`,
`context_based_disparity`. Use the class form
(`EqualOpportunityGap`, …) when you want the full `MetricResult` with
diagnostics.

Shared loaders:

```python
from fairllms.datasets import CrowSPairs, BBQ, StereoSet
from fairllms.models import load_masked_lm

crows = CrowSPairs().load()
bbq = BBQ(categories=["Age"]).load()
loaded = load_masked_lm("bert-base-uncased")
```

Leaf runners under `definition/` are short demos of the same public API:

```bash
python -m fairllms.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cps.main
```

See also `examples/` at the repository root.

## Package layout

```
fairllms/
├── metrics/        # Public API: CrowSPairsScore, WEAT, … (all expose compute)
│   ├── data.py     #   Validated input containers (WordSets, ProbeSet, …)
│   └── functional.py #  sklearn.metrics-style functions (model-free metrics)
├── datasets/       # CrowSPairs, StereoSet, BBQ, BiasInBios, WinoBias, …
├── models/         # HuggingFaceModel, OpenAIModel, load_* helpers
├── utils/          # PLL / masking / association / path helpers
├── data/           # Bundled CrowS-Pairs + BBQ files
├── artifacts/      # Preferred output dir for metric CSVs
└── definition/     # Internal implementations + short public-API demos (main.py)
```

Repo-root `examples/` has additional runnable snippets.

Every metric exposes the same method: `compute(...)`.

## Datasets

| Class | Source | Notes |
|-------|--------|--------|
| `CrowSPairs` | Bundled CSV under `fairllms/data/crows_pairs/` | Stereotype / anti pairs |
| `StereoSet` | Hugging Face (`stereoset` / `McGill-NLP/stereoset`) | Pairs or triples |
| `BBQ` | Bundled jsonl under `fairllms/data/bbq/` | Optional `context_condition` filter |
| `BiasInBios` | Hugging Face `LabHC/bias_in_bios` | Profession / gender helpers |
| `WinoBias` | Hugging Face `wino_bias` | Occupation direction helpers |
| `XNLIReligionPairs` | Hugging Face XNLI + templates | Religion swap pairs |

Loaders prefer canonical files in `fairllms/data/`, then fall back to legacy copies under `definition/` so existing scripts keep working.

## Models

```python
from fairllms.models import (
    HuggingFaceModel,
    load_masked_lm,      # task="mlm"
    load_encoder,        # task="encoder"
    load_sequence_classifier,
    load_seq2seq,
    load_causal_lm,
    OpenAIModel,
)

HuggingFaceModel("roberta-base", task="sequence_classification").load()
OpenAIModel("davinci-002").load()  # needs OPENAI_API_KEY; pip install fairllms[openai]
```

Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for gated models such as Llama-2.

## Design principles

1. **One verb for metrics** — `compute` (sklearn’s `fit` / `predict` analogue).
2. **Separate metrics from datasets** — reuse the same metric on CrowS-Pairs, StereoSet, or custom data.
3. **Stable public surface** — internals under `definition/` can change without breaking user code.
4. **Book-aligned taxonomy** — `definition/{encoder_only,encoder_decoder,decoder_only}/{intrinsic_bias,extrinsic_bias}/…` mirrors the conceptual organization of the accompanying textbook.

## Development

```bash
pip install -e ".[dev]"
pytest                  # contract suite over every metric in the registry
# Run a leaf metric script:
python -m fairllms.definition.encoder_only.intrinsic_bias.similarity_based.weat.main
```

`tests/test_common.py` is the analogue of scikit-learn's `check_estimator`: it
runs the parameter/repr/keyword contract across `METRIC_REGISTRY`, so a new
metric that breaks the shape fails there rather than surprising a user. It needs
no network or model weights.

Tests that do need model weights skip cleanly when the weights aren't cached, so
the suite is hermetic:

```bash
HF_HUB_OFFLINE=1 pytest -q     # what CI runs; a few seconds, no downloads
```

CI ([`.github/workflows/test.yml`](.github/workflows/test.yml)) runs this on
Python 3.10 and 3.13 for every push and pull request. A second `clean-install`
job builds the wheel, installs it into a fresh environment, and imports it from a
directory with no source checkout on `sys.path`.

That second job exists because an editable install cannot catch a whole class of
packaging bug: `fairllms.metrics` eagerly imports every metric family, so any
third-party module imported at module scope under `fairllms/definition/` is a
hard requirement of `import fairllms`. If such a dependency is only listed in an
extra, `pip install fairllms` produces a package that cannot be imported — while
every local test still passes, because the developer's environment already has
it. [`tests/test_packaging.py`](tests/test_packaging.py) also guards this
statically, walking the AST and naming the offending file.

### Publishing an edit

Edits reach users through a tagged release, not through `main`:

1. Edit locally — your editable install picks changes up immediately.
2. `pytest` — the contract suite catches API breakage before it ships.
3. Commit and push. CI verifies the matrix and the clean install.
4. Bump [`fairllms/_version.py`](fairllms/_version.py), then tag and push the tag.
5. The publish workflow uploads to PyPI; users get it with
   `pip install --upgrade fairllms`.

Metric result CSVs should go under `fairllms/artifacts/` (via `fairllms.utils.results_to_csv`); leaf-local `*_results.csv` files are gitignored.

## License

MIT
