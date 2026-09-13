# FairLMs

[![PyPI](https://img.shields.io/pypi/v/fairlms)](https://pypi.org/project/fairlms/)
[![Python](https://img.shields.io/pypi/pyversions/fairlms)](https://pypi.org/project/fairlms/)
[![Tests](https://github.com/michaellarionov/FairLMs/actions/workflows/test.yml/badge.svg)](https://github.com/michaellarionov/FairLMs/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Fairness definitions and bias metrics for large language models — a companion library for studying and evaluating bias in LMs.

The long-term goal is a **stable, sklearn-style API**: import a metric, call `compute(...)`, and get a result — without caring where the implementation lives.

## Install

```bash
pip install fairlms
```

That's all most users need — [`fairlms` is on PyPI](https://pypi.org/project/fairlms/),
so installing does not require access to this repository.

The project is **FairLMs**; both the PyPI distribution and the import name are
`fairlms`, all-lowercase per PEP 8.

### Optional extras

```bash
pip install "fairlms[openai]"   # API-backed decoder metrics (CR, CTF, BA)
pip install "fairlms[dev]"      # pytest
pip install "fairlms[all]"      # openai + common extras
```

### From source

To track unreleased work, or to develop the library:

```bash
git clone https://github.com/michaellarionov/FairLMs.git && cd FairLMs && pip install -e ".[dev]"
```

An editable install means your edits take effect immediately, with no reinstall.
You can also install a specific commit or tag directly:

```bash
pip install "git+https://github.com/michaellarionov/FairLMs.git@v0.4.0"
```

Prefer a tag over `@main` if you do this: `main` tracks unreleased work, so any
push can change behaviour under you. Tags don't move.

The import name is `fairlms` regardless of the repository name.

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
[`fairlms/_version.py`](fairlms/_version.py); `pyproject.toml` reads it via
`[tool.setuptools.dynamic]`, so bump that one value.

To cut a release, bump `_version.py`, then tag. Pushing a `v*` tag triggers
[`.github/workflows/publish.yml`](.github/workflows/publish.yml), which builds,
validates with `twine check --strict`, checks the tag matches the version, and
uploads to PyPI via Trusted Publishing (OIDC — no API token is stored anywhere):

```bash
git tag -a "v$(python scripts/package_version.py)" -m "Release" && git push origin --tags
```

Users then upgrade with `pip install --upgrade fairlms`.

## Quick start

```python
from fairlms.metrics import CrowSPairsScore, list_metrics
from fairlms.datasets import CrowSPairs
from fairlms.models import HuggingFaceModel

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
  sequence, or a typed container from `fairlms.metrics.data` for metrics that
  need several labelled sets.
* **Unknown keywords raise `TypeError`** instead of silently using a default.

```python
from fairlms.metrics import WEAT, SEAT, WordSets

words = WordSets(target_1=t1, target_2=t2, attribute_1=a1, attribute_2=a2)
WEAT(n_samples=10_000).compute(model, words)
SEAT(pooling="cls").compute(model, words)      # same data, different metric

WEAT(pooling="cls").get_params()               # {'n_samples': 10000, 'pooling': 'cls'}
```

The published association tests ship pre-wrapped in `fairlms.data`, so a
standard run needs no term lists at all — and swapping the checkpoint does not
change the call:

```python
from fairlms.data import weat_c1, list_word_sets
from fairlms.metrics import SEAT
from fairlms.models import HuggingFaceModel

metric = SEAT(n_samples=1_000, seed=0)
for ckpt in ("bert-base-uncased", "roberta-base"):
    result = metric.compute(HuggingFaceModel(ckpt, task="encoder"), weat_c1)
    print(ckpt, result.score, result.details["p_value"])

list_word_sets()   # ['weat_c1', …, 'seat_c1', …]
```

`seed` is ordinary constructor config, so a reported p-value is reproducible
from the metric's own parameters — no `np.random.seed` at the call site, and
nothing about the run recorded outside `get_params()`:

```python
SEAT(n_samples=1_000, seed=0).get_params()
# {'n_samples': 1000, 'pooling': 'mean', 'seed': 0, 'templates': None}
```

`weat_c1` *is* a `WordSets`, so user-supplied evidence goes through the exact
same call with no adapter code:

```python
mine = WordSets(target_1=names_a, target_2=names_b,
                attribute_1=pleasant, attribute_2=unpleasant)
metric.compute(model, mine)
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
from fairlms.metrics import equal_opportunity_gap, accuracy_disparity

equal_opportunity_gap(y_true, y_pred, groups, g1="A", g2="B")   # -> float
accuracy_disparity(scores_stereotype, scores_counter)           # -> float
```

Also available: `inference_bias_score`, `fair_inference_score`,
`context_based_disparity`. Use the class form
(`EqualOpportunityGap`, …) when you want the full `MetricResult` with
diagnostics.

### Dataset diagnostics

Dataset diagnostics are a separate, dataset-first API. They consume explicit
evidence and audit intent rather than a model, and return a structured report
whose components can be `ready`, `blocked`, or `not_applicable`. Missing
evidence is never represented as a score of zero.

The first diagnostic is axis-level representativeness (`b_rep`): smoothed
`KL(observed || reference)` in nats. It requires an explicit reference and its
provenance; the package does not infer a population prior from a dataset name.

```python
from fairlms.diagnostics import (
    DatasetAuditSpec,
    ReferenceDistribution,
    RepresentationEvidence,
    audit_representativeness,
)

evidence = RepresentationEvidence(
    axis="community",
    counts={"amber": 3, "teal": 1},
    source="My benchmark, evaluation split",
)
reference = ReferenceDistribution(
    axis="community",
    probabilities={"amber": 0.5, "teal": 0.5},
    source="Benchmark design specification v1",
    purpose="design_target",
    population="Intended benchmark composition",
)
spec = DatasetAuditSpec(
    target_name="my-unregistered-benchmark",
    target_kind="benchmark_dataset",
    task_family="free_text",
    design_stance="stress_test",
    references={"community": reference},
)

report = audit_representativeness(evidence, spec)
result = report.components["b_rep"]
print(result.status.value, result.value)
print(report.to_json())
```

`population_proxy` and `stress_test` use the same mathematics but not the same
interpretation. A stress test may deliberately over-sample a category; the
report therefore warns that divergence is descriptive evidence rather than an
automatic fairness failure. Use `RepresentationEvidence.from_records(...)` or
`.from_dataframe(...)` with explicit field names, support, and optional value
mapping for unfamiliar schemas. Adapters preserve the complete JSON-safe value
mapping in provenance so a recoding can be reproduced. Reference probabilities
outside an absolute `1e-9` sum tolerance are rejected; values inside that
tolerance are canonicalized onto the probability simplex and the report records
both the input sum and whether canonicalization occurred.

Start with [Preparing audit evidence](docs/preparing_audit_evidence.md) for the
evidence-layer boundary and input checklist. The detailed
[representativeness guide](docs/preparing_representativeness_evidence.md)
covers supported input paths, raw text, coverage, references, and a complete
`b_rep` example.

#### Scoring instrument audits

Scorer diagnostics audit scores that already exist at row level. They do not
infer groups from raw text or run a model/scorer to generate scores.
`score_mean_gap` is the descriptive maximum absolute group-mean difference in
the scorer's native score units. `score_rate_gap` first applies one explicit,
serialized score-to-event rule to every group, then reports the maximum absolute
group event-rate difference as a proportion. `score_wasserstein_1_gap` compares
the complete one-dimensional empirical score distributions and reports their
maximum pairwise Wasserstein-1 distance in the scorer's native score units.
`score_counterfactual_sensitivity` separately averages absolute score changes
inside complete, explicitly declared two-condition pairs.

```python
from fairlms.diagnostics import (
    DatasetAuditSpec,
    ScoreRateTransform,
    ScoredGroups,
    ScorerMeanGap,
    ScorerRateGap,
    ScorerWasserstein1Gap,
    audit_scores,
)

evidence = ScoredGroups(
    axis="cohort",
    groups=("amber", "amber", "teal", "teal"),
    scores=(0.1, 0.3, 0.8, 1.0),
    score_name="example_safety_score",
    source="Existing row-level score export v1",
    score_range=(0.0, 1.0),
)
rate_transform = ScoreRateTransform(
    event_name="score_at_or_above_policy_threshold",
    threshold=0.5,
    direction="higher",
    inclusive=True,
    provenance={"rule_source": "Example policy v1"},
)
spec = DatasetAuditSpec(
    target_name="example-score-table",
    target_kind="score_table",
    task_family="scored_rows",
    design_stance="stress_test",
    references={},
    requested_components=(
        "score_mean_gap",
        "score_rate_gap",
        "score_wasserstein_1_gap",
    ),
)

report = audit_scores(
    evidence,
    spec,
    diagnostics=(
        ScorerMeanGap(),
        ScorerRateGap(transform=rate_transform),
        ScorerWasserstein1Gap(),
    ),
)
mean_result = report.components["score_mean_gap"]
rate_result = report.components["score_rate_gap"]
w1_result = report.components["score_wasserstein_1_gap"]
print(mean_result.value, mean_result.details["unit"])  # 0.7 score_units
print(rate_result.value, rate_result.details["unit"])  # 1.0 proportion
print(w1_result.value, w1_result.details["unit"])  # 0.7 score_units
```

Paired sensitivity uses independent evidence because group marginals do not
preserve which rows are counterparts:

```python
from fairlms.diagnostics import (
    PairedScores,
    ScorerCounterfactualSensitivity,
)

paired = PairedScores(
    axis="declared_identity_intervention",
    pair_ids=("p1", "p1", "p2", "p2"),
    conditions=("baseline", "swap", "baseline", "swap"),
    scores=(0.1, 0.4, 0.8, 0.3),
    condition_roles=("baseline", "swap"),
    score_name="example_safety_score",
    source="Existing paired score export v1",
    pairing_basis="Reviewed minimal identity-token substitutions",
    score_range=(0.0, 1.0),
)
paired_spec = DatasetAuditSpec(
    target_name="example-paired-score-table",
    target_kind="score_table",
    task_family="paired_sentences",
    design_stance="stress_test",
    references={},
    requested_components=("score_counterfactual_sensitivity",),
)
paired_report = audit_scores(
    paired,
    paired_spec,
    diagnostic=ScorerCounterfactualSensitivity(),
)
print(paired_report.components["score_counterfactual_sensitivity"].value)  # 0.4
```

There is no implicit threshold, direction, or boundary rule. `score_range`
validates scores; it is neither a threshold nor a Wasserstein normalization. A
missing rate transform is `blocked`, not a zero gap. Mean, rate, and W1 gaps
answer different questions, and none is a causal claim, fairness pass/fail rule,
or error-rate metric. Paired sensitivity supports identity-isolated causal
language only if the pairs really differ solely in the declared intervention;
the package validates pair completeness but cannot verify that semantic claim
from scores. W1 and paired sensitivity are symmetric primary values and carry
no higher/lower direction.
`higher` plus `inclusive=True` is the paper-exact `score >= threshold`
definition; the report marks lower-tail and exclusive-boundary variants
separately. Treat each value as a per-dataset, per-scorer diagnostic, and rate
gaps additionally as per-rule, rather than using them to rank datasets or
unrelated scorer scales. See
[Preparing audit evidence](docs/preparing_audit_evidence.md) and the runnable
[`score_rate_gap`](examples/scorer_rate_gap_diagnostic.py) and
[`score_wasserstein_1_gap`](examples/scorer_distribution_gap_diagnostic.py)
and
[`score_counterfactual_sensitivity`](examples/scorer_counterfactual_sensitivity_diagnostic.py)
examples.

Shared loaders:

```python
from fairlms.datasets import CrowSPairs, BBQ, StereoSet
from fairlms.models import load_masked_lm

crows = CrowSPairs().load()
bbq = BBQ(categories=["Age"]).load()
loaded = load_masked_lm("bert-base-uncased")
```

Leaf runners under `definition/` are short demos of the same public API:

```bash
python -m fairlms.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cps.main
```

See also `examples/` at the repository root.

## Package layout

```
fairlms/
├── metrics/        # Public API: CrowSPairsScore, WEAT, … (all expose compute)
│   ├── data.py     #   Validated input containers (WordSets, ProbeSet, …)
│   └── functional.py #  sklearn.metrics-style functions (model-free metrics)
├── diagnostics/    # Dataset/result-table evidence, applicability, and reports
├── datasets/       # CrowSPairs, StereoSet, BBQ, BiasInBios, WinoBias, …
├── models/         # HuggingFaceModel, OpenAIModel, load_* helpers
├── utils/          # PLL / masking / association / path helpers
├── data/           # Bundled CrowS-Pairs + BBQ files; exports WEAT/SEAT word sets
├── artifacts/      # Preferred output dir for metric CSVs
└── definition/     # Internal implementations + short public-API demos (main.py)
```

Repo-root `examples/` has additional runnable snippets.

Every metric exposes the same method: `compute(...)`.

## Datasets

| Class | Source | Notes |
|-------|--------|--------|
| `CrowSPairs` | Bundled CSV under `fairlms/data/crows_pairs/` | Stereotype / anti pairs |
| `StereoSet` | Hugging Face (`stereoset` / `McGill-NLP/stereoset`) | Pairs or triples |
| `BBQ` | Bundled jsonl under `fairlms/data/bbq/` | Optional `context_condition` filter |
| `BiasInBios` | Hugging Face `LabHC/bias_in_bios` | Profession / gender helpers |
| `WinoBias` | Hugging Face `wino_bias` | Occupation direction helpers |
| `XNLIReligionPairs` | Hugging Face XNLI + templates | Religion swap pairs |

Loaders prefer canonical files in `fairlms/data/`, then fall back to legacy copies under `definition/` so existing scripts keep working.

### Bundled word sets

Association tests need four labelled term lists rather than a corpus, so they
are importable constants instead of loader classes. Each one is an
already-validated `WordSets`, ready to pass straight to `compute`:

| Name | Test | Terms |
|------|------|-------|
| `weat_c1` … `weat_c4` | Caliskan et al. (2017) C1–C4 | Race, gender, disease, age |
| `seat_c1` … `seat_c4` | May et al. (2019), expanded name lists | Same four axes |

```python
from fairlms.data import weat_c2, get_word_set, WORD_SET_LABELS

get_word_set("seat_c1")                # same objects, by name
WORD_SET_LABELS["weat_c2"]             # 'C2 – Gender (Male/Female names × Career/Family)'
```

Both families work with `WEAT` and `SEAT`; the `seat_*` lists are the larger
name sets the sentence templates were sized for. All eight have balanced target
lists, which `SEAT` requires. CEAT is not included — it consumes `ContextSets`
(terms keyed to context sentences), not four flat lists.

## Models

```python
from fairlms.models import (
    HuggingFaceModel,
    load_masked_lm,      # task="mlm"
    load_encoder,        # task="encoder"
    load_sequence_classifier,
    load_seq2seq,
    load_causal_lm,
    OpenAIModel,
)

HuggingFaceModel("roberta-base", task="sequence_classification").load()
OpenAIModel("davinci-002").load()  # needs OPENAI_API_KEY; pip install fairlms[openai]
```

Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for gated models such as Llama-2.

## Design principles

1. **One verb for metrics** — `compute` (sklearn’s `fit` / `predict` analogue).
2. **Separate metrics from datasets** — reuse the same metric on CrowS-Pairs, StereoSet, or custom data.
3. **Stable public surface** — internals under `definition/` can change without breaking user code.
4. **Book-aligned taxonomy** — `definition/{encoder_only,encoder_decoder,decoder_only}/{intrinsic_bias,extrinsic_bias}/…` mirrors the conceptual organization of the accompanying textbook.
5. **Applicability before computation** — dataset diagnostics report missing or
   incompatible evidence instead of manufacturing a numeric result.

## Development

```bash
pip install -e ".[dev]"
pytest                  # contract suite over every metric in the registry
# Run a leaf metric script:
python -m fairlms.definition.encoder_only.intrinsic_bias.similarity_based.weat.main
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
packaging bug: `fairlms.metrics` eagerly imports every metric family, so any
third-party module imported at module scope under `fairlms/definition/` is a
hard requirement of `import fairlms`. If such a dependency is only listed in an
extra, `pip install fairlms` produces a package that cannot be imported — while
every local test still passes, because the developer's environment already has
it. [`tests/test_packaging.py`](tests/test_packaging.py) also guards this
statically, walking the AST and naming the offending file.

### Publishing an edit

Edits reach users through a tagged release, not through `main`:

1. Edit locally — your editable install picks changes up immediately.
2. `pytest` — the contract suite catches API breakage before it ships.
3. Commit and push. CI verifies the matrix and the clean install.
4. Bump [`fairlms/_version.py`](fairlms/_version.py), then tag and push the tag.
5. The publish workflow uploads to PyPI; users get it with
   `pip install --upgrade fairlms`.

Metric result CSVs should go under `fairlms/artifacts/` (via `fairlms.utils.results_to_csv`); leaf-local `*_results.csv` files are gitignored.

## License

MIT
