# fairLLMs

Fairness definitions and bias metrics for large language models — a companion library for studying and evaluating bias in LMs.

The long-term goal is a **stable, sklearn-style API**: import a metric, call `compute(...)`, and get a result — without caring where the implementation lives.

> **Status:** Phases 1–2 and 4. Import metrics from `fairLLMs.metrics` and call
> `compute(...)`. Leaf `main.py` files and `examples/` are short public-API demos.
> Implementation math still lives under `fairLLMs/definition/`.

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[openai]"   # API-backed decoder metrics
pip install -e ".[dev]"      # pytest
pip install -e ".[all]"      # openai + common extras
```

Requires Python ≥ 3.9. Core deps: `torch`, `transformers`, `datasets`, `numpy`, `pandas`, `scipy`.

## Quick start

```python
from fairLLMs.metrics import CrowSPairsScore, LogProbabilityBiasScore, list_metrics
from fairLLMs.datasets import CrowSPairs
from fairLLMs.models import HuggingFaceModel

model = HuggingFaceModel("bert-base-uncased", task="mlm")
result = CrowSPairsScore().compute(model=model, dataset=CrowSPairs(n_max=50))
print(result.score, result.by_category)

# Discover metrics
print(list_metrics())
```

Shared loaders:

```python
from fairLLMs.datasets import CrowSPairs, BBQ, StereoSet
from fairLLMs.models import load_masked_lm

crows = CrowSPairs().load()
bbq = BBQ(categories=["Age"]).load()
loaded = load_masked_lm("bert-base-uncased")
```

Leaf runners under `definition/` are short demos of the same public API:

```bash
python -m fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cps.main
```

See also `examples/` at the repository root.

## Package layout

```
fairLLMs/
├── metrics/        # Public API: CrowSPairsScore, WEAT, … (all expose compute)
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
| `CrowSPairs` | Bundled CSV under `fairLLMs/data/crows_pairs/` | Stereotype / anti pairs |
| `StereoSet` | Hugging Face (`stereoset` / `McGill-NLP/stereoset`) | Pairs or triples |
| `BBQ` | Bundled jsonl under `fairLLMs/data/bbq/` | Optional `context_condition` filter |
| `BiasInBios` | Hugging Face `LabHC/bias_in_bios` | Profession / gender helpers |
| `WinoBias` | Hugging Face `wino_bias` | Occupation direction helpers |
| `XNLIReligionPairs` | Hugging Face XNLI + templates | Religion swap pairs |

Loaders prefer canonical files in `fairLLMs/data/`, then fall back to legacy copies under `definition/` so existing scripts keep working.

## Models

```python
from fairLLMs.models import (
    HuggingFaceModel,
    load_masked_lm,      # task="mlm"
    load_encoder,        # task="encoder"
    load_sequence_classifier,
    load_seq2seq,
    load_causal_lm,
    OpenAIModel,
)

HuggingFaceModel("roberta-base", task="sequence_classification").load()
OpenAIModel("davinci-002").load()  # needs OPENAI_API_KEY; pip install fairLLMs[openai]
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
# Run a leaf metric script:
python -m fairLLMs.definition.encoder_only.intrinsic_bias.similarity_based.weat.main
```

Metric result CSVs should go under `fairLLMs/artifacts/` (via `fairLLMs.utils.results_to_csv`); leaf-local `*_results.csv` files are gitignored.

## License

MIT
