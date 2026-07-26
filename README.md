# fairLLMs

Fairness definitions and bias metrics for large language models — a companion library for studying and evaluating bias in LMs.

The long-term goal is a **stable, sklearn-style API**: import a metric, call `compute(...)`, and get a result — without caring where the implementation lives.

> **Status:** Phase 1 (shared infrastructure). Metric math still lives under `fairLLMs/definition/`; the public `metrics` layer with a uniform `compute()` interface is next.

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

## Quick start (Phase 1)

Shared loaders and model adapters are ready to use today:

```python
from fairLLMs.datasets import CrowSPairs, BBQ, StereoSet
from fairLLMs.models import load_masked_lm, HuggingFaceModel

# Datasets
crows = CrowSPairs().load()          # stereotype / anti_stereotype pairs
bbq = BBQ(categories=["Age"]).load() # BBQ jsonl rows
# stereoset = StereoSet().load()     # downloads from Hugging Face

# Models
loaded = load_masked_lm("bert-base-uncased")
tokenizer, model, device = loaded.tokenizer, loaded.model, loaded.device

# Or more generally:
hf = HuggingFaceModel("bert-base-uncased", task="mlm").load()
```

Existing metric runners (pre-API) still work, for example:

```bash
python -m fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cps.main
```

## Package layout

```
fairLLMs/
├── datasets/       # CrowSPairs, StereoSet, BBQ, BiasInBios, WinoBias, …
├── models/         # HuggingFaceModel, OpenAIModel, load_* helpers
├── utils/          # PLL / masking / association / path helpers
├── data/           # Bundled CrowS-Pairs + BBQ files
├── artifacts/      # Preferred output dir for metric CSVs
└── definition/     # Book taxonomy + current metric implementations
    ├── encoder_only/
    ├── encoder_decoder/
    └── decoder_only/
```

### Intended public API (upcoming)

```python
from fairLLMs.metrics import CrowSPairsScore
from fairLLMs.datasets import CrowSPairs
from fairLLMs.models import HuggingFaceModel

model = HuggingFaceModel("bert-base-uncased", task="mlm")
result = CrowSPairsScore().compute(model=model, dataset=CrowSPairs())
print(result.score)
```

Every metric will expose the same method: `compute(...)`.

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
