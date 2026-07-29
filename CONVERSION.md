# Converting fairLLMs to a sklearn-style library

**Where we are:** Phases 1, 2, and 4 are done. Phase 3 (book-taxonomy
`definitions/` aliases) was **skipped** — users import from
`fairLLMs.metrics` only.

**Goal:** Researchers import a metric and call one method — without knowing
the folder path.

```python
from fairLLMs.metrics import CrowSPairsScore
from fairLLMs.datasets import CrowSPairs
from fairLLMs.models import HuggingFaceModel

result = CrowSPairsScore().compute(
    model=HuggingFaceModel("bert-base-uncased", task="mlm"),
    dataset=CrowSPairs(),
)
print(result.score)
```

---

## Already done

| Piece | Location |
|-------|----------|
| Dataset loaders | `fairLLMs/datasets/` |
| Model adapters | `fairLLMs/models/` |
| Shared helpers | `fairLLMs/utils/` |
| Metric API | `fairLLMs/metrics/` (33 classes, all expose `compute`) |
| Leaf demos | `fairLLMs/definition/**/main.py` use the public API |
| Examples | `examples/` |
| Per-metric docs | leaf `README.md` files show Public API first |
| Canonical data | `fairLLMs/data/` |
| Install + docs | `pyproject.toml`, `README.md` |

---

## Remaining work

### Phase 3 — Book taxonomy aliases — SKIPPED

Not planned. Prefer:

```python
from fairLLMs.metrics import CrowSPairsScore
```

### Phase 4 — Demote scripts — DONE

- Leaf `main.py` files are short public-API demos
- Leaf READMEs document `metric.compute(...)`
- Repo-root `examples/` mirrors the preferred usage

### Phase 5 — Library polish

1. Smoke tests: import + `compute` contract (small `n_max`, CPU)
2. Document which metrics need which extras (`openai`, etc.)
3. Deduplicate leftover CrowS/BBQ copies under `definition/**/data/`
4. Document/handle local-only corpora (`red_pill_corpus.csv`, BBQ zips)

---

## Success check

A new user never needs to know `definition/encoder_only/.../cps/`. They only need:

```python
from fairLLMs.metrics import CrowSPairsScore
```

and a single method: **`compute`**.
