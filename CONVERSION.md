# Converting fairlms to a sklearn-style library

**Where we are:** Phases 1, 2, 4, 6, 7 and 8 are done; Phase 5 is in progress.
Phase 3 (book-taxonomy `definitions/` aliases) was **skipped** — users import
from `fairlms.metrics` only.

**Goal:** Researchers import a metric and call one method — without knowing
the folder path.

```python
from fairlms.metrics import CrowSPairsScore
from fairlms.datasets import CrowSPairs
from fairlms.models import HuggingFaceModel

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
| Dataset loaders | `fairlms/datasets/` |
| Model adapters | `fairlms/models/` |
| Shared helpers | `fairlms/utils/` |
| Metric API | `fairlms/metrics/` (33 classes, all expose `compute`) |
| Leaf demos | `fairlms/definition/**/main.py` use the public API |
| Examples | `examples/` |
| Per-metric docs | leaf `README.md` files show Public API first |
| Canonical data | `fairlms/data/` |
| Install + docs | `pyproject.toml`, `README.md` |

---

## Remaining work

### Phase 3 — Book taxonomy aliases — SKIPPED

Not planned. Prefer:

```python
from fairlms.metrics import CrowSPairsScore
```

### Phase 4 — Demote scripts — DONE

- Leaf `main.py` files are short public-API demos
- Leaf READMEs document `metric.compute(...)`
- Repo-root `examples/` mirrors the preferred usage

### Phase 5 — Library polish — IN PROGRESS

1. ~~Smoke tests: import + `compute` contract~~ — **done**, `tests/`
   (`test_common.py` is a `check_estimator`-style conformance suite run over
   every entry in `METRIC_REGISTRY`)
2. Document which metrics need which extras (`openai`, etc.) — CR / CTF / BA
   require `OPENAI_API_KEY`; all others run locally
3. Deduplicate leftover CrowS/BBQ copies under `definition/**/data/`
4. Document/handle local-only corpora (`red_pill_corpus.csv`, BBQ zips)

### Phase 6 — Uniform call contract — DONE (all 33 metrics)

The `compute(model, dataset, **kwargs)` signature was one method name over 33
private calling conventions: unknown kwargs were silently ignored, and 21 sites
read the same knob from both `__init__` and `kwargs`. All 33 metrics across all
11 families now share one shape:

* **Config is keyword-only in `__init__`**, stored verbatim →
  `get_params()` / `set_params()` / `__repr__` on `FairnessMetric`
  (sklearn's `BaseEstimator` protocol).
* **Data is positional and typed** — 17 containers in
  `fairlms/metrics/data.py` validate shape at construction.
* **`compute(model, data)`** — exactly two positional arguments, everywhere.
* **Unknown kwargs raise `TypeError`** via `_reject_unknown_kwargs`.
* Legacy keywords (`T1_terms=`, `prompts=`, `y_true=`, …) still work, with a
  `DeprecationWarning` naming the replacement.

```python
from fairlms.metrics import WEAT, WordSets
WEAT(pooling="mean", n_samples=10_000).compute(model, WordSets(t1, t2, a1, a2))
```

The 5 model-free metrics are additionally exposed as plain functions in
`fairlms/metrics/functional.py`, in the style of `sklearn.metrics`:

```python
from fairlms.metrics import equal_opportunity_gap
equal_opportunity_gap(y_true, y_pred, groups, g1="A", g2="B")   # -> float
```

Making parameters explicit surfaced five dead-config bugs the `**kwargs` shape
had been hiding, all fixed:

| Bug | Effect |
|-----|--------|
| `compute_weat` ignored `n_samples` | hardcoded 10 000 |
| `compute_seat` ignored `pooling` | hardcoded `"mean"`; `SEAT(pooling="cls")` lied |
| `compute_seat` never passed `device` | SEAT broken on CUDA |
| `compute_ba` never received a model name | `OpenAIModel("…")` silently ignored |
| `compute_ba`'s `max_new_tokens` | accepted, never read |

Two metrics also stopped demanding facts the model already knows: `NaturalIndirectEffect`
derives `n_layers`/`n_heads`/`head_dim` from `model.config`, and
`StereotypicalValueAttribution` derives both the shape and the stereotype
`direction` (previously a hand-supplied 768-vector).

**Remaining:** `tests/test_common.py` runs the contract over
`METRIC_REGISTRY`, so new metrics are covered automatically.

### Phase 7 — Package rename `fairLLMs` → `fairllms` — DONE

All-lowercase per PEP 8 and sklearn convention. The old camelCase name also
masked import-case bugs: macOS's case-insensitive filesystem accepted
`import fairLLMs` and `import fairllms` interchangeably, while a case-sensitive
Linux CI would only accept the exact spelling.

No `fairLLMs` compatibility shim was provided, deliberately:

1. A `fairLLMs.py` shim cannot sit reliably beside a `fairllms/` package on a
   case-insensitive filesystem — the two names collide on disk.
2. In this environment the name `fairLLMs` is already claimed by a *different*
   editable install (`~/Projects/FairnessDefinitionsLLMs`), so a shim there would
   create genuine ambiguity about which project `import fairLLMs` means.

### Phase 8 — Project rename `fairllms` → `fairlms` — DONE

The project and GitHub repository are titled **FairLMs**; the PyPI distribution
and the import package are both **`fairlms`**. Dropping the doubled `l` reads
better and, unlike `fairllms`, does not case-fold onto the unrelated `fairLLMs`
install noted in Phase 7 — so `import fairlms` is unambiguous even on macOS.

Only the display title is camel-cased. Everything a user types is lowercase:

* `pip install fairlms` — installs the distribution
* `import fairlms` — the module path, all-lowercase per PEP 8

Update imports to `fairlms` and reinstall with `pip install -e .`.

---

## Success check

A new user never needs to know `definition/encoder_only/.../cps/`. They only need:

```python
from fairlms.metrics import CrowSPairsScore
```

and a single method: **`compute`**.
