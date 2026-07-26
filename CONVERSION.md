# Converting fairLLMs to a sklearn-style library

**Where we are:** Phase 1 is done. Shared `datasets/`, `models/`, and `utils/` exist; metric math still lives as leaf scripts under `definition/`.

**Goal:** Researchers import a metric and call one method — without knowing the folder path.

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
| Dataset loaders | `fairLLMs/datasets/` (`CrowSPairs`, `StereoSet`, `BBQ`, …) |
| Model adapters | `fairLLMs/models/` (`HuggingFaceModel`, `OpenAIModel`, `load_*`) |
| Shared helpers | `fairLLMs/utils/` (PLL, masking, association, paths) |
| Canonical data | `fairLLMs/data/` (CrowS CSV, BBQ jsonl) |
| Install + docs | `pyproject.toml`, `README.md` |

---

## Remaining work

### Phase 2 — Metric API (core conversion)

1. **Add contracts**
   - `fairLLMs/metrics/base.py`: `FairnessMetric` ABC with `compute(model, dataset=None, **kwargs) -> MetricResult`
   - `MetricResult` with at least `.score` (optional `.details`, `.by_category`)

2. **Wrap existing `compute_*` functions** — do not rewrite the math
   - Create `fairLLMs/metrics/` (flat public surface)
   - One class per metric, e.g. `CrowSPairsScore` → calls `cps.compute_cps(...)`
   - Pilot first: **CPS**, then **LPBS**, then **WEAT/SEAT**
   - Roll through the remaining ~30 leaves with the same pattern

3. **Wire adapters inside `compute`**
   - Accept `HuggingFaceModel` / `LoadedModel` / raw HF objects
   - Accept `CrowSPairs()`-style datasets or preloaded example lists
   - Map dataset schemas → whatever the existing `compute_*` expects

4. **Export cleanly**
   ```python
   from fairLLMs.metrics import CrowSPairsScore, LogProbabilityBiasScore, WEAT
   ```

### Phase 3 — Book taxonomy aliases

5. **Add `fairLLMs/definitions/`** (plural) that **re-exports** metric classes — no duplicated logic
   ```python
   from fairLLMs.definitions.intrinsic_bias import CrowSPairsScore
   ```
6. Attach metadata on classes (`bias_type`, `architectures`) so metrics can be listed/filtered later
7. Keep `definition/` (singular) working as a compatibility shim until callers migrate

### Phase 4 — Demote scripts

8. Turn each leaf `main.py` into a short example that uses the public API, **or** move to `examples/`
9. Update per-metric READMEs to show `metric.compute(...)`
10. Optionally keep `python -m fairLLMs.definition....main` via thin wrappers for one release

### Phase 5 — Library polish

11. Smoke tests: import + `compute` contract (small `n_max`, CPU) per wrapped metric
12. Metric registry: `list_metrics()` / `get_metric(name)`
13. Optional deps already sketched (`[openai]`, `[dev]`) — document which metrics need which extras
14. Deduplicate leftover CrowS/BBQ copies under `definition/**/data/` once all callers use `fairLLMs.data`
15. Decide fate of local-only corpora (`red_pill_corpus.csv`, BBQ zips) — download script or docs, not git

---

## Suggested order

```
base.py + MetricResult
    → wrap CPS (end-to-end reference)
        → wrap LPBS, WEAT
            → definitions/ re-exports
                → remaining metrics
                    → examples + tests + cleanup
```

## Success check

A new user never opens `definition/encoder_only/.../cps/`. They only need:

```python
from fairLLMs.metrics import CrowSPairsScore
```

and a single method: **`compute`**.
