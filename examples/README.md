# Examples

Short scripts that use the **public** fairLLMs API:

```python
from fairLLMs.metrics import CrowSPairsScore
```

Leaf demos under `fairLLMs/definition/**/main.py` also call this API and remain
runnable via `python -m fairLLMs.definition....main`. Prefer these examples
(or the Public API sections in each metric README) for new code.

## Quick start

```bash
pip install -e .
python examples/equal_opportunity_gap.py
python examples/crows_pairs_score.py
```

## Scripts

| Script | Metric(s) |
|--------|-----------|
| `crows_pairs_score.py` | CrowSPairsScore (CPS) |
| `log_probability_bias.py` | LogProbabilityBiasScore (LPBS) |
| `weat.py` | WEAT |
| `equal_opportunity_gap.py` | EqualOpportunityGap |
| `accuracy_disparity.py` | AccuracyDisparity |

Heavier metrics (seq2seq, OpenAI, full BBQ) are documented in their leaf
READMEs; use `list_metrics()` to discover class names.
