# Divergences from released runners

Where the original reference implementations and FairLMs disagree, the
difference is deliberate and recorded here.

## Unavailable evidence is not a zero

The systematic divergence is in the diagnostics layer. Several released bias
runners return a numeric result — usually `0.0` — when a required input is
absent: no reference distribution, no threshold rule, an incomplete
counterfactual pair. FairLMs refuses.

`ComponentResult` enforces this structurally: a non-`ready` component must carry
`value=None`, and constructing one with a numeric sentinel raises `ValueError`.
A report therefore distinguishes "the groups do not differ" (`ready`, value
`0.0`) from "you have not supplied what this component needs" (`blocked`,
value `None`).

| Input condition | Typical released behaviour | FairLMs |
|---|---|---|
| No score-to-event rule supplied to a rate gap | returns `0.0` | `blocked`, `value=None` |
| No reference distribution for an axis | infers a uniform or corpus prior | `blocked`, `value=None` |
| Incomplete counterfactual pair | drops the row silently | pair-completeness validation, then `blocked` |
| Reference probabilities not summing to 1 | renormalizes silently | rejected outside a `1e-9` tolerance; canonicalization recorded in the report |

## Known metric-level differences

| Metric | Difference | Rationale |
|---|---|---|
| AUL / AULA | attention weights are read with the attention implementation forced to `eager` | SDPA and flash-attention backends silently return no attentions even when `output_attentions=True`, which would otherwise yield AUL scores mislabelled as AULA |
| WEAT / SEAT / CEAT | the permutation-test seed is explicit constructor configuration (`seed=`, default `None`) | it appears in `get_params()`, so a reported p-value can be pinned and reproduced rather than depending on ambient global RNG state |

## Status

This page is not yet exhaustive. Each row should eventually link to a golden
fixture pinning both behaviours so that a change in either implementation fails
CI; the diagnostics rows are covered by the contract suite today, the
metric-level rows are not.
