# Mitigating bias

`fairlms.mitigation` ships fourteen components across the four intervention
categories: **pre** (4), **in** (4), **intra** (3) and **post** (3).

Mitigators adopt the same contract as metrics. Each declares an intervention
category, an access level, supported architectures, the capabilities it needs
from a model, and the evidence containers it consumes. Applicability is decided
from those declarations by the same matcher the metrics use.

```python
from fairlms.mitigation import MITIGATOR_REGISTRY, list_by_category

list_by_category("post")
# ['fairness_aware_reranking', 'group_aware_thresholding', 'score_calibration']

mitigator = MITIGATOR_REGISTRY["score_calibration"](method="isotonic")
outcome = mitigator.apply(model, evidence)     # -> MitigationResult
```

Configuration goes to `__init__`, data goes to `apply`. That is the same
scikit-learn convention the metrics follow, so `get_params`, `set_params` and
`sklearn.base.clone` all work.

## Refused, not approximated

An unsatisfiable pairing raises before any compute, and the message names the
missing item rather than reporting "not applicable":

```python
from fairlms.mitigation import IterativeNullspaceProjection
from fairlms.models import OpenAIModel

IterativeNullspaceProjection().apply(OpenAIModel(), evidence)
# TypeError: IterativeNullspaceProjection requires `hidden_states`;
# OpenAIModel('davinci-002') provides only `free_generation`, `token_logprobs`.
# Load the checkpoint locally with HuggingFaceModel(name, task=...) so the
# quantity is observable.
```

The same mechanism covers the metrics, so pointing an API-served decoder at SEAT
is refused the same way, before any weights are pulled.

Access levels are ordered `black_box` < `gray_box` < `white_box`, and a component
declares the **minimum** it needs:

| Level | What the deployment must expose |
|---|---|
| `black_box` | inputs and outputs only |
| `gray_box` | activations and logits, no weight updates |
| `white_box` | parameters and gradients |

## What each category hands back

`MitigationResult.result` differs by category, and the type is enforced:

| Category | `.result` contains |
|---|---|
| `pre` | transformed evidence, or per-row weights, plus transform provenance |
| `in` | a loss/regularizer callable composable with a normal training loop |
| `intra` | a `ModelAdapter` wrapping the original |
| `post` | a fitted decision rule: calibrator, thresholds, or reranker |

Every result carries `provenance` (configuration, declarations, library version)
and serializes with `to_dict()` / `to_json()`. A live payload — a model or a
closure — is recorded as a typed stub rather than stringified, so a report never
contains something that looks like data but is not.

`MitigationResult` has no `__float__`. Unlike `MetricResult` it is not a scalar.

## Post-processing: fit a rule on scores

Post-processing needs no model at all, which makes it the quickest way to see the
contract end to end.

```python
from fairlms.diagnostics import LabeledScoredGroups, ScoredGroups
from fairlms.mitigation import GroupAwareThresholding

evidence = LabeledScoredGroups(
    scored=ScoredGroups(
        axis="gender",
        groups=["f", "f", "f", "f", "m", "m", "m", "m"],
        scores=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
        score_name="p_hire",
        source="validation-split",
        score_range=[0.0, 1.0],
    ),
    labels=["no", "no", "yes", "yes", "no", "no", "yes", "yes"],
    label_name="outcome",
    positive_label="yes",          # mandatory, never inferred
)

rule = GroupAwareThresholding().apply(None, evidence).result
rule["thresholds"]         # {'f': 0.3, 'm': 0.8}
rule["achieved_tpr_gap"]   # 0.0
rule["achieved_utility"]   # 1.0
```

`positive_label` is required because equal opportunity is defined as an equal
true-positive rate *for the positive class*: which label is positive is part of
the question, not a property of the data. The rule also reports the gap it
**actually achieved**, since with finite data the rates rarely match exactly.

`achieved_utility` is reported alongside it because a closed gap is not on its
own evidence of a good rule. Rejecting every candidate equalises every rate
perfectly, and so does accepting every candidate; both score a gap of zero.
`GroupAwareThresholding` therefore minimises the gap first and then maximises
Youden's J, `mean(TPR) - mean(FPR)`, which is zero for both of those degenerate
rules and positive for any rule that actually discriminates. Read the two
numbers together: a gap of `0.0` with a utility of `0.0` means the search
equalised the rates by refusing to decide anything.

!!! note
    `LabeledScoredGroups` presumes a **binary** label. Under multi-class labels
    equal opportunity is defined per class, and this type does not model that.

## Intra-processing: the result is a model

An intra-processing result *is* a `ModelAdapter`, so any registered metric
re-evaluates the mitigated model with no modification:

```python
from fairlms.metrics import METRIC_REGISTRY
from fairlms.mitigation import SubspaceProjection

mitigated = SubspaceProjection().apply(base_model, pairs).result

weat = METRIC_REGISTRY["weat"]()        # taken straight from the registry
before = float(weat.compute(base_model, word_sets))
after = float(weat.compute(mitigated, word_sets))
```

!!! warning "Report removal relative to the probe family"
    A projection removes what a *linear* probe of that family could read. Gonen
    and Goldberg (2019) showed this can **hide** bias rather than remove it: the
    geometry survives in clusters a linear probe no longer detects. Every result
    records `probe_family` and a `removal_claim` in provenance, and
    `iterative_nullspace_projection` reports the probe accuracy it actually
    reached. Do not describe any of this as removal in absolute terms.

## In-processing: a loss term, not a trainer

There is **no trainer in the core package**. In-processing components return a
callable you compose into your own objective; you keep the optimizer, the
schedule and the data loader.

```python
from fairlms.mitigation import AdversarialDebiasing

component = AdversarialDebiasing(lambda_=1.0).apply(None, vectors).result

loss = task_loss + component(pooled_states, attribute_labels)
loss.backward()
```

`influence_guided_suppression` is a deliberate reduction of IF-Guide: it ships
the objective only and consumes **precomputed** influence scores. Estimating
influence is an optional backend, never a core dependency.

## Pre-processing: transform the data

Pre-processing transforms evidence and refuses what it cannot handle:

```python
from fairlms.mitigation import (
    CorpusWithLexicon, CounterfactualDataAugmentation, SwapLexicon, TextRecords)

evidence = CorpusWithLexicon(
    records=TextRecords(texts=["He is a nurse."], source="my-corpus"),
    lexicon=SwapLexicon(
        axis="gender", pairs=[("he", "she")], source="my-lexicon"),
)
result = CounterfactualDataAugmentation().apply(None, evidence)
list(result.result.texts)
# ['He is a nurse.', 'She is a nurse.']
```

A record containing no lexicon term cannot be rewritten, and is **refused rather
than dropped**: silently emitting only the augmentable subset would skew the
corpus toward exactly the rows that mention the protected attribute. Pass
`on_unrewritable="keep"` to opt into keeping it, which is recorded in provenance.

`debiasing_prompt` is pre-processing but applies only to generative models, so
it is declared decoder-only and encoder-decoder and refused elsewhere.

## Comparing before and after

```python
from fairlms.mitigation import compare_before_after

report = compare_before_after(
    base_model,
    mitigated_model,
    metrics={"weat": word_sets, "equal_opportunity_gap": predictions},
    utility={"accuracy": my_accuracy_fn},
)
report.to_dict()
```

Three rules the report enforces:

- **Fairness and utility are reported separately.** They are not commensurable,
  and combining them hides the trade-off that is the reason to measure both.
- **Intrinsic and extrinsic are reported separately.** An intrinsic improvement
  is not evidence of an extrinsic one.
- **There is no composite "mitigation effectiveness score."** No defensible
  aggregation exists, so `ComparisonReport` has no `__float__` and exposes no
  overall number.

A metric that raises is recorded with its error rather than dropped, so a partial
comparison is visibly partial.

## Not provided

See the [mitigator registry](../registry/mitigation.md) for the full list of
methods named in the paper and book that this library deliberately does **not**
implement, recorded so that omission is not mistaken for oversight.
