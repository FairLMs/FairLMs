# Auditing a dataset

Diagnostics audit the evaluation data and the scoring instrument directly,
rather than inferring their quality from the scores a model happens to produce.
They take explicit evidence plus a `DatasetAuditSpec` — never a model — and
return a `DiagnosticReport` whose components carry a status alongside any value.

## The four states

| State | Meaning |
|---|---|
| `ready` | the component was computed |
| `blocked` | a required input — evidence, reference or rule — was not supplied |
| `not_applicable` | the evidence cannot support this component |
| `failed` | the computation raised |

`blocked` and `not_applicable` carry `value=None`, never `0.0`. The report's own
status aggregates its components into `success`, `partial`, `blocked`,
`not_applicable` or `failed`.

## Representativeness

`b_rep` is smoothed `KL(observed ‖ reference)` in nats over one axis. It
requires an explicit reference distribution *and* its provenance; the package
will not infer a population prior from a dataset's name.

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
    target_name="my-benchmark",
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

`RepresentationEvidence.from_records(...)` and `.from_dataframe(...)` accept
unfamiliar schemas with explicit field names, support and optional value
mapping; the mapping is preserved in provenance so a recoding is reproducible.

## Design stance matters

`population_proxy` and `stress_test` use the same mathematics and *not* the same
interpretation. A stress test may deliberately over-sample a category, so the
report warns that divergence is descriptive evidence rather than an automatic
fairness failure. The spec records the stance and each reference's purpose, and
the report serializes both.

## Auditing scores

Four components describe one scoring instrument, at row level. None of them runs
a model or infers groups from raw text.

| Component | Question | Unit |
|---|---|---|
| `score_mean_gap` | largest difference in group means | native score units |
| `score_rate_gap` | largest difference in group event rates, after one explicit score-to-event rule | proportion |
| `score_wasserstein_1_gap` | largest pairwise W₁ between full empirical distributions | native score units |
| `score_counterfactual_sensitivity` | mean absolute score change within complete declared pairs | native score units |

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
    diagnostics=(ScorerMeanGap(), ScorerRateGap(transform=rate_transform), ScorerWasserstein1Gap()),
)
for name, component in report.components.items():
    print(name, component.status.value, component.value, component.details.get("unit"))
```

Paired sensitivity needs its own evidence type, because group marginals do not
record which rows are counterparts:

```python
from fairlms.diagnostics import PairedScores, ScorerCounterfactualSensitivity

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
    paired, paired_spec, diagnostic=ScorerCounterfactualSensitivity()
)
print(paired_report.components["score_counterfactual_sensitivity"].value)   # 0.4
```

## What these numbers are not

There is no implicit threshold, direction or boundary rule anywhere.
`score_range` validates scores; it is neither a threshold nor a W₁
normalization. Mean, rate and W₁ gaps answer different questions, and none is a
causal claim, a pass/fail rule or an error-rate metric. Paired sensitivity
supports identity-isolated causal language only if the pairs really differ
solely in the declared intervention — the package validates pair completeness
but cannot verify that semantic claim from scores alone.

Treat each value as a per-dataset, per-scorer diagnostic (and rate gaps
additionally as per-rule) rather than as a way to rank datasets or unrelated
scorer scales.

Further reading: [Preparing audit evidence](../preparing_audit_evidence.md) and
the [representativeness guide](../preparing_representativeness_evidence.md).
