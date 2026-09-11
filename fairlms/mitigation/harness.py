"""Before/after comparison across a declared metric set.

Comparing a base and a mitigated system "needs no new code" because an
intra-processing result *is* a :class:`~fairlms.models.base.ModelAdapter`. This
module is only the thin helper that runs a declared metric set against both and
reports the deltas.

Three rules, all of them refusals:

* **Fairness and utility deltas are reported separately.** They are not
  commensurable and combining them hides the trade-off that is the entire point
  of measuring both.
* **Intrinsic and extrinsic deltas are reported separately.** An intrinsic
  improvement is not evidence of an extrinsic one.
* **There is no composite "mitigation effectiveness score."** No defensible
  aggregation exists and the paper does not claim one. :class:`ComparisonReport`
  therefore has no ``__float__`` and exposes no overall number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from fairlms.metrics import METRIC_REGISTRY

__all__ = ["ComparisonReport", "MetricDelta", "compare_before_after"]


@dataclass(frozen=True, kw_only=True)
class MetricDelta:
    """One metric's score before and after mitigation."""

    metric: str
    bias_type: str
    before: Optional[float]
    after: Optional[float]
    error: Optional[str] = None

    @property
    def delta(self) -> Optional[float]:
        """``after - before``, or ``None`` when either side failed."""
        if self.before is None or self.after is None:
            return None
        return self.after - self.before

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "bias_type": self.bias_type,
            "before": self.before,
            "after": self.after,
            "delta": self.delta,
            "error": self.error,
        }


@dataclass(frozen=True, kw_only=True)
class ComparisonReport:
    """Before/after deltas, partitioned and never aggregated.

    Deliberately not float-convertible: there is no single number here.
    """

    fairness: Sequence[MetricDelta] = ()
    utility: Sequence[MetricDelta] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def intrinsic(self) -> tuple:
        """Fairness deltas from intrinsic metrics."""
        return tuple(d for d in self.fairness if d.bias_type == "intrinsic")

    @property
    def extrinsic(self) -> tuple:
        """Fairness deltas from extrinsic metrics."""
        return tuple(d for d in self.fairness if d.bias_type == "extrinsic")

    def to_dict(self) -> dict:
        return {
            "fairness": {
                "intrinsic": [d.to_dict() for d in self.intrinsic],
                "extrinsic": [d.to_dict() for d in self.extrinsic],
            },
            "utility": [d.to_dict() for d in self.utility],
            "provenance": dict(self.provenance),
            "note": (
                "Fairness and utility are reported separately, as are intrinsic "
                "and extrinsic fairness. There is no composite effectiveness "
                "score: no defensible aggregation exists."
            ),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(
            self.to_dict(), allow_nan=False, indent=indent, sort_keys=True
        )


def compare_before_after(
    base_model: Any,
    mitigated_model: Any,
    *,
    metrics: Mapping[str, Any],
    utility: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> ComparisonReport:
    """Run a declared metric set against a base and a mitigated system.

    Parameters
    ----------
    base_model, mitigated_model:
        Anything the metrics accept. ``mitigated_model`` is typically
        ``result.result`` from an intra-processing mitigator, which is a
        :class:`~fairlms.models.base.ModelAdapter` and so needs no adaptation.
    metrics:
        Mapping of registry metric name -> the evidence to score it on. Named
        explicitly rather than "all applicable metrics": which metrics
        constitute the fairness claim is the caller's declaration.
    utility:
        Mapping of ``name -> callable(model) -> float`` for task performance.
        Reported separately and never combined with the fairness deltas.
    provenance:
        Extra provenance recorded verbatim in the report.

    Returns
    -------
    ComparisonReport
        Partitioned deltas. A metric that raises is recorded with its error
        rather than dropped, so a partial comparison is visibly partial.
    """
    unknown = sorted(set(metrics) - set(METRIC_REGISTRY))
    if unknown:
        raise KeyError(
            f"unknown metric(s): {', '.join(unknown)}. Available: "
            f"{', '.join(sorted(METRIC_REGISTRY))}"
        )

    fairness = []
    for name in sorted(metrics):
        cls = METRIC_REGISTRY[name]
        metric, evidence = cls(), metrics[name]
        before = after = None
        error = None
        try:
            before = float(metric.compute(base_model, evidence))
            after = float(metric.compute(mitigated_model, evidence))
        except Exception as exc:  # recorded, not swallowed
            error = f"{type(exc).__name__}: {exc}"
        fairness.append(
            MetricDelta(
                metric=name,
                bias_type=cls.bias_type,
                before=before,
                after=after,
                error=error,
            )
        )

    utility_deltas = []
    for name in sorted(utility or {}):
        function = utility[name]
        before = after = None
        error = None
        try:
            before = float(function(base_model))
            after = float(function(mitigated_model))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        utility_deltas.append(
            MetricDelta(
                metric=name,
                bias_type="utility",
                before=before,
                after=after,
                error=error,
            )
        )

    return ComparisonReport(
        fairness=tuple(fairness),
        utility=tuple(utility_deltas),
        provenance=dict(provenance or {}),
    )
