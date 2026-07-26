"""Encoder-only extrinsic metric wrappers."""

from __future__ import annotations

from typing import Any

from fairLLMs.definition.encoder_only.extrinsic_bias.context_based_disparity.context_based import (  # noqa: E501
    compute_s_amb,
    compute_s_dis,
)
from fairLLMs.definition.encoder_only.extrinsic_bias.equal_opportunity.equal_opportunity import (  # noqa: E501
    gap_g_y,
)
from fairLLMs.definition.encoder_only.extrinsic_bias.fair_inference.fair_inference import (  # noqa: E501
    evaluate_fair_inference,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_examples, require_kwargs


class FairInferenceScore(FairnessMetric):
    """Dev et al. fair-inference rates over NLI prediction dicts.

    ``dataset`` (or ``predictions``) is a sequence of dicts with class
    probability keys including ``neutral``.
    """

    name = "fair_inference_score"
    bias_type = "extrinsic"
    architectures = ("encoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        predictions = kwargs.get("predictions")
        if predictions is None:
            predictions = get_examples(dataset)
        if not predictions:
            raise ValueError("predictions=... or dataset of prediction dicts is required")
        nn, fn, t05, t07 = evaluate_fair_inference(predictions)
        return MetricResult(
            score=float(fn),
            details={"nn": nn, "fn": fn, "t05": t05, "t07": t07},
        )


class EqualOpportunityGap(FairnessMetric):
    """TPR gap between two groups (equal opportunity)."""

    name = "equal_opportunity_gap"
    bias_type = "extrinsic"
    architectures = ("encoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        y_true = kwargs.get("y_true")
        y_pred = kwargs.get("y_pred")
        groups = kwargs.get("groups")
        if dataset is not None and None in (y_true, y_pred, groups):
            examples = get_examples(dataset) or []
            if examples and isinstance(examples[0], dict):
                y_true = [ex["y_true"] for ex in examples]
                y_pred = [ex["y_pred"] for ex in examples]
                groups = [ex["group"] for ex in examples]
        require_kwargs(
            {"y_true": y_true, "y_pred": y_pred, "groups": groups, "g1": kwargs.get("g1"), "g2": kwargs.get("g2")},
            "y_true",
            "y_pred",
            "groups",
            "g1",
            "g2",
        )
        result = gap_g_y(
            y_true,
            y_pred,
            groups,
            kwargs["g1"],
            kwargs["g2"],
            y=kwargs.get("y", 1),
        )
        return MetricResult(
            score=float(result.gap),
            details=result._asdict(),
        )


class ContextBasedDisparityScore(FairnessMetric):
    """BBQ-style S_DIS / S_AMB context-based disparity.

    ``dataset`` should be a sequence of output dicts with keys
    ``cond`` (``disambig``/``ambig``), ``output``, ``expected``.
    """

    name = "context_based_disparity"
    bias_type = "extrinsic"
    architectures = ("encoder_only",)

    def __init__(self, score: str = "s_dis"):
        if score not in ("s_dis", "s_amb"):
            raise ValueError("score must be 's_dis' or 's_amb'")
        self.score = score

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        outputs = kwargs.get("outputs")
        if outputs is None:
            outputs = get_examples(dataset)
        if not outputs:
            raise ValueError("outputs=... or dataset of BBQ-style result dicts is required")

        s_dis, n_dis, n_non_unk, n_biased = compute_s_dis(outputs)
        details = {
            "s_dis": s_dis,
            "n_disambig": n_dis,
            "n_non_unknown": n_non_unk,
            "n_biased": n_biased,
        }
        if self.score == "s_amb" or kwargs.get("also_s_amb", False):
            s_amb, acc_amb, n_amb = compute_s_amb(outputs, s_dis)
            details.update(
                {"s_amb": s_amb, "accuracy_ambig": acc_amb, "n_ambig": n_amb}
            )
            if self.score == "s_amb":
                return MetricResult(score=float(s_amb), details=details)
        return MetricResult(score=float(s_dis), details=details)
