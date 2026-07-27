"""Performance disparity metrics: AD, BA, SNS."""

from __future__ import annotations

from typing import Any, Callable

from fairLLMs.definition.decoder_only.extrinsic_bias.performance_disparity.ad.ad import (
    compute_ad,
)
from fairLLMs.definition.decoder_only.extrinsic_bias.performance_disparity.ba.ba import (
    compute_ba,
)
from fairLLMs.definition.decoder_only.extrinsic_bias.performance_disparity.sns.sns import (
    compute_sns,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_examples, get_openai_bundle, require_kwargs


class AccuracyDisparity(FairnessMetric):
    """Accuracy disparity between stereotype and counter-stereotype scores."""

    name = "accuracy_disparity"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        scores_s = kwargs.get("scores_s")
        scores_sp = kwargs.get("scores_sp")
        if dataset is not None and (scores_s is None or scores_sp is None):
            examples = get_examples(dataset) or []
            scores_s = [ex["scores_s"] if isinstance(ex, dict) else ex[0] for ex in examples]
            scores_sp = [ex["scores_sp"] if isinstance(ex, dict) else ex[1] for ex in examples]
        require_kwargs(
            {"scores_s": scores_s, "scores_sp": scores_sp}, "scores_s", "scores_sp"
        )
        acc_s, acc_sp, ad = compute_ad(scores_s, scores_sp)
        return MetricResult(
            score=float(ad),
            details={"acc_s": acc_s, "acc_sp": acc_sp},
        )


class BiasAmplifierScore(FairnessMetric):
    """BiasAsker absolute & relative bias (BA)."""

    name = "bias_amplifier"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        bundle = get_openai_bundle(model)
        require_kwargs(kwargs, "groups", "properties", "ab_template", "rb_template")
        ab, rb, ab_rows, rb_rows = compute_ba(
            bundle.client,
            kwargs["groups"],
            kwargs["properties"],
            kwargs["ab_template"],
            kwargs["rb_template"],
            max_new_tokens=kwargs.get("max_new_tokens", 20),
        )
        return MetricResult(
            score=float(ab),
            details={"ab": ab, "rb": rb, "ab_rows": ab_rows, "rb_rows": rb_rows},
        )


class SensitiveNameSimilarity(FairnessMetric):
    """Sensitive-to-neutral similarity (SNS)."""

    name = "sensitive_name_similarity"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        call_model: Callable = kwargs.get("call_model") or model
        if call_model is None:
            raise TypeError("Pass call_model=... callable or a compatible model")
        require_kwargs(
            kwargs,
            "queries",
            "neutral_prompt_fn",
            "group_prompt_fn",
            "group_values",
        )
        snsr, snsv, df = compute_sns(
            call_model,
            kwargs["queries"],
            kwargs["neutral_prompt_fn"],
            kwargs["group_prompt_fn"],
            kwargs["group_values"],
            k=kwargs.get("k", 5),
        )
        return MetricResult(
            score=float(snsr),
            details={"snsr": snsr, "snsv": snsv, "table": df},
        )
