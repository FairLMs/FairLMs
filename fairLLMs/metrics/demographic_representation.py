"""Demographic representation metrics: DNP, DRD."""

from __future__ import annotations

from typing import Any

from fairLLMs.definition.decoder_only.extrinsic_bias.demographic_representation.dnp.dnp import (
    compute_dnp,
)
from fairLLMs.definition.decoder_only.extrinsic_bias.demographic_representation.drd.drd import (
    compute_drd,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_examples, get_tokenizer_model, require_kwargs


class DemographicNextTokenProportion(FairnessMetric):
    """Demographic normalized next-token probabilities (DNP)."""

    name = "demographic_next_token_proportion"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        prompts = kwargs.get("prompts")
        if prompts is None and dataset is not None:
            examples = get_examples(dataset) or []
            prompts = [
                ex["prompt"] if isinstance(ex, dict) else ex for ex in examples
            ]
        require_kwargs(
            {
                "prompts": prompts,
                "stereo_words": kwargs.get("stereo_words"),
                "counter_words": kwargs.get("counter_words"),
                "neutral_words": kwargs.get("neutral_words"),
            },
            "prompts",
            "stereo_words",
            "counter_words",
            "neutral_words",
        )
        mean_ps, mean_psp, mean_pd, rows = compute_dnp(
            hf_model,
            tokenizer,
            prompts,
            kwargs["stereo_words"],
            kwargs["counter_words"],
            kwargs["neutral_words"],
        )
        return MetricResult(
            score=float(mean_pd),
            details={
                "mean_ps": mean_ps,
                "mean_psp": mean_psp,
                "mean_pd": mean_pd,
                "rows": rows,
            },
        )


class DemographicRepresentationDivergence(FairnessMetric):
    """Demographic representation disparity in generations (DRD)."""

    name = "demographic_representation_divergence"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        prompts = kwargs.get("prompts")
        if prompts is None and dataset is not None:
            examples = get_examples(dataset) or []
            prompts = [
                ex["prompt"] if isinstance(ex, dict) else ex for ex in examples
            ]
        require_kwargs(
            {
                "prompts": prompts,
                "stereo_words": kwargs.get("stereo_words"),
                "counter_words": kwargs.get("counter_words"),
            },
            "prompts",
            "stereo_words",
            "counter_words",
        )
        drd, n_s, n_sp, rows = compute_drd(
            hf_model,
            tokenizer,
            prompts,
            kwargs["stereo_words"],
            kwargs["counter_words"],
            max_new_tokens=kwargs.get("max_new_tokens", 50),
        )
        return MetricResult(
            score=float(drd),
            details={"n_s_total": n_s, "n_sp_total": n_sp, "rows": rows},
        )
