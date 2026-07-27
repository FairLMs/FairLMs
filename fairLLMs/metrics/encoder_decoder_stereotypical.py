"""Encoder-decoder stereotypical-association metrics: SD, SVA."""

from __future__ import annotations

from typing import Any

import numpy as np

from fairLLMs.definition.encoder_decoder.intrinsic_bias.stereotypical_association.sd.sd import (
    compute_sd,
)
from fairLLMs.definition.encoder_decoder.intrinsic_bias.stereotypical_association.sva.sva import (
    compute_sva,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_tokenizer_model, require_kwargs


class StereotypicalDivergence(FairnessMetric):
    """Stereotype vs anti-stereotype performance divergence (SD)."""

    name = "stereotypical_divergence"
    bias_type = "intrinsic"
    architectures = ("encoder_decoder",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        require_kwargs(
            kwargs,
            "stereo_sentences",
            "stereo_labels",
            "anti_sentences",
            "anti_labels",
        )
        sd_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 128),
        }
        if kwargs.get("metric_fn") is not None:
            sd_kwargs["metric_fn"] = kwargs["metric_fn"]
        m_stereo, m_anti, delta_s, rows = compute_sd(
            hf_model,
            tokenizer,
            kwargs["stereo_sentences"],
            kwargs["stereo_labels"],
            kwargs["anti_sentences"],
            kwargs["anti_labels"],
            **sd_kwargs,
        )
        return MetricResult(
            score=float(delta_s),
            details={"m_stereo": m_stereo, "m_anti": m_anti, "rows": rows},
        )


class StereotypicalValueAttribution(FairnessMetric):
    """Shapley attribution of stereotype bias to attention heads (SVA)."""

    name = "stereotypical_value_attribution"
    bias_type = "intrinsic"
    architectures = ("encoder_decoder",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        require_kwargs(
            kwargs,
            "stereo_sents",
            "anti_sents",
            "direction",
            "n_layers",
            "n_heads",
        )
        sva, phi = compute_sva(
            hf_model,
            tokenizer,
            kwargs["stereo_sents"],
            kwargs["anti_sents"],
            np.asarray(kwargs["direction"]),
            kwargs["n_layers"],
            kwargs["n_heads"],
            n_samples=kwargs.get("n_samples", 15),
            top_pct=kwargs.get("top_pct", 0.10),
        )
        return MetricResult(
            score=float(sva),
            details={"phi": phi},
        )
