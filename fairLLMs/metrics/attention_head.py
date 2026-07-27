"""Attention-head disparity metrics: GBE, NIE."""

from __future__ import annotations

from typing import Any

import numpy as np

from fairLLMs.definition.decoder_only.intrinsic_bias.attention_head_based_disparity.gbe.gbe import (
    compute_gbe,
    compute_gbe_matrix,
)
from fairLLMs.definition.decoder_only.intrinsic_bias.attention_head_based_disparity.nie.nie import (
    compute_nie,
    compute_nie_matrix,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_tokenizer_model, require_kwargs


class GradientBasedBiasEstimation(FairnessMetric):
    """Gradient-based bias on GPT-2 Value heads (GBE)."""

    name = "gradient_based_bias_estimation"
    bias_type = "intrinsic"
    architectures = ("decoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        if "gbe_matrix" in kwargs and kwargs["gbe_matrix"] is not None:
            matrix = np.asarray(kwargs["gbe_matrix"])
        else:
            tokenizer, hf_model, device = get_tokenizer_model(
                model, kwargs.get("tokenizer")
            )
            require_kwargs(kwargs, "X", "Y", "A", "B")
            matrix = compute_gbe_matrix(
                hf_model,
                tokenizer,
                device,
                kwargs["X"],
                kwargs["Y"],
                kwargs["A"],
                kwargs["B"],
                loss_scale=kwargs.get("loss_scale", 1.0),
                verbose=kwargs.get("verbose", False),
            )
        score = float(compute_gbe(matrix))
        return MetricResult(score=score, details={"gbe_matrix": matrix})


class NaturalIndirectEffect(FairnessMetric):
    """Natural Indirect Effect / attention mediation (NIE)."""

    name = "natural_indirect_effect"
    bias_type = "intrinsic"
    architectures = ("decoder_only",)

    def __init__(self, threshold: float = 0.003):
        self.threshold = threshold

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        if "nie" in kwargs and kwargs["nie"] is not None:
            nie = np.asarray(kwargs["nie"])
        else:
            tokenizer, hf_model, device = get_tokenizer_model(
                model, kwargs.get("tokenizer")
            )
            require_kwargs(kwargs, "probes", "N_LAYERS", "N_HEADS", "HEAD_DIM")
            nie = compute_nie_matrix(
                hf_model,
                tokenizer,
                device,
                kwargs["probes"],
                kwargs["N_LAYERS"],
                kwargs["N_HEADS"],
                kwargs["HEAD_DIM"],
            )
        score = float(compute_nie(nie, threshold=kwargs.get("threshold", self.threshold)))
        return MetricResult(score=score, details={"nie": nie})
