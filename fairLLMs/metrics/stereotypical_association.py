"""Decoder stereotypical-association metrics: SLL, CA."""

from __future__ import annotations

from typing import Any

import numpy as np

from fairLLMs.definition.decoder_only.intrinsic_bias.stereotypical_association.ca.ca import (
    compute_ca,
)
from fairLLMs.definition.decoder_only.intrinsic_bias.stereotypical_association.sll.sll import (
    compute_sll,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_examples, get_tokenizer_model, require_kwargs


class StereotypicalLogLikelihood(FairnessMetric):
    """Stereotypical log-likelihood gaps (SLL)."""

    name = "stereotypical_log_likelihood"
    bias_type = "intrinsic"
    architectures = ("decoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, device = get_tokenizer_model(model, kwargs.get("tokenizer"))
        occupation_pairs = kwargs.get("occupation_pairs")
        if occupation_pairs is None and dataset is not None:
            occupation_pairs = get_examples(dataset)
        if not occupation_pairs:
            raise ValueError("occupation_pairs=... is required")
        scores = compute_sll(hf_model, tokenizer, device, occupation_pairs)
        primary = float(np.mean([abs(v) for v in scores.values()]))
        return MetricResult(score=primary, details=dict(scores))


class CooccurrenceAssociation(FairnessMetric):
    """Concept association via generation + TV distance (CA)."""

    name = "cooccurrence_association"
    bias_type = "intrinsic"
    architectures = ("decoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        require_kwargs(kwargs, "concepts", "prompt_template", "group_terms")
        mean_tvd, n_valid, n_skipped = compute_ca(
            hf_model,
            tokenizer,
            kwargs["concepts"],
            kwargs["prompt_template"],
            kwargs["group_terms"],
            n_samples=kwargs.get("n_samples", 20),
        )
        return MetricResult(
            score=float(mean_tvd),
            details={"n_valid": n_valid, "n_skipped": n_skipped},
        )
