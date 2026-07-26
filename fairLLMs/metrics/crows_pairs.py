"""CrowS-Pairs Score (CPS) wrapper."""

from __future__ import annotations

from typing import Any

from fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cps.cps import (  # noqa: E501
    compute_cps,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_examples, get_tokenizer_model


class CrowSPairsScore(FairnessMetric):
    """Pseudo-log-likelihood CrowS-Pairs Score (Nangia et al.).

    ``dataset`` should yield dicts with keys
    ``stereotype``, ``anti_stereotype``, and optionally ``bias_type``
    (see :class:`~fairLLMs.datasets.CrowSPairs`).
    """

    name = "crows_pairs_score"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        pairs = get_examples(dataset)
        if not pairs:
            raise ValueError("dataset with stereotype/anti_stereotype pairs is required")
        score, accuracy, per_bias_type = compute_cps(tokenizer, hf_model, pairs)
        return MetricResult(
            score=float(score),
            details={"accuracy": accuracy, "n_pairs": len(pairs)},
            by_category=dict(per_bias_type),
        )
