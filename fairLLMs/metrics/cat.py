"""Context Association Test (CAT / iCAT) wrapper."""

from __future__ import annotations

from typing import Any

from fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cat.cat import (  # noqa: E501
    compute_ss,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_examples, get_tokenizer_model


class ContextAssociationTestScore(FairnessMetric):
    """StereoSet-style SS / LMS / iCAT on sentence triples."""

    name = "context_association_test"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))

        stereo = kwargs.get("stereo_sentences")
        anti = kwargs.get("anti_sentences")
        related = kwargs.get("related_sentences")

        if dataset is not None and None in (stereo, anti, related):
            examples = get_examples(dataset) or []
            stereo, anti, related = [], [], []
            for ex in examples:
                if isinstance(ex, dict):
                    stereo.append(ex["stereotype"])
                    anti.append(ex["anti_stereotype"])
                    related.append(ex["unrelated"])
                else:
                    stereo.append(ex[0])
                    anti.append(ex[1])
                    related.append(ex[2])

        if None in (stereo, anti, related):
            raise ValueError(
                "Provide stereo/anti/related sentence lists or a triples dataset"
            )

        ss, lms, icat, rows = compute_ss(
            hf_model, tokenizer, stereo, anti, related
        )
        return MetricResult(
            score=float(icat),
            details={"ss": ss, "lms": lms, "rows": rows, "n": len(stereo)},
        )
