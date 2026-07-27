"""Algorithmic disparity metrics: LFP, MCD."""

from __future__ import annotations

from typing import Any, List

from fairLLMs.definition.encoder_decoder.intrinsic_bias.algorithmic_disparity.lfp.lfp import (
    compute_lfp,
)
from fairLLMs.definition.encoder_decoder.intrinsic_bias.algorithmic_disparity.mcd.mcd import (
    compute_mcd,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_examples, get_tokenizer_model


def _sentences_from_dataset(dataset, key: str = "sentence") -> List[str]:
    examples = get_examples(dataset) or []
    if not examples:
        return []
    if isinstance(examples[0], str):
        return list(examples)
    return [ex.get(key) or ex.get("text") or ex.get("premise") for ex in examples]


class LexicalFrequencyProportion(FairnessMetric):
    """Lexical frequency profile of translations (LFP)."""

    name = "lexical_frequency_proportion"
    bias_type = "intrinsic"
    architectures = ("encoder_decoder",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        sentences = kwargs.get("sentences")
        if sentences is None:
            sentences = _sentences_from_dataset(dataset)
        if not sentences:
            raise ValueError("sentences=... is required")
        pb1, pb2, pb3, rows = compute_lfp(
            hf_model,
            tokenizer,
            sentences,
            max_new_tokens=kwargs.get("max_new_tokens", 128),
        )
        return MetricResult(
            score=float(pb1),
            details={"pb1": pb1, "pb2": pb2, "pb3": pb3, "rows": rows},
        )


class MorphologicalChoiceDivergence(FairnessMetric):
    """Morphological complexity disparity (MCD)."""

    name = "morphological_choice_divergence"
    bias_type = "intrinsic"
    architectures = ("encoder_decoder",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        sentences = kwargs.get("sentences")
        if sentences is None:
            sentences = _sentences_from_dataset(dataset)
        if not sentences:
            raise ValueError("sentences=... is required")
        mean_h, mean_d, rows = compute_mcd(
            hf_model,
            tokenizer,
            sentences,
            max_new_tokens=kwargs.get("max_new_tokens", 128),
        )
        return MetricResult(
            score=float(mean_h),
            details={"mean_h": mean_h, "mean_d": mean_d, "rows": rows},
        )
