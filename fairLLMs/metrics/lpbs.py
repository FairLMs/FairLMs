"""Log Probability Bias Score (LPBS) wrapper."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

from fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.lpbs.lpbs import (  # noqa: E501
    DEFAULT_TEMPLATE,
    compute_lpbs,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_tokenizer_model


class LogProbabilityBiasScore(FairnessMetric):
    """Kurita et al. log-probability bias score for masked LMs.

    Pass target attributes via ``attribute_words`` (or ``dataset`` as a list of
    attribute strings). Demographic pair defaults to ``("he", "she")``.
    """

    name = "log_probability_bias_score"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)

    def __init__(
        self,
        gender_words: Tuple[str, str] = ("he", "she"),
        template: str = DEFAULT_TEMPLATE,
        gender_comes_first: bool = True,
    ):
        self.gender_words = gender_words
        self.template = template
        self.gender_comes_first = gender_comes_first

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        attribute_words = kwargs.get("attribute_words")
        if attribute_words is None and dataset is not None:
            if hasattr(dataset, "load"):
                examples = list(dataset.load())
            else:
                examples = list(dataset)
            if examples and isinstance(examples[0], dict):
                attribute_words = [
                    ex.get("profession_name") or ex.get("attribute") or ex.get("word")
                    for ex in examples
                ]
                attribute_words = [a for a in attribute_words if a]
            else:
                attribute_words = examples
        if not attribute_words:
            raise ValueError(
                "Provide attribute_words=... or a dataset of attribute strings"
            )
        gender_words = kwargs.get("gender_words", self.gender_words)
        template = kwargs.get("template", self.template)
        gender_comes_first = kwargs.get("gender_comes_first", self.gender_comes_first)

        outcomes, mean_lpbs, std_lpbs, prop = compute_lpbs(
            tokenizer,
            hf_model,
            gender_words,
            attribute_words,
            template=template,
            gender_comes_first=gender_comes_first,
        )
        return MetricResult(
            score=float(mean_lpbs),
            details={
                "std": std_lpbs,
                "proportion_favoring_group1": prop,
                "outcomes": outcomes,
                "n_attributes": len(attribute_words),
            },
        )
