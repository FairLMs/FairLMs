"""Decoder counterfactual fairness metrics: CR, CTF."""

from __future__ import annotations

from typing import Any

from fairLLMs.definition.decoder_only.extrinsic_bias.counterfactual_fairness.cr.cr import (
    compute_cr,
)
from fairLLMs.definition.decoder_only.extrinsic_bias.counterfactual_fairness.ctf.ctf import (
    compute_ctf,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_examples, get_openai_bundle, require_kwargs


def _prompt_pairs(dataset, kwargs):
    factual = kwargs.get("factual_prompts")
    counterfactual = kwargs.get("counterfactual_prompts")
    if dataset is not None and (factual is None or counterfactual is None):
        examples = get_examples(dataset) or []
        factual, counterfactual = [], []
        for ex in examples:
            if isinstance(ex, dict):
                factual.append(ex.get("factual") or ex.get("stereotype"))
                counterfactual.append(
                    ex.get("counterfactual") or ex.get("anti_stereotype")
                )
            else:
                factual.append(ex[0])
                counterfactual.append(ex[1])
    return factual, counterfactual


class CounterfactualRobustness(FairnessMetric):
    """Change rate of top-1 token under counterfactual flip (CR)."""

    name = "counterfactual_robustness"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)

    def __init__(self, completion_model: str = "gpt-3.5-turbo-instruct"):
        self.completion_model = completion_model

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        bundle = get_openai_bundle(model)
        factual, counterfactual = _prompt_pairs(dataset, kwargs)
        require_kwargs(
            {"factual_prompts": factual, "counterfactual_prompts": counterfactual},
            "factual_prompts",
            "counterfactual_prompts",
        )
        cr, rows = compute_cr(
            bundle.client,
            factual,
            counterfactual,
            model=kwargs.get("completion_model", self.completion_model),
        )
        return MetricResult(score=float(cr), details={"rows": rows})


class CounterfactualFairnessScore(FairnessMetric):
    """Mean TVD of next-token distributions (CTF)."""

    name = "counterfactual_fairness"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)

    def __init__(self, completion_model: str = "gpt-3.5-turbo-instruct"):
        self.completion_model = completion_model

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        bundle = get_openai_bundle(model)
        factual, counterfactual = _prompt_pairs(dataset, kwargs)
        require_kwargs(
            {"factual_prompts": factual, "counterfactual_prompts": counterfactual},
            "factual_prompts",
            "counterfactual_prompts",
        )
        ctf, rows = compute_ctf(
            bundle.client,
            factual,
            counterfactual,
            model=kwargs.get("completion_model", self.completion_model),
        )
        return MetricResult(score=float(ctf), details={"rows": rows})
