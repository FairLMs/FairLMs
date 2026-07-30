"""Decoder counterfactual fairness metrics: CR, CTF.

Both call an OpenAI *Completions* model, so they need ``OPENAI_API_KEY`` and the
``openai`` extra. ``data`` is a
:class:`~fairllms.metrics.data.PromptPairs`.
"""

from __future__ import annotations

from typing import Any

from fairllms.definition.decoder_only.extrinsic_bias.counterfactual_fairness.cr.cr import (
    compute_cr,
)
from fairllms.definition.decoder_only.extrinsic_bias.counterfactual_fairness.ctf.ctf import (
    compute_ctf,
)
from fairllms.metrics._compat import as_examples, take, unwrap, warn_legacy
from fairllms.metrics.base import FairnessMetric, MetricResult
from fairllms.metrics.data import PromptPairs
from fairllms.metrics.resolve import get_openai_bundle


class _PromptPairMetric(FairnessMetric):
    """Shared plumbing for CR and CTF. Abstract: subclasses add compute."""

    bias_type = "extrinsic"
    architectures = ("decoder_only",)

    def __init__(self, *, completion_model: str = "gpt-3.5-turbo-instruct"):
        self.completion_model = completion_model

    def _prepare(self, data: Any, legacy: dict) -> PromptPairs:
        data = unwrap(
            data if data is not None else legacy.pop("dataset", None), PromptPairs
        )

        if not isinstance(data, PromptPairs):
            factual, k1 = take(legacy, "factual_prompts")
            counterfactual, k2 = take(legacy, "counterfactual_prompts")
            if factual is not None and counterfactual is not None:
                warn_legacy(type(self).__name__, [k1, k2], "PromptPairs")
                legacy.pop(k1, None)
                legacy.pop(k2, None)
                data = PromptPairs(factual, counterfactual)
            elif data is not None:
                data = PromptPairs.from_examples(
                    as_examples(
                        data,
                        type(self).__name__,
                        "a PromptPairs or sequence of (factual, counterfactual) pairs",
                    )
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} requires factual/counterfactual prompt "
                    f"pairs. Pass a PromptPairs as the second argument, e.g. "
                    f"compute(model, PromptPairs(factual, counterfactual))."
                )

        self._reject_unknown_kwargs(legacy, "completion_model")
        return data


class CounterfactualRobustness(_PromptPairMetric):
    """Rate at which the top-1 completion changes under a counterfactual flip (CR).

    Higher means the model's prediction is more sensitive to the swapped
    attribute, i.e. less counterfactually robust.
    """

    name = "counterfactual_robustness"

    def compute(
        self, model: Any = None, data: Any = None, **legacy: Any
    ) -> MetricResult:
        pairs = self._prepare(data, legacy)
        bundle = get_openai_bundle(model)
        cr, rows = compute_cr(
            bundle.client,
            list(pairs.factual),
            list(pairs.counterfactual),
            model=legacy.get("completion_model", self.completion_model),
        )
        return MetricResult(
            score=float(cr), details={"rows": rows, "n_pairs": len(pairs)}
        )


class CounterfactualFairnessScore(_PromptPairMetric):
    """Mean total-variation distance between next-token distributions (CTF).

    Note the API exposes only the top-5 logprobs, so the distance carries a
    residual-mass correction for the unobserved tail. A local causal LM would
    give the full distribution.
    """

    name = "counterfactual_fairness"

    def compute(
        self, model: Any = None, data: Any = None, **legacy: Any
    ) -> MetricResult:
        pairs = self._prepare(data, legacy)
        bundle = get_openai_bundle(model)
        ctf, rows = compute_ctf(
            bundle.client,
            list(pairs.factual),
            list(pairs.counterfactual),
            model=legacy.get("completion_model", self.completion_model),
        )
        return MetricResult(
            score=float(ctf), details={"rows": rows, "n_pairs": len(pairs)}
        )
