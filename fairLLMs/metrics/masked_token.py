"""Masked-token metrics: DisCo, LPBS, CBS."""

from __future__ import annotations

from typing import Any, Tuple

from fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.cbs.cbs import (  # noqa: E501
    compute_cbs,
)
from fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.disco.disco import (  # noqa: E501
    compute_disco,
)
from fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.lpbs.lpbs import (  # noqa: E501
    DEFAULT_TEMPLATE,
    compute_lpbs,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_tokenizer_model, require_kwargs


class DiscoveryOfCorrelationsScore(FairnessMetric):
    """DisCo: top-k prediction divergence across demographic pairs."""

    name = "discovery_of_correlations"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)

    def __init__(self, k: int = 3, n_bootstrap: int = 1000, seed: int = 42, templates=None):
        self.k = k
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.templates = templates

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        pipe = kwargs.get("pipe", model)
        if pipe is not None and not callable(getattr(pipe, "__call__", None)):
            from transformers import pipeline

            tokenizer, hf_model, device = get_tokenizer_model(model, kwargs.get("tokenizer"))
            pipe = pipeline(
                "fill-mask",
                model=hf_model,
                tokenizer=tokenizer,
                device=0 if str(device).startswith("cuda") else -1,
            )

        if pipe is None:
            raise TypeError("Pass a fill-mask pipeline or HuggingFace MLM as model")

        g1 = kwargs.get("group1_words")
        g2 = kwargs.get("group2_words")
        if g1 is None or g2 is None:
            raise ValueError("DisCo requires group1_words= and group2_words=")

        disco, ci_low, ci_high = compute_disco(
            pipe,
            g1,
            g2,
            templates=kwargs.get("templates", self.templates),
            k=kwargs.get("k", self.k),
            n_bootstrap=kwargs.get("n_bootstrap", self.n_bootstrap),
            seed=kwargs.get("seed", self.seed),
        )
        return MetricResult(
            score=float(disco),
            details={"ci_low": ci_low, "ci_high": ci_high},
        )


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


class ContrastBasedScore(FairnessMetric):
    """Contrast-based masked preference score (CBS)."""

    name = "contrast_based_score"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)

    def __init__(self, n_bootstrap: int = 1000, n_perm: int = 1000, seed: int = 42):
        self.n_bootstrap = n_bootstrap
        self.n_perm = n_perm
        self.seed = seed

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        require_kwargs(kwargs, "group_terms", "contrast_pairs", "templates")
        per_group, info = compute_cbs(
            tokenizer,
            hf_model,
            kwargs["group_terms"],
            kwargs["contrast_pairs"],
            kwargs["templates"],
            n_bootstrap=kwargs.get("n_bootstrap", self.n_bootstrap),
            n_perm=kwargs.get("n_perm", self.n_perm),
            seed=kwargs.get("seed", self.seed),
            group_placeholder=kwargs.get("group_placeholder", "{N}"),
            attr_placeholder=kwargs.get("attr_placeholder", "{A}"),
        )
        stereo = info.get("stereo") if isinstance(info, dict) else None
        if isinstance(stereo, dict) and "cbs" in stereo:
            score = float(stereo["cbs"])
        else:
            scores = [
                v.get("cbs")
                for v in (per_group or {}).values()
                if isinstance(v, dict) and v.get("cbs") is not None
            ]
            score = float(sum(scores) / len(scores)) if scores else float("nan")
        return MetricResult(
            score=score,
            details={"per_group": per_group, "info": info},
        )
