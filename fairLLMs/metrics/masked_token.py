"""Remaining encoder-only intrinsic metric wrappers (CEAT, DisCo, CBS)."""

from __future__ import annotations

from typing import Any, Optional

from fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.cbs.cbs import (  # noqa: E501
    compute_cbs,
)
from fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.disco.disco import (  # noqa: E501
    compute_disco,
)
from fairLLMs.definition.encoder_only.intrinsic_bias.similarity_based.ceat.ceat import (
    compute_ceat,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_tokenizer_model, require_kwargs


class CEAT(FairnessMetric):
    """Contextualized Embedding Association Test."""

    name = "ceat"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)

    def __init__(
        self,
        pooling: str = "cls",
        sample_size: int = 10,
        n_trials: int = 100,
        seed: Optional[int] = None,
    ):
        self.pooling = pooling
        self.sample_size = sample_size
        self.n_trials = n_trials
        self.seed = seed

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, device = get_tokenizer_model(model, kwargs.get("tokenizer"))
        require_kwargs(kwargs, "T1_contexts", "T2_contexts", "A1_contexts", "A2_contexts")
        result = compute_ceat(
            hf_model,
            tokenizer,
            kwargs["T1_contexts"],
            kwargs["T2_contexts"],
            kwargs["A1_contexts"],
            kwargs["A2_contexts"],
            pooling=kwargs.get("pooling", self.pooling),
            sample_size=kwargs.get("sample_size", self.sample_size),
            n_trials=kwargs.get("n_trials", self.n_trials),
            seed=kwargs.get("seed", self.seed),
            device=device,
        )
        return MetricResult(
            score=float(result["CES"]),
            details=dict(result),
        )


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
        # ``model`` may be a fill-mask pipeline, or we build one from an MLM.
        pipe = kwargs.get("pipe", model)
        if pipe is not None and not callable(getattr(pipe, "__call__", None)):
            # LoadedModel / HuggingFaceModel → build pipeline
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
        # Primary score: confirmatory stereo CBS when present
        stereo = info.get("stereo") if isinstance(info, dict) else None
        if isinstance(stereo, dict) and "cbs" in stereo:
            score = float(stereo["cbs"])
        else:
            # fall back to mean exploratory cbs
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
