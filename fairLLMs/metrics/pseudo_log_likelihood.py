"""Pseudo-log-likelihood metrics: PLL, CPS, AUL, AULA, CAT."""

from __future__ import annotations

from typing import Any

from fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.aul.aul import (  # noqa: E501
    compute_aul,
)
from fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.aula.aula import (  # noqa: E501
    compute_aula,
)
from fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cat.cat import (  # noqa: E501
    compute_ss,
)
from fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cps.cps import (  # noqa: E501
    compute_cps,
)
from fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.pll.pll import (  # noqa: E501
    compute_pll,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_examples, get_tokenizer_model


def _run_pair_metric(compute_fn, model, dataset, tokenizer=None, **compute_kwargs):
    tok, hf_model, _ = get_tokenizer_model(model, tokenizer)
    pairs = get_examples(dataset)
    if not pairs:
        raise ValueError("dataset with stereotype/anti_stereotype pairs is required")
    score, accuracy, per_bias_type = compute_fn(tok, hf_model, pairs, **compute_kwargs)
    return MetricResult(
        score=float(score),
        details={"accuracy": accuracy, "n_pairs": len(pairs)},
        by_category=dict(per_bias_type),
    )


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
        return _run_pair_metric(
            compute_cps, model, dataset, tokenizer=kwargs.get("tokenizer")
        )


class PseudoLogLikelihoodScore(FairnessMetric):
    """Full-sentence pseudo log-likelihood preference score."""

    name = "pseudo_log_likelihood_score"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        return _run_pair_metric(
            compute_pll, model, dataset, tokenizer=kwargs.get("tokenizer")
        )


class AllUnmaskedLikelihoodScore(FairnessMetric):
    """All Unmasked Likelihood (AUL) bias score."""

    name = "all_unmasked_likelihood_score"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)

    def __init__(self, use_attention: bool = False):
        self.use_attention = use_attention

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        return _run_pair_metric(
            compute_aul,
            model,
            dataset,
            tokenizer=kwargs.get("tokenizer"),
            use_attention=kwargs.get("use_attention", self.use_attention),
        )


class AllUnmaskedLikelihoodAttentionScore(FairnessMetric):
    """Attention-weighted AUL (AULA)."""

    name = "all_unmasked_likelihood_attention_score"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        return _run_pair_metric(
            compute_aula,
            model,
            dataset,
            tokenizer=kwargs.get("tokenizer"),
            use_attention=kwargs.get("use_attention", True),
        )


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
