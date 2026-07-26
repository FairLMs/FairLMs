"""Encoder-decoder metric wrappers."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from fairLLMs.definition.encoder_decoder.extrinsic_bias.counterfactual_fairness.auc import (
    compute_auc,
)
from fairLLMs.definition.encoder_decoder.extrinsic_bias.fair_inference.ibs import (
    compute_ibs,
)
from fairLLMs.definition.encoder_decoder.extrinsic_bias.individual_fairness.ss import (
    compute_ss as compute_translation_ss,
)
from fairLLMs.definition.encoder_decoder.extrinsic_bias.position_based.npd import (
    compute_npd,
)
from fairLLMs.definition.encoder_decoder.intrinsic_bias.algorithmic_disparity.lfp.lfp import (
    compute_lfp,
)
from fairLLMs.definition.encoder_decoder.intrinsic_bias.algorithmic_disparity.mcd.mcd import (
    compute_mcd,
)
from fairLLMs.definition.encoder_decoder.intrinsic_bias.stereotypical_association.sd.sd import (
    compute_sd,
)
from fairLLMs.definition.encoder_decoder.intrinsic_bias.stereotypical_association.sva.sva import (
    compute_sva,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_examples, get_tokenizer_model, require_kwargs


def _sentences_from_dataset(dataset, key: str = "sentence") -> List[str]:
    examples = get_examples(dataset) or []
    if not examples:
        return []
    if isinstance(examples[0], str):
        return list(examples)
    return [ex.get(key) or ex.get("text") or ex.get("premise") for ex in examples]


class CounterfactualAucScore(FairnessMetric):
    """Counterfactual fairness via attribute recoverability (AUC)."""

    name = "counterfactual_auc"
    bias_type = "extrinsic"
    architectures = ("encoder_decoder",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        sentences = kwargs.get("sentences")
        labels = kwargs.get("labels")
        if dataset is not None and (sentences is None or labels is None):
            examples = get_examples(dataset) or []
            sentences = [ex["sentence"] if isinstance(ex, dict) else ex[0] for ex in examples]
            labels = [ex["label"] if isinstance(ex, dict) else ex[1] for ex in examples]
        require_kwargs({"sentences": sentences, "labels": labels}, "sentences", "labels")
        auc_mean, auc_std, n0, n1, rows = compute_auc(
            hf_model,
            tokenizer,
            sentences,
            labels,
            pair_ids=kwargs.get("pair_ids"),
            test_ratio=kwargs.get("test_ratio", 0.2),
            seed=kwargs.get("seed", 42),
            n_seeds=kwargs.get("n_seeds", 10),
        )
        return MetricResult(
            score=float(auc_mean),
            details={"auc_std": auc_std, "n0": n0, "n1": n1, "rows": rows},
        )


class InferenceBiasScore(FairnessMetric):
    """Idealized Bias Score (IBS) over labeled prediction pairs."""

    name = "inference_bias_score"
    bias_type = "extrinsic"
    architectures = ("encoder_decoder",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        predictions = kwargs.get("predictions")
        if predictions is None:
            predictions = get_examples(dataset)
        if not predictions:
            raise ValueError("predictions=... sequence of (label, prediction) is required")
        ibs, counts = compute_ibs(predictions)
        return MetricResult(score=float(ibs), details={"counts": counts})


class TranslationSimilarityScore(FairnessMetric):
    """Semantic similarity of counterfactual translations (LaBSE)."""

    name = "translation_similarity_score"
    bias_type = "extrinsic"
    architectures = ("encoder_decoder",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        require_kwargs(kwargs, "labse_model", "labse_tokenizer")
        pairs = kwargs.get("pairs")
        if pairs is None:
            pairs = get_examples(dataset)
        if not pairs:
            raise ValueError("pairs=... of (original, counterfactual) is required")
        mean_ss, std_ss, rows = compute_translation_ss(
            hf_model,
            tokenizer,
            kwargs["labse_model"],
            kwargs["labse_tokenizer"],
            pairs,
            tgt_lang=kwargs.get("tgt_lang", "French"),
            max_new_tokens=kwargs.get("max_new_tokens", 128),
        )
        return MetricResult(
            score=float(mean_ss),
            details={"std": std_ss, "rows": rows},
        )


class NormalizedPositionDistance(FairnessMetric):
    """Normalized position disparity for summarization."""

    name = "normalized_position_distance"
    bias_type = "extrinsic"
    architectures = ("encoder_decoder",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        articles = kwargs.get("articles")
        if articles is None:
            articles = _sentences_from_dataset(dataset, key="article")
        if not articles:
            raise ValueError("articles=... or a dataset of articles is required")
        mean_npd, rows = compute_npd(
            hf_model,
            tokenizer,
            articles,
            gold_summaries=kwargs.get("gold_summaries"),
            max_new_tokens=kwargs.get("max_new_tokens", 128),
            K=kwargs.get("K", 10),
        )
        return MetricResult(score=float(mean_npd), details={"rows": rows})


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


class StereotypicalDivergence(FairnessMetric):
    """Stereotype vs anti-stereotype performance divergence (SD)."""

    name = "stereotypical_divergence"
    bias_type = "intrinsic"
    architectures = ("encoder_decoder",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        require_kwargs(
            kwargs,
            "stereo_sentences",
            "stereo_labels",
            "anti_sentences",
            "anti_labels",
        )
        sd_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 128),
        }
        if kwargs.get("metric_fn") is not None:
            sd_kwargs["metric_fn"] = kwargs["metric_fn"]
        m_stereo, m_anti, delta_s, rows = compute_sd(
            hf_model,
            tokenizer,
            kwargs["stereo_sentences"],
            kwargs["stereo_labels"],
            kwargs["anti_sentences"],
            kwargs["anti_labels"],
            **sd_kwargs,
        )
        return MetricResult(
            score=float(delta_s),
            details={"m_stereo": m_stereo, "m_anti": m_anti, "rows": rows},
        )


class StereotypicalValueAttribution(FairnessMetric):
    """Shapley attribution of stereotype bias to attention heads (SVA)."""

    name = "stereotypical_value_attribution"
    bias_type = "intrinsic"
    architectures = ("encoder_decoder",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        require_kwargs(
            kwargs,
            "stereo_sents",
            "anti_sents",
            "direction",
            "n_layers",
            "n_heads",
        )
        sva, phi = compute_sva(
            hf_model,
            tokenizer,
            kwargs["stereo_sents"],
            kwargs["anti_sents"],
            np.asarray(kwargs["direction"]),
            kwargs["n_layers"],
            kwargs["n_heads"],
            n_samples=kwargs.get("n_samples", 15),
            top_pct=kwargs.get("top_pct", 0.10),
        )
        return MetricResult(
            score=float(sva),
            details={"phi": phi},
        )
