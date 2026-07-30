"""Encoder-decoder extrinsic metrics: counterfactual AUC, IBS, NPD, translation SS."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from fairllms.definition.encoder_decoder.extrinsic_bias.counterfactual_fairness.auc import (
    compute_auc,
)
from fairllms.definition.encoder_decoder.extrinsic_bias.fair_inference.ibs import (
    compute_ibs,
)
from fairllms.definition.encoder_decoder.extrinsic_bias.individual_fairness.ss import (
    compute_ss as compute_translation_ss,
)
from fairllms.definition.encoder_decoder.extrinsic_bias.position_based.npd import (
    compute_npd,
)
from fairllms.metrics._compat import as_examples, take, unwrap, warn_legacy
from fairllms.metrics.base import FairnessMetric, MetricResult
from fairllms.metrics.data import LabeledSentences
from fairllms.metrics.resolve import get_tokenizer_model


def _texts(data: Any, metric: str, what: str, key: str) -> List[str]:
    """Coerce a dataset/sequence to a list of strings."""
    examples = as_examples(data, metric, what)
    if isinstance(examples[0], str):
        return list(examples)
    out = []
    for ex in examples:
        if isinstance(ex, dict):
            value = ex.get(key) or ex.get("text") or ex.get("sentence") or ex.get("premise")
            if value is None:
                raise ValueError(
                    f"{metric}: example dict has no {key!r}/'text'/'sentence' key; "
                    f"got {sorted(ex)}."
                )
            out.append(value)
        else:
            raise TypeError(
                f"{metric}: expected strings or dicts, got {type(ex).__name__}."
            )
    return out


class CounterfactualAucScore(FairnessMetric):
    """Counterfactual fairness via protected-attribute recoverability (AUC).

    ``data`` is a :class:`~fairllms.metrics.data.LabeledSentences`. An AUC near
    0.5 means the attribute is not linearly recoverable from the encoder
    representation; near 1.0 means it is.

    Parameters
    ----------
    test_ratio:
        Held-out fraction per split.
    seed:
        Base RNG seed.
    n_seeds:
        Number of random splits averaged.
    """

    name = "counterfactual_auc"
    bias_type = "extrinsic"
    architectures = ("encoder_decoder",)

    def __init__(self, *, test_ratio: float = 0.2, seed: int = 42, n_seeds: int = 10):
        self.test_ratio = test_ratio
        self.seed = seed
        self.n_seeds = n_seeds

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = unwrap(
            data if data is not None else legacy.pop("dataset", None), LabeledSentences
        )

        if data is None:
            sents, k1 = take(legacy, "sentences")
            labels, k2 = take(legacy, "labels")
            if None in (sents, labels):
                raise ValueError(
                    "CounterfactualAucScore requires sentences and labels. Pass a "
                    "LabeledSentences as the second argument, e.g. "
                    "CounterfactualAucScore().compute(model, "
                    "LabeledSentences(sentences, labels))."
                )
            warn_legacy("CounterfactualAucScore", [k1, k2], "LabeledSentences")
            legacy.pop(k1, None)
            legacy.pop(k2, None)
            pair_ids, k3 = take(legacy, "pair_ids")
            legacy.pop(k3, None)
            data = LabeledSentences(sents, labels, pair_ids=pair_ids)

        self._reject_unknown_kwargs(legacy, "test_ratio", "seed", "n_seeds", "pair_ids")

        if not isinstance(data, LabeledSentences):
            data = LabeledSentences.from_examples(
                as_examples(
                    data,
                    "CounterfactualAucScore",
                    "a LabeledSentences or sequence of (sentence, label)",
                )
            )

        tok, hf_model, _ = get_tokenizer_model(model, tokenizer)
        auc_mean, auc_std, n0, n1, rows = compute_auc(
            hf_model,
            tok,
            list(data.sentences),
            list(data.labels),
            pair_ids=list(data.pair_ids) if data.pair_ids is not None else None,
            test_ratio=legacy.get("test_ratio", self.test_ratio),
            seed=legacy.get("seed", self.seed),
            n_seeds=legacy.get("n_seeds", self.n_seeds),
        )
        return MetricResult(
            score=float(auc_mean),
            details={
                "auc_mean": float(auc_mean),
                "auc_std": auc_std,
                "n_class_0": n0,
                "n_class_1": n1,
                "rows": rows,
            },
        )


class InferenceBiasScore(FairnessMetric):
    """Idealized Bias Score (IBS) over ``(label, prediction)`` pairs.

    ``data`` is a sequence of 2-tuples. No model is used.
    """

    name = "inference_bias_score"
    bias_type = "extrinsic"
    architectures = ("encoder_decoder",)

    def compute(
        self, model: Any = None, data: Any = None, **legacy: Any
    ) -> MetricResult:
        data = unwrap(data if data is not None else legacy.pop("dataset", None))
        if data is None:
            data, key = take(legacy, "predictions")
            if data is None:
                raise ValueError(
                    "InferenceBiasScore requires (label, prediction) pairs. Pass "
                    "them as the second argument."
                )
            warn_legacy("InferenceBiasScore", [key], "the (label, prediction) pairs")
            legacy.pop(key, None)
        self._reject_unknown_kwargs(legacy)

        pairs = as_examples(
            data, "InferenceBiasScore", "a sequence of (label, prediction) pairs"
        )
        bad = [p for p in pairs if not isinstance(p, (list, tuple)) or len(p) < 2]
        if bad:
            raise ValueError(
                f"InferenceBiasScore: each item must be a (label, prediction) pair; "
                f"got {bad[0]!r}."
            )
        ibs, counts = compute_ibs(pairs)
        return MetricResult(
            score=float(ibs), details={"counts": counts, "n": len(pairs)}
        )


class NormalizedPositionDistance(FairnessMetric):
    """Normalized position disparity for summarization (position bias).

    ``data`` is a sequence of article strings (or dicts with an ``article`` key).

    Parameters
    ----------
    max_new_tokens:
        Generation budget per article.
    K:
        Number of leading sentences treated as the lead baseline.
    gold_summaries:
        Optional reference summaries, aligned with the articles.
    """

    name = "normalized_position_distance"
    bias_type = "extrinsic"
    architectures = ("encoder_decoder",)

    def __init__(
        self,
        *,
        max_new_tokens: int = 128,
        K: int = 10,
        gold_summaries: Optional[Sequence[str]] = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.K = K
        self.gold_summaries = gold_summaries

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = unwrap(data if data is not None else legacy.pop("dataset", None))
        if data is None:
            data, key = take(legacy, "articles")
            if data is None:
                raise ValueError(
                    "NormalizedPositionDistance requires articles. Pass them as the "
                    "second argument, e.g. compute(model, [article_text, ...])."
                )
            warn_legacy("NormalizedPositionDistance", [key], "the article list")
            legacy.pop(key, None)

        self._reject_unknown_kwargs(legacy, "max_new_tokens", "K", "gold_summaries")
        articles = _texts(
            data, "NormalizedPositionDistance", "a sequence of articles", "article"
        )

        tok, hf_model, _ = get_tokenizer_model(model, tokenizer)
        mean_npd, rows = compute_npd(
            hf_model,
            tok,
            articles,
            gold_summaries=legacy.get("gold_summaries", self.gold_summaries),
            max_new_tokens=legacy.get("max_new_tokens", self.max_new_tokens),
            K=legacy.get("K", self.K),
        )
        return MetricResult(
            score=float(mean_npd), details={"rows": rows, "n_articles": len(articles)}
        )


class TranslationSimilarityScore(FairnessMetric):
    """Semantic similarity of counterfactual translations (LaBSE / SS).

    ``data`` is a sequence of ``(original, counterfactual)`` sentence pairs.

    Parameters
    ----------
    labse_model, labse_tokenizer:
        The sentence encoder used to measure similarity of the two translations.
        Required — there is no sensible default.
    tgt_lang:
        Target language name inserted into the translation prompt.
    max_new_tokens:
        Generation budget per sentence.
    """

    name = "translation_similarity_score"
    bias_type = "extrinsic"
    architectures = ("encoder_decoder",)

    def __init__(
        self,
        *,
        labse_model: Any = None,
        labse_tokenizer: Any = None,
        tgt_lang: str = "French",
        max_new_tokens: int = 128,
    ):
        self.labse_model = labse_model
        self.labse_tokenizer = labse_tokenizer
        self.tgt_lang = tgt_lang
        self.max_new_tokens = max_new_tokens

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = unwrap(data if data is not None else legacy.pop("dataset", None))
        if data is None:
            data, key = take(legacy, "pairs")
            if data is None:
                raise ValueError(
                    "TranslationSimilarityScore requires (original, counterfactual) "
                    "pairs. Pass them as the second argument."
                )
            warn_legacy("TranslationSimilarityScore", [key], "the sentence pairs")
            legacy.pop(key, None)

        self._reject_unknown_kwargs(
            legacy, "labse_model", "labse_tokenizer", "tgt_lang", "max_new_tokens"
        )
        labse_model = legacy.get("labse_model", self.labse_model)
        labse_tokenizer = legacy.get("labse_tokenizer", self.labse_tokenizer)
        if labse_model is None or labse_tokenizer is None:
            raise ValueError(
                "TranslationSimilarityScore needs a sentence encoder to compare "
                "translations. Pass labse_model= and labse_tokenizer= to the "
                "constructor."
            )

        pairs = as_examples(
            data,
            "TranslationSimilarityScore",
            "a sequence of (original, counterfactual) pairs",
        )
        bad = [p for p in pairs if not isinstance(p, (list, tuple)) or len(p) < 2]
        if bad:
            raise ValueError(
                f"TranslationSimilarityScore: each item must be an "
                f"(original, counterfactual) pair; got {bad[0]!r}."
            )

        tok, hf_model, _ = get_tokenizer_model(model, tokenizer)
        mean_ss, std_ss, rows = compute_translation_ss(
            hf_model,
            tok,
            labse_model,
            labse_tokenizer,
            pairs,
            tgt_lang=legacy.get("tgt_lang", self.tgt_lang),
            max_new_tokens=legacy.get("max_new_tokens", self.max_new_tokens),
        )
        return MetricResult(
            score=float(mean_ss),
            details={"mean": float(mean_ss), "std": std_ss, "rows": rows, "n": len(pairs)},
        )
