"""Similarity-based association metrics: WEAT, SEAT, CEAT."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from fairLLMs.definition.encoder_only.intrinsic_bias.similarity_based.ceat.ceat import (
    compute_ceat,
)
from fairLLMs.definition.encoder_only.intrinsic_bias.similarity_based.seat.seat import (
    compute_seat,
)
from fairLLMs.definition.encoder_only.intrinsic_bias.similarity_based.weat.weat import (
    compute_weat,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_tokenizer_model, require_kwargs
from fairLLMs.utils import encode_sentence


class WEAT(FairnessMetric):
    """Word Embedding Association Test.

    Prefer precomputed vectors via ``T1_vecs``, ``T2_vecs``, ``A_vecs``,
    ``B_vecs``. Alternatively pass word lists (``T1_terms``, …) plus an encoder
    ``model`` to embed them.
    """

    name = "weat"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)

    def __init__(self, pooling: str = "mean", n_samples: int = 10_000):
        self.pooling = pooling
        self.n_samples = n_samples

    def _embed_terms(self, model, tokenizer, terms, device):
        vecs = []
        for term in terms:
            vec = encode_sentence(
                model, tokenizer, term, pooling=self.pooling, device=device
            )
            vecs.append(np.asarray(vec))
        return vecs

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        T1 = kwargs.get("T1_vecs")
        T2 = kwargs.get("T2_vecs")
        A = kwargs.get("A_vecs")
        B = kwargs.get("B_vecs")

        if None in (T1, T2, A, B):
            terms = (
                kwargs.get("T1_terms"),
                kwargs.get("T2_terms"),
                kwargs.get("A_terms") or kwargs.get("A1_terms"),
                kwargs.get("B_terms") or kwargs.get("A2_terms"),
            )
            if dataset is not None and any(t is None for t in terms):
                data = dataset if isinstance(dataset, dict) else None
                if data is None and hasattr(dataset, "load"):
                    loaded = dataset.load()
                    data = loaded if isinstance(loaded, dict) else None
                if isinstance(data, dict):
                    terms = (
                        data.get("t1") or data.get("T1"),
                        data.get("t2") or data.get("T2"),
                        data.get("a1") or data.get("A1"),
                        data.get("a2") or data.get("A2"),
                    )
            if any(t is None for t in terms):
                raise ValueError(
                    "Provide T1_vecs/T2_vecs/A_vecs/B_vecs or term lists "
                    "(T1_terms, T2_terms, A_terms, B_terms) with a model"
                )
            if model is None:
                raise ValueError("model is required to embed WEAT term lists")
            tokenizer, hf_model, device = get_tokenizer_model(
                model, kwargs.get("tokenizer")
            )
            T1, T2, A, B = (
                self._embed_terms(hf_model, tokenizer, terms[0], device),
                self._embed_terms(hf_model, tokenizer, terms[1], device),
                self._embed_terms(hf_model, tokenizer, terms[2], device),
                self._embed_terms(hf_model, tokenizer, terms[3], device),
            )

        d, p = compute_weat(T1, T2, A, B)
        return MetricResult(score=float(d), details={"p_value": float(p)})


class SEAT(FairnessMetric):
    """Sentence Encoder Association Test."""

    name = "seat"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)

    def __init__(self, pooling: str = "mean", n_samples: int = 10_000, templates=None):
        self.pooling = pooling
        self.n_samples = n_samples
        self.templates = templates

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, device = get_tokenizer_model(model, kwargs.get("tokenizer"))

        T1 = kwargs.get("T1_terms")
        T2 = kwargs.get("T2_terms")
        A1 = kwargs.get("A1_terms")
        A2 = kwargs.get("A2_terms")

        if dataset is not None and None in (T1, T2, A1, A2):
            data = dataset if isinstance(dataset, dict) else None
            if data is None and hasattr(dataset, "load"):
                loaded = dataset.load()
                data = loaded if isinstance(loaded, dict) else None
            if isinstance(data, dict):
                T1 = T1 or data.get("t1") or data.get("T1")
                T2 = T2 or data.get("t2") or data.get("T2")
                A1 = A1 or data.get("a1") or data.get("A1")
                A2 = A2 or data.get("a2") or data.get("A2")

        if None in (T1, T2, A1, A2):
            raise ValueError("Provide T1_terms, T2_terms, A1_terms, A2_terms")

        effect, p = compute_seat(
            hf_model,
            tokenizer,
            T1,
            T2,
            A1,
            A2,
            templates=kwargs.get("templates", self.templates),
            pooling=kwargs.get("pooling", self.pooling),
            n_samples=kwargs.get("n_samples", self.n_samples),
            device=device,
        )
        return MetricResult(score=float(effect), details={"p_value": float(p)})


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
