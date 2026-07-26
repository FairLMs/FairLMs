"""WEAT wrapper."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from fairLLMs.definition.encoder_only.intrinsic_bias.similarity_based.weat.weat import (
    compute_weat,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_tokenizer_model
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
                # dataset may be a dict-like stimulus set
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
