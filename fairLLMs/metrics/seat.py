"""SEAT wrapper."""

from __future__ import annotations

from typing import Any

from fairLLMs.definition.encoder_only.intrinsic_bias.similarity_based.seat.seat import (
    compute_seat,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import get_tokenizer_model


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
