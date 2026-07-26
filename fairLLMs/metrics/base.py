"""Base contracts for sklearn-style fairness metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


@dataclass
class MetricResult:
    """Standard return type for every :meth:`FairnessMetric.compute` call."""

    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    by_category: Optional[Dict[str, Any]] = None

    def __float__(self) -> float:
        return float(self.score)

    def __repr__(self) -> str:
        parts = [f"score={self.score!r}"]
        if self.by_category is not None:
            parts.append(f"by_category={self.by_category!r}")
        if self.details:
            parts.append(f"details_keys={list(self.details)}")
        return f"MetricResult({', '.join(parts)})"


class FairnessMetric(ABC):
    """Abstract base class: every metric exposes ``compute``."""

    name: str = "metric"
    bias_type: str = ""  # "intrinsic" | "extrinsic"
    architectures: Tuple[str, ...] = ()

    @abstractmethod
    def compute(
        self,
        model: Any = None,
        dataset: Any = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Evaluate this metric.

        Parameters
        ----------
        model:
            A :class:`~fairLLMs.models.HuggingFaceModel`,
            :class:`~fairLLMs.models.LoadedModel`,
            :class:`~fairLLMs.models.OpenAIModel`, or (for some metrics)
            precomputed inputs when ``model`` is unused.
        dataset:
            A :class:`~fairLLMs.datasets.FairnessDataset`, a sequence of
            examples, or metric-specific structures documented on the subclass.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
