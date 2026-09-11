"""Base contracts for sklearn-style fairness metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from fairlms.params import ParameterizedComponent


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


class FairnessMetric(ParameterizedComponent, ABC):
    """Abstract base class: every metric exposes ``compute``.

    Follows scikit-learn's estimator conventions, inherited from
    :class:`~fairlms.params.ParameterizedComponent`: ``__init__`` takes
    configuration only and stores it verbatim, data goes to :meth:`compute`, and
    ``get_params`` / ``set_params`` come for free.

    A metric also **declares what it needs** - architectures, capabilities and
    evidence containers - so an unsatisfiable pairing is refused by name rather
    than failing deep inside a forward pass. See :mod:`fairlms.applicability`.
    """

    name: str = "metric"
    bias_type: str = ""  # "intrinsic" | "extrinsic"
    architectures: Tuple[str, ...] = ()

    #: Capabilities the model must expose, from
    #: :data:`fairlms.applicability.CAPABILITIES`. Empty means the metric reads
    #: no model output: it scores precomputed predictions.
    requires: frozenset = frozenset()

    #: Evidence container types :meth:`compute` consumes as ``data``. Empty
    #: means no declared constraint.
    accepts: Tuple[type, ...] = ()

    #: The ``task`` a Hugging Face checkpoint must be loaded with for this
    #: metric to read the quantity it is defined on: one of ``mlm``,
    #: ``encoder``, ``sequence_classification``, ``seq2seq``, ``causal``.
    #: Checked by :func:`fairlms.metrics.resolve.check_task` when the metric
    #: resolves a model. ``None`` means the metric imposes no requirement:
    #: it scores precomputed predictions, calls an API, or accepts any head.
    required_task: Optional[str] = None

    @abstractmethod
    def compute(
        self,
        model: Any = None,
        data: Any = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Evaluate this metric.

        Parameters
        ----------
        model:
            A :class:`~fairlms.models.HuggingFaceModel`,
            :class:`~fairlms.models.LoadedModel`,
            :class:`~fairlms.models.OpenAIModel`, or (for metrics that score
            precomputed predictions) ``None``.
        data:
            The inputs this metric scores. Metrics that consume a corpus accept
            a :class:`~fairlms.datasets.FairnessDataset` or sequence of
            examples; metrics that need structured word/context sets accept a
            typed container from :mod:`fairlms.metrics.data`.
        """
