"""Base contracts for sklearn-style fairness metrics."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
    """Abstract base class: every metric exposes ``compute``.

    Follows scikit-learn's estimator conventions:

    * ``__init__`` takes **configuration only**, stores each argument verbatim
      under its own name, and performs no validation or computation. This is
      what makes :meth:`get_params` / :meth:`set_params` (and therefore cloning
      and parameter sweeps) possible.
    * **Data** is passed to :meth:`compute`, never to ``__init__``.
    * :meth:`get_params` introspects ``__init__``, so subclasses get parameter
      introspection for free just by declaring named arguments.
    """

    name: str = "metric"
    bias_type: str = ""  # "intrinsic" | "extrinsic"
    architectures: Tuple[str, ...] = ()

    # -- sklearn-style parameter introspection ------------------------------
    @classmethod
    def _param_names(cls) -> List[str]:
        """Names of this metric's configuration parameters, from ``__init__``."""
        init = cls.__init__
        if init is FairnessMetric.__init__ or init is object.__init__:
            return []
        params = inspect.signature(init).parameters.values()
        names = [
            p.name
            for p in params
            if p.name != "self"
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]
        return sorted(names)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return this metric's configuration as a dict.

        Mirrors ``sklearn.base.BaseEstimator.get_params``, so
        ``type(m)(**m.get_params())`` reconstructs an equivalent metric and
        ``sklearn.base.clone`` works on fairllms metrics.

        Parameters
        ----------
        deep:
            When ``True``, also report the parameters of any nested object that
            exposes ``get_params``, under ``name__subname`` keys — sklearn's
            convention. Accepting this argument is what lets sklearn utilities
            (which always pass ``deep``) operate on these metrics.
        """
        params: Dict[str, Any] = {}
        for name in self._param_names():
            value = getattr(self, name)
            params[name] = value
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                for sub_name, sub_value in value.get_params(deep=True).items():
                    params[f"{name}__{sub_name}"] = sub_value
        return params

    def set_params(self, **params: Any) -> "FairnessMetric":
        """Set configuration parameters in place and return ``self``."""
        valid = self._param_names()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"Invalid parameter {key!r} for {type(self).__name__}. "
                    f"Valid parameters are: {', '.join(valid) or '(none)'}."
                )
            setattr(self, key, value)
        return self

    def _reject_unknown_kwargs(self, kwargs: Dict[str, Any], *allowed: str) -> None:
        """Raise ``TypeError`` for kwargs this metric does not understand.

        ``compute(**kwargs)`` signatures silently swallow typos, so a misspelled
        ``n_bootstrp=`` quietly uses the default instead of failing. Metrics call
        this to make unknown keys an error, as sklearn does.
        """
        unknown = sorted(set(kwargs) - set(allowed))
        if unknown:
            raise TypeError(
                f"{type(self).__name__}.compute() got unexpected keyword "
                f"argument(s): {', '.join(unknown)}. "
                f"Accepted: {', '.join(sorted(allowed)) or '(none)'}."
            )

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
            A :class:`~fairllms.models.HuggingFaceModel`,
            :class:`~fairllms.models.LoadedModel`,
            :class:`~fairllms.models.OpenAIModel`, or (for metrics that score
            precomputed predictions) ``None``.
        data:
            The inputs this metric scores. Metrics that consume a corpus accept
            a :class:`~fairllms.datasets.FairnessDataset` or sequence of
            examples; metrics that need structured word/context sets accept a
            typed container from :mod:`fairllms.metrics.data`.
        """

    def __repr__(self) -> str:
        params = self.get_params(deep=False)
        defaults = {}
        if params:
            sig = inspect.signature(type(self).__init__).parameters
            defaults = {
                k: v.default
                for k, v in sig.items()
                if v.default is not inspect.Parameter.empty
            }
        shown = [
            f"{k}={v!r}"
            for k, v in sorted(params.items())
            if k not in defaults or defaults[k] != v
        ]
        return f"{type(self).__name__}({', '.join(shown)})"
