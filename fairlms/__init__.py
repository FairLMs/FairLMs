"""fairlms: fairness definitions and bias metrics for large language models.

Shared infrastructure: ``datasets``, ``models``, ``utils``.
Public metric API: ``metrics`` (sklearn-style ``metric.compute(...)``).
Public mitigation API: ``mitigation`` (``mitigator.apply(model, evidence)``).
Dataset-first audits: ``diagnostics``.
Bundled evidence: ``data`` (ready-to-use WEAT/SEAT word sets).

Every component - metric or mitigator - declares the model capabilities and
evidence it needs, so it runs against whatever satisfies them and is refused by
name otherwise. See ``applicability``.
"""

from fairlms import (
    applicability,
    data,
    datasets,
    diagnostics,
    metrics,
    mitigation,
    models,
    utils,
)
from fairlms._version import __version__

__all__ = [
    "applicability",
    "data",
    "datasets",
    "diagnostics",
    "metrics",
    "mitigation",
    "models",
    "utils",
    "__version__",
]
