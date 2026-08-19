"""fairlms: fairness definitions and bias metrics for large language models.

Shared infrastructure: ``datasets``, ``models``, ``utils``.
Public metric API: ``metrics`` (sklearn-style ``metric.compute(...)``).
"""

from fairlms import datasets, metrics, models, utils
from fairlms._version import __version__

__all__ = ["datasets", "metrics", "models", "utils", "__version__"]
