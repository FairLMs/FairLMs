"""fairllms: fairness definitions and bias metrics for large language models.

Shared infrastructure: ``datasets``, ``models``, ``utils``.
Public metric API: ``metrics`` (sklearn-style ``metric.compute(...)``).
"""

from fairllms import datasets, metrics, models, utils
from fairllms._version import __version__

__all__ = ["datasets", "metrics", "models", "utils", "__version__"]
