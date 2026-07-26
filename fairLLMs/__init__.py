"""fairLLMs: fairness definitions and bias metrics for large language models.

Shared infrastructure: ``datasets``, ``models``, ``utils``.
Public metric API: ``metrics`` (sklearn-style ``metric.compute(...)``).
"""

from fairLLMs import datasets, metrics, models, utils

__all__ = ["datasets", "metrics", "models", "utils"]
__version__ = "0.1.0"
