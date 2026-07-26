"""fairLLMs: fairness definitions and bias metrics for large language models.

Phase 1 shared infrastructure lives under ``datasets``, ``models``, and
``utils``. Metric leaf scripts under ``definition/`` remain runnable; Phase 2
will wrap them behind a sklearn-style ``metric.compute(...)`` API.
"""

from fairLLMs import datasets, models, utils

__all__ = ["datasets", "models", "utils"]
__version__ = "0.1.0"
