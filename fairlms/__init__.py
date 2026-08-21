"""fairlms: fairness definitions and bias metrics for large language models.

Shared infrastructure: ``datasets``, ``models``, ``utils``.
Public metric API: ``metrics`` (sklearn-style ``metric.compute(...)``).
Bundled evidence: ``data`` (ready-to-use WEAT/SEAT word sets).
"""

from fairlms import data, datasets, metrics, models, utils
from fairlms._version import __version__

__all__ = ["data", "datasets", "metrics", "models", "utils", "__version__"]
