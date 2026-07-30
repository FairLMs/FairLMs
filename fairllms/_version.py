"""Single source of truth for the package version.

Kept free of imports so the build backend can read ``__version__`` statically
(``[tool.setuptools.dynamic]`` in ``pyproject.toml``) without importing
``fairllms``, which would require torch at build time.

Bump this one value; ``pyproject.toml`` and ``fairllms.__version__`` follow.
"""

__version__ = "0.2.0"
