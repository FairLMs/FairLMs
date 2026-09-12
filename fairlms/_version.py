"""Single source of truth for the package version.

Kept free of imports so the build backend can read ``__version__`` statically
(``[tool.setuptools.dynamic]`` in ``pyproject.toml``) without importing
``fairlms``, which would require torch at build time.

Bump this one value; ``pyproject.toml`` and ``fairlms.__version__`` follow.
"""

__version__ = "0.5.0.dev0"
