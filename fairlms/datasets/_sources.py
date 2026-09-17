"""Where loaders get their bytes: a local directory first, the Hub second.

Six of the benchmarks in this package are published as plain files in a Hugging
Face dataset repository, four are only distributed from their own project page,
and one is a set of templates with no release at all. The helpers here give all
of them the same two-step resolution -- an explicit ``root=`` the caller passes,
otherwise the packaged fallback -- and one failure message that names the
directory layout that was expected and where to get it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

PathLike = Union[str, Path]


def hub_file(
    repo_id: str,
    filename: str,
    revision: Optional[str] = None,
) -> Path:
    """Download one file from a Hub *dataset* repo; return the cached path.

    ``hf_hub_download`` rather than ``datasets.load_dataset`` on purpose. Several
    of these repositories still carry a loading script (``honest.py``,
    ``gap.py``, ``equity_evaluation_corpus.py``), and ``datasets>=3`` refuses to
    execute one, so ``load_dataset(repo_id)`` fails outright on a current
    install. Fetching the published data file works on every ``datasets``
    version and never runs repository code.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - huggingface_hub is a dep
        raise RuntimeError(
            "huggingface_hub is required to download this dataset. "
            "Install it with `pip install huggingface_hub`, or pass a local "
            "`root=` pointing at an existing copy."
        ) from exc

    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
            )
        )
    except Exception as exc:  # noqa: BLE001 - network, auth, gating, 404, ...
        raise RuntimeError(
            f"Failed to download {filename!r} from the Hugging Face dataset "
            f"repo {repo_id!r}: {type(exc).__name__}: {exc}. Pass `root=` to "
            "read a local copy instead."
        ) from exc


def resolve_root(
    root: Optional[PathLike],
    *,
    dataset: str,
    directory: str,
    sentinels: Sequence[PathLike],
    homepage: str,
) -> Path:
    """Return the dataset directory holding any one of ``sentinels``.

    ``root`` may be either the dataset directory itself or a parent holding a
    ``directory`` subfolder, so both ``root="data"`` and
    ``root="data/RedditBias"`` work against the layout these benchmarks ship in.
    """
    expected = " or ".join(str(s) for s in sentinels)
    if root is None:
        raise FileNotFoundError(
            f"{dataset} is not redistributed with this package and has no "
            f"Hugging Face release this loader can rely on. Download it from "
            f"{homepage}, then pass `root=` pointing at the directory holding "
            f"{expected}."
        )

    base = Path(root).expanduser()
    for candidate in (base, base / directory):
        if any((candidate / sentinel).exists() for sentinel in sentinels):
            return candidate.resolve()

    raise FileNotFoundError(
        f"{dataset} not found under {base}: no {expected} in {base} or "
        f"{base / directory}. Download the dataset from {homepage}."
    )


def require_choice(value: str, allowed: Sequence[str], label: str) -> str:
    if value not in allowed:
        raise ValueError(
            f"Unknown {label} {value!r}. Expected one of: {', '.join(allowed)}."
        )
    return value
