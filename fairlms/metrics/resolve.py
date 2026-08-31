"""Helpers to normalize model / dataset arguments for metric wrappers."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch

from fairlms.models.base import LoadedModel, ModelAdapter
from fairlms.models.openai import OpenAILoadedModel, OpenAIModel


def get_examples(dataset: Any) -> Optional[List[Any]]:
    """Load examples from a FairnessDataset or pass through a sequence."""
    if dataset is None:
        return None
    if hasattr(dataset, "load") and callable(dataset.load):
        return list(dataset.load())
    if isinstance(dataset, (list, tuple)):
        return list(dataset)
    # Hugging Face Dataset / other iterables
    try:
        return list(dataset)
    except TypeError as exc:
        raise TypeError(
            f"dataset must be a FairnessDataset or sequence, got {type(dataset)}"
        ) from exc


def check_task(loaded: Any, metric: Any) -> None:
    """Refuse a model loaded for the wrong head.

    A metric declares ``required_task``; a :class:`~fairlms.models.LoadedModel`
    records the ``task`` it was loaded with. When both are known and disagree,
    the metric would otherwise fail deep in its own numerics: with an
    ``AttributeError`` on a missing ``.logits``, or, worse, with plausible
    numbers read off a randomly initialized head. Fail here instead, naming
    both sides.

    Silent when either side is unknown: raw ``(tokenizer, model)`` tuples carry
    no task, and metrics that work on any head leave ``required_task`` as
    ``None``.
    """
    required = getattr(metric, "required_task", None)
    if required is None:
        return
    actual = getattr(loaded, "task", None)
    if actual is None or actual == required:
        return
    name = type(metric).__name__ if not isinstance(metric, str) else metric
    raise TypeError(
        f"{name} requires a model loaded with task={required!r}, got "
        f"task={actual!r}. Reload the checkpoint as "
        f"HuggingFaceModel(name, task={required!r}). The task selects which "
        f"head is attached, and this metric reads a quantity that "
        f"task={actual!r} does not expose."
    )


def get_tokenizer_model(
    model: Any,
    tokenizer: Any = None,
    *,
    metric: Any = None,
) -> Tuple[Any, Any, torch.device]:
    """Return ``(tokenizer, model, device)`` from adapters or raw objects.

    Pass ``metric=self`` from a metric to have the model's ``task`` checked
    against that metric's ``required_task``. See :func:`check_task`.
    """
    if model is None:
        raise TypeError("model is required for this metric")

    if isinstance(model, ModelAdapter):
        loaded = model.load()
        check_task(loaded, metric)
        return loaded.tokenizer, loaded.model, loaded.device

    if isinstance(model, LoadedModel):
        check_task(model, metric)
        return model.tokenizer, model.model, model.device

    if isinstance(model, (tuple, list)) and len(model) >= 2:
        tok, mod = model[0], model[1]
        if len(model) >= 3 and model[2] is not None:
            device = torch.device(model[2]) if not isinstance(model[2], torch.device) else model[2]
        else:
            device = next(mod.parameters()).device
        return tok, mod, device

    if tokenizer is not None:
        device = next(model.parameters()).device
        return tokenizer, model, device

    raise TypeError(
        "Pass a HuggingFaceModel / LoadedModel, a (tokenizer, model) tuple, "
        "or model=... with tokenizer=..."
    )


def get_openai_bundle(model: Any) -> OpenAILoadedModel:
    """Return an OpenAI client bundle from an adapter or loaded object."""
    if model is None:
        return OpenAIModel().load()
    if isinstance(model, OpenAIModel):
        return model.load()
    if isinstance(model, OpenAILoadedModel):
        return model
    if hasattr(model, "completions") or hasattr(model, "chat"):
        # Raw OpenAI client
        return OpenAILoadedModel(name="openai", client=model, model="davinci-002")
    raise TypeError(
        "Pass an OpenAIModel / OpenAILoadedModel, or a raw OpenAI client"
    )


def require_kwargs(kwargs: dict, *keys: str) -> None:
    missing = [k for k in keys if k not in kwargs or kwargs[k] is None]
    if missing:
        raise TypeError(f"Missing required argument(s): {', '.join(missing)}")
