"""Helpers to normalize model / dataset arguments for metric wrappers."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch

from fairllms.models.base import LoadedModel, ModelAdapter
from fairllms.models.openai import OpenAILoadedModel, OpenAIModel


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


def get_tokenizer_model(
    model: Any,
    tokenizer: Any = None,
) -> Tuple[Any, Any, torch.device]:
    """Return ``(tokenizer, model, device)`` from adapters or raw objects."""
    if model is None:
        raise TypeError("model is required for this metric")

    if isinstance(model, ModelAdapter):
        loaded = model.load()
        return loaded.tokenizer, loaded.model, loaded.device

    if isinstance(model, LoadedModel):
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
