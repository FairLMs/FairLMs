"""Model adapters for local Hugging Face models and OpenAI APIs."""

from fairllms.models.base import LoadedModel, ModelAdapter
from fairllms.models.huggingface import (
    HuggingFaceModel,
    load_causal_lm,
    load_encoder,
    load_masked_lm,
    load_seq2seq,
    load_sequence_classifier,
    resolve_device,
)
from fairllms.models.openai import OpenAILoadedModel, OpenAIModel, get_openai_client

__all__ = [
    "HuggingFaceModel",
    "LoadedModel",
    "ModelAdapter",
    "OpenAILoadedModel",
    "OpenAIModel",
    "get_openai_client",
    "load_causal_lm",
    "load_encoder",
    "load_masked_lm",
    "load_seq2seq",
    "load_sequence_classifier",
    "resolve_device",
]
