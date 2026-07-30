"""Shared fixtures. Model-backed tests skip cleanly when weights are unavailable."""

import pytest


def _try_load(task):
    from fairllms.models import HuggingFaceModel

    try:
        adapter = HuggingFaceModel("bert-base-uncased", task=task)
        adapter.load()
        return adapter
    except Exception as exc:  # offline, no cache, gated, …
        pytest.skip(f"bert-base-uncased ({task}) unavailable: {type(exc).__name__}")


@pytest.fixture(scope="session")
def encoder_model():
    """A small cached encoder. Skips the test if it cannot be loaded."""
    return _try_load("encoder")


@pytest.fixture
def word_sets():
    from fairllms.metrics import WordSets

    return WordSets(
        target_1=["Adam", "Chip", "Harry", "Josh"],
        target_2=["Alonzo", "Jamel", "Lerone", "Percell"],
        attribute_1=["caress", "freedom", "health", "love"],
        attribute_2=["abuse", "crash", "filth", "murder"],
    )


@pytest.fixture
def context_sets():
    from fairllms.metrics import ContextSets

    return ContextSets(
        target_1={"Adam": ["Adam went home.", "Adam is here."]},
        target_2={"Alonzo": ["Alonzo went home.", "Alonzo is here."]},
        attribute_1={"freedom": ["Freedom matters.", "We value freedom."]},
        attribute_2={"abuse": ["Abuse is harmful.", "They reported abuse."]},
    )


@pytest.fixture
def vector_sets():
    import numpy as np

    from fairllms.metrics import VectorSets

    rng = np.random.default_rng(0)
    return VectorSets(*(rng.normal(size=(4, 16)) for _ in range(4)))
