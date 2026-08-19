"""Focused tests for the refactored similarity-based family (WEAT / SEAT / CEAT).

Covers the three things the refactor was meant to buy:

1. Malformed input fails immediately with an actionable message.
2. Configuration declared in ``__init__`` actually reaches the computation
   (several of these params used to be silently ignored).
3. The legacy keyword API still works, but warns.
"""

import warnings

import numpy as np
import pytest

from fairlms.metrics import CEAT, SEAT, WEAT, ContextSets, VectorSets, WordSets

TERMS = (
    ["Adam", "Chip", "Harry", "Josh"],
    ["Alonzo", "Jamel", "Lerone", "Percell"],
    ["caress", "freedom", "health", "love"],
    ["abuse", "crash", "filth", "murder"],
)


# --------------------------------------------------------------------------
# Container validation
# --------------------------------------------------------------------------
class TestWordSets:
    def test_rejects_bare_string(self):
        with pytest.raises(TypeError, match="wrap it in a list"):
            WordSets("Adam", ["a"], ["b"], ["c"])

    def test_rejects_empty_role(self):
        with pytest.raises(ValueError, match="at least one term"):
            WordSets([], ["a"], ["b"], ["c"])

    def test_rejects_non_string_members(self):
        with pytest.raises(TypeError, match="only strings"):
            WordSets([1], ["a"], ["b"], ["c"])

    def test_normalizes_to_tuples(self):
        ws = WordSets(*TERMS)
        assert isinstance(ws.target_1, tuple)

    def test_balanced_targets_check(self):
        WordSets(["a", "b"], ["c", "d"], ["e"], ["f"]).require_balanced_targets("X")
        with pytest.raises(ValueError, match="same length"):
            WordSets(["a", "b"], ["c"], ["e"], ["f"]).require_balanced_targets("X")


class TestVectorSets:
    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            VectorSets([1, 2], [3, 4], [5, 6], [7, 8])

    def test_rejects_mismatched_dims(self):
        with pytest.raises(ValueError, match="share an embedding dimension"):
            VectorSets([[1, 2]], [[1, 2, 3]], [[1, 2]], [[1, 2]])

    def test_reports_n_dims(self):
        rng = np.random.default_rng(0)
        assert VectorSets(*(rng.normal(size=(3, 16)) for _ in range(4))).n_dims == 16


class TestContextSets:
    def test_rejects_flat_list(self):
        with pytest.raises(TypeError, match="must be a mapping"):
            ContextSets(["a"], ["b"], ["c"], ["d"])

    def test_min_contexts(self, context_sets):
        assert context_sets.min_contexts() == 2

    def test_require_contexts(self, context_sets):
        context_sets.require_contexts(2, "CEAT")
        with pytest.raises(ValueError, match="sample_size=5"):
            context_sets.require_contexts(5, "CEAT")


# --------------------------------------------------------------------------
# Config actually reaches the computation
# --------------------------------------------------------------------------
class TestConfigIsHonoured:
    def test_weat_n_samples_threaded(self, vector_sets):
        """``n_samples`` used to be hardcoded to 10_000 inside compute_weat."""
        result = WEAT(n_samples=250).compute(None, vector_sets)
        assert result.details["n_samples"] == 250

    def test_weat_is_deterministic_for_fixed_vectors(self, vector_sets):
        a = WEAT(n_samples=200).compute(None, vector_sets).score
        b = WEAT(n_samples=200).compute(None, vector_sets).score
        assert a == b

    @pytest.mark.parametrize("pooling", ["mean", "cls"])
    def test_seat_pooling_reported(self, encoder_model, pooling):
        result = SEAT(n_samples=100, pooling=pooling).compute(
            encoder_model, WordSets(*TERMS)
        )
        assert result.details["pooling"] == pooling

    def test_seat_pooling_changes_result(self, encoder_model):
        """``pooling`` was accepted but hardcoded to "mean"; it must now matter."""
        ws = WordSets(*TERMS)
        mean = SEAT(n_samples=100, pooling="mean").compute(encoder_model, ws).score
        cls_ = SEAT(n_samples=100, pooling="cls").compute(encoder_model, ws).score
        assert mean != cls_

    def test_ceat_seed_makes_runs_reproducible(self, encoder_model, context_sets):
        kw = dict(sample_size=2, n_trials=5, seed=7)
        a = CEAT(**kw).compute(encoder_model, context_sets).score
        b = CEAT(**kw).compute(encoder_model, context_sets).score
        assert a == b


# --------------------------------------------------------------------------
# Argument handling
# --------------------------------------------------------------------------
class TestArgumentHandling:
    def test_unknown_kwarg_raises(self, vector_sets):
        with pytest.raises(TypeError, match="n_bootstrp"):
            WEAT().compute(None, vector_sets, n_bootstrp=20)

    def test_weat_without_model_needs_vectors(self):
        with pytest.raises(ValueError, match="needs a model"):
            WEAT().compute(None, WordSets(*TERMS))

    def test_weat_with_no_data_at_all(self):
        with pytest.raises(ValueError, match="four term sets"):
            WEAT().compute(None)

    def test_wrong_container_type_is_reported(self, encoder_model):
        with pytest.raises(TypeError, match="expects a ContextSets"):
            CEAT().compute(encoder_model, WordSets(*TERMS))

    def test_mapping_is_accepted(self, vector_sets):
        """A dict keyed by role aliases coerces to a container."""
        t1, t2, a1, a2 = TERMS
        ws = {"t1": t1, "t2": t2, "a1": a1, "a2": a2}
        with pytest.raises(ValueError, match="needs a model"):
            WEAT().compute(None, ws)  # coerced to WordSets, then needs a model

    def test_incomplete_mapping_names_missing_roles(self):
        with pytest.raises(ValueError, match="missing attribute_1"):
            WEAT().compute(None, {"t1": ["a"], "t2": ["b"]})


# --------------------------------------------------------------------------
# Legacy keyword API
# --------------------------------------------------------------------------
class TestLegacyApi:
    def test_legacy_terms_still_work_and_warn(self, encoder_model):
        t1, t2, a1, a2 = TERMS
        with pytest.warns(DeprecationWarning, match="deprecated"):
            legacy = WEAT(n_samples=100).compute(
                model=encoder_model, T1_terms=t1, T2_terms=t2, A_terms=a1, B_terms=a2
            )
        modern = WEAT(n_samples=100).compute(encoder_model, WordSets(*TERMS))
        assert legacy.score == modern.score

    def test_legacy_a1_a2_aliases(self, encoder_model):
        t1, t2, a1, a2 = TERMS
        with pytest.warns(DeprecationWarning):
            result = SEAT(n_samples=100).compute(
                model=encoder_model, T1_terms=t1, T2_terms=t2, A1_terms=a1, A2_terms=a2
            )
        assert isinstance(result.score, float)

    def test_legacy_vecs_path(self, vector_sets):
        with pytest.warns(DeprecationWarning, match="VectorSets"):
            result = WEAT(n_samples=100).compute(
                None,
                T1_vecs=vector_sets.target_1,
                T2_vecs=vector_sets.target_2,
                A_vecs=vector_sets.attribute_1,
                B_vecs=vector_sets.attribute_2,
            )
        assert isinstance(result.score, float)

    def test_legacy_contexts_path(self, encoder_model, context_sets):
        with pytest.warns(DeprecationWarning, match="ContextSets"):
            result = CEAT(sample_size=2, n_trials=3, seed=0).compute(
                model=encoder_model,
                T1_contexts=dict(context_sets.target_1),
                T2_contexts=dict(context_sets.target_2),
                A1_contexts=dict(context_sets.attribute_1),
                A2_contexts=dict(context_sets.attribute_2),
            )
        assert isinstance(result.score, float)

    def test_modern_path_does_not_warn(self, vector_sets):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            WEAT(n_samples=100).compute(None, vector_sets)
