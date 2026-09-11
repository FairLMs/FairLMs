"""Contract tests for the eight-slot ``b_constr`` construction vector.

Every test here is offline and deterministic: no network, no model download,
no randomness. The assertions are on the published contract -- slot presence
and order, applicability statuses, reason codes, serialized output and numeric
values -- never on private implementation details.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from fairlms.diagnostics import (
    BACKEND_CONSTRUCTION_SLOTS,
    CONSTRUCTION_BACKEND_REQUIREMENTS,
    CONSTRUCTION_SLOTS,
    LIGHTWEIGHT_CONSTRUCTION_SLOTS,
    SELF_IDENTIFICATION_FRAME,
    ComponentOverride,
    ComponentResult,
    DatasetAuditSpec,
    DatasetDiagnostic,
    DatasetEvidence,
    DiagnosticReport,
    DiagnosticStatus,
    FramePredicate,
    FramingDisparity,
    GroupedTexts,
    IdentityMaskConfig,
    InjectedFramePredicate,
    LengthDisparity,
    MinimalPairResidual,
    OptionItems,
    OptionLengthBias,
    OptionRoleContrast,
    PairedTexts,
    ReportStatus,
    TemplateGroups,
    TemplateImbalance,
    TokenizationMode,
    TokenizationRule,
    audit_construction,
    construction_vector,
)
from fairlms.diagnostics.construction import (
    BACKEND_BLOCKED_WARNING,
    CONSTRUCTION_VECTOR_WARNING,
    INJECTED_PREDICATE_WARNING,
    TOKENIZATION_DIVERGENCE_WARNING,
)

AXIS = "gender-reference"
TARGET = "synthetic-gender-reference-suite"
ROLES = ("stereotype", "anti_stereotype")
GROUPS = ("feminine", "masculine")

SURFACE_TOKENS = TokenizationRule(
    mode=TokenizationMode.REGEX, pattern=r"\b[\w_]+\b"
)


# --------------------------------------------------------------------------
# Local builders
# --------------------------------------------------------------------------


def _spec(**overrides):
    parameters = {
        "target_name": TARGET,
        "target_kind": "benchmark_dataset",
        "task_family": "free_text",
        "design_stance": "stress_test",
        "references": {},
        "requested_components": CONSTRUCTION_SLOTS,
    }
    parameters.update(overrides)
    return DatasetAuditSpec(**parameters)


def _paired(pairs=None, *, axis=AXIS):
    pairs = pairs or (
        ("mp-001", "he is a nurse", "she is a nurse"),
        ("mp-002", "he is a brilliant engineer", "she is an engineer"),
    )
    pair_ids: list[str] = []
    conditions: list[str] = []
    texts: list[str] = []
    for pair_id, stereotype_text, anti_text in pairs:
        pair_ids.extend((pair_id, pair_id))
        conditions.extend(ROLES)
        texts.extend((stereotype_text, anti_text))
    return PairedTexts(
        axis=axis,
        pair_ids=pair_ids,
        conditions=conditions,
        texts=texts,
        condition_roles=ROLES,
        pairing_basis="one declared gender-term substitution per pair",
        source="synthetic minimal-pair fixture",
    )


def _grouped(rows=None, *, declared_groups=GROUPS, axis=AXIS):
    rows = rows or (
        ("feminine", "as a nurse she worked"),
        ("feminine", "the clinic opened early"),
        ("masculine", "i am the engineer"),
        ("masculine", "the report went out"),
    )
    return GroupedTexts(
        axis=axis,
        groups=[group for group, _ in rows],
        texts=[text for _, text in rows],
        declared_groups=declared_groups,
        source="synthetic grouped-text fixture",
    )


def _unbalanced_grouped(*, declared_groups=GROUPS, axis=AXIS):
    """Eight three-token texts against two thirteen-token texts.

    The sample-weighted pooled mean length is 50 / 10 = 5.0, while the
    unweighted mean of the two group means is (3 + 13) / 2 = 8.0, so the two
    denominators give visibly different answers for the same gap of 10.
    """
    rows = [("feminine", f"short line {index}") for index in range(8)]
    long_text = " ".join(f"word{index}" for index in range(13))
    rows.extend([("masculine", long_text), ("masculine", long_text.upper())])
    return _grouped(tuple(rows), declared_groups=declared_groups, axis=axis)


def _options(rows=None, *, declared_roles=ROLES, axis=AXIS):
    rows = rows or (
        ("q1", "stereotype", "a short option"),
        ("q1", "anti_stereotype", "a considerably longer anti stereotype option"),
        ("q2", "stereotype", "brief"),
        ("q2", "anti_stereotype", "four token option here"),
    )
    return OptionItems(
        axis=axis,
        item_ids=[item_id for item_id, _, _ in rows],
        roles=[role for _, role, _ in rows],
        options=[option for _, _, option in rows],
        declared_roles=declared_roles,
        question_family="multiple_choice_stereotype_contrast",
        source="synthetic option fixture",
    )


def _templates(rows=None, *, declared_groups=GROUPS, axis=AXIS):
    rows = rows or (
        ("feminine", "tpl-a"),
        ("feminine", "tpl-a"),
        ("feminine", "tpl-b"),
        ("feminine", "tpl-c"),
        ("masculine", "tpl-a"),
        ("masculine", "tpl-a"),
    )
    return TemplateGroups(
        axis=axis,
        groups=[group for group, _ in rows],
        template_ids=[template_id for _, template_id in rows],
        declared_groups=declared_groups,
        template_identity_rule="exact normalized template string",
        source="synthetic template fixture",
    )


def _evidence(*, paired=None, grouped=None, options=None, templates=None):
    views: dict[str, dict] = {}
    if paired is not None:
        views["paired_texts"] = {paired.axis: paired}
    if grouped is not None:
        views["grouped_texts"] = {grouped.axis: grouped}
    if options is not None:
        views["option_items"] = {options.axis: options}
    if templates is not None:
        views["template_groups"] = {templates.axis: templates}
    return DatasetEvidence(target_name=TARGET, **views)


def _mask(terms=("he", "her", "him", "his", "she")):
    return IdentityMaskConfig(identity_terms=terms)


def _contrast():
    return OptionRoleContrast(
        stereotype_role="stereotype",
        anti_stereotype_role="anti_stereotype",
    )


def _full_diagnostics():
    return (
        MinimalPairResidual(identity_mask=_mask()),
        FramingDisparity(predicate=SELF_IDENTIFICATION_FRAME),
        OptionLengthBias(role_contrast=_contrast()),
    )


def _full_evidence():
    return _evidence(
        paired=_paired(),
        grouped=_grouped(),
        options=_options(),
        templates=_templates(),
    )


def _statuses(report):
    return {name: result.status.value for name, result in report.components.items()}


def _reason_codes(report):
    return {name: result.reason_code for name, result in report.components.items()}


class _ExplodingTemplateImbalance(TemplateImbalance):
    """A b_temp instance whose plan is ready but whose computation fails."""

    def compute(self, evidence, spec):  # noqa: D102 - behaviour is the point
        raise OverflowError("synthetic kernel failure")


class _UnimplementedEmbeddingSlot(DatasetDiagnostic):
    """A caller-supplied stand-in for a backend slot; must be refused."""

    name = "b_equiv"

    def plan(self, evidence, spec):  # pragma: no cover - never reached
        raise AssertionError("a backend slot diagnostic must never be planned.")

    def compute(self, evidence, spec):  # pragma: no cover - never reached
        raise AssertionError("a backend slot diagnostic must never run.")


# --------------------------------------------------------------------------
# The vector is always eight slots, in the declared order
# --------------------------------------------------------------------------


def test_report_always_carries_all_eight_slots_in_the_frozen_order():
    report = audit_construction(
        _full_evidence(), _spec(), axis=AXIS, diagnostics=_full_diagnostics()
    )

    assert CONSTRUCTION_SLOTS == (
        "b_min",
        "b_equiv",
        "b_gram",
        "b_diff_len",
        "b_diff_dep",
        "b_frame",
        "b_opt",
        "b_temp",
    )
    assert set(report.components) == set(CONSTRUCTION_SLOTS)
    assert len(report.components) == 8
    assert report.provenance["slot_order"] == CONSTRUCTION_SLOTS
    assert tuple(
        result.component for result in construction_vector(report)
    ) == CONSTRUCTION_SLOTS
    assert report.provenance["implemented_slots"] == LIGHTWEIGHT_CONSTRUCTION_SLOTS
    assert report.provenance["backend_slots"] == BACKEND_CONSTRUCTION_SLOTS

    payload = report.to_dict()
    assert set(payload["components"]) == set(CONSTRUCTION_SLOTS)
    # The serialized mapping is alphabetized by ``DiagnosticReport``; the slot
    # order lives in provenance and in ``construction_vector``, never in the
    # serialized key order.
    assert list(payload["components"]) == sorted(CONSTRUCTION_SLOTS)
    assert payload["provenance"]["slot_order"] == list(CONSTRUCTION_SLOTS)
    assert payload["provenance"]["entry_point"] == "audit_construction"
    assert payload["provenance"]["axis"] == AXIS
    assert payload["provenance"]["target_name"] == TARGET
    assert payload["provenance"]["views_used"] == {
        "b_diff_dep": "grouped_texts",
        "b_diff_len": "grouped_texts",
        "b_equiv": "paired_texts",
        "b_frame": "grouped_texts",
        "b_gram": "paired_texts",
        "b_min": "paired_texts",
        "b_opt": "option_items",
        "b_temp": "template_groups",
    }
    assert set(payload["provenance"]["diagnostics"]) == set(
        LIGHTWEIGHT_CONSTRUCTION_SLOTS
    )


def test_all_eight_slots_are_present_with_only_one_evidence_view():
    report = audit_construction(
        _evidence(grouped=_grouped()), _spec(), axis=AXIS
    )

    assert set(report.components) == set(CONSTRUCTION_SLOTS)
    assert tuple(
        result.component for result in construction_vector(report)
    ) == CONSTRUCTION_SLOTS
    assert _statuses(report)["b_opt"] == "not_applicable"
    assert _reason_codes(report)["b_opt"] == "evidence_view_not_supplied"


def test_all_eight_slots_are_present_when_no_slot_was_requested():
    report = audit_construction(
        _full_evidence(),
        _spec(requested_components=("b_rep",)),
        axis=AXIS,
        diagnostics=_full_diagnostics(),
    )

    assert set(report.components) == set(CONSTRUCTION_SLOTS)
    assert set(_statuses(report).values()) == {"not_applicable"}
    assert set(_reason_codes(report).values()) == {"component_not_requested"}
    assert all(result.value is None for result in report.components.values())
    assert report.status is ReportStatus.NOT_APPLICABLE


def test_construction_vector_refuses_a_report_that_is_missing_a_slot():
    partial = DiagnosticReport(
        spec=_spec(),
        components={
            "b_min": ComponentResult(
                component="b_min",
                status=DiagnosticStatus.NOT_APPLICABLE,
                reason_code="component_not_requested",
                reason="Not requested.",
            )
        },
    )

    with pytest.raises(KeyError) as excinfo:
        construction_vector(partial)

    message = str(excinfo.value)
    assert "missing construction slot" in message
    assert "b_temp" in message


# --------------------------------------------------------------------------
# Backend slots: blocked, named, and harmless to everything else
# --------------------------------------------------------------------------


def test_backend_slots_are_blocked_with_a_reason_naming_the_backend():
    report = audit_construction(
        _full_evidence(), _spec(), axis=AXIS, diagnostics=_full_diagnostics()
    )

    expected = {
        "b_equiv": (
            "embedding_backend_unavailable",
            "EmbeddingBackend",
            "sentence-embedding backend",
        ),
        "b_gram": (
            "grammar_backend_unavailable",
            "GrammarCheckerBackend",
            "grammar-checking backend",
        ),
        "b_diff_dep": (
            "dependency_parser_backend_unavailable",
            "DependencyParserBackend",
            "dependency-parser backend",
        ),
    }
    for slot, (reason_code, protocol, backend_phrase) in expected.items():
        result = report.components[slot]
        requirement = CONSTRUCTION_BACKEND_REQUIREMENTS[slot]
        assert result.status is DiagnosticStatus.BLOCKED
        assert result.value is None
        assert result.reason_code == reason_code
        assert protocol in result.reason
        assert backend_phrase in result.reason
        assert result.reason.startswith(f"{slot} requires a ")
        assert result.details["slot"] == slot
        assert result.details["required_backend"] == requirement["required_backend"]
        assert result.details["required_protocol"] == protocol
        assert result.details["availability"] == "not_implemented_in_this_release"
        assert result.details["milestone"] == "P2C-06"
        assert result.provenance["backend_requirement"] == dict(requirement)


def test_a_blocked_backend_slot_does_not_fail_the_report_or_the_other_slots():
    report = audit_construction(
        _full_evidence(), _spec(), axis=AXIS, diagnostics=_full_diagnostics()
    )

    assert report.status is ReportStatus.PARTIAL
    for slot in BACKEND_CONSTRUCTION_SLOTS:
        assert report.components[slot].status is DiagnosticStatus.BLOCKED
    for slot in LIGHTWEIGHT_CONSTRUCTION_SLOTS:
        result = report.components[slot]
        assert result.status is DiagnosticStatus.READY, slot
        assert isinstance(result.value, float)

    blocked_warning = BACKEND_BLOCKED_WARNING.format(
        slots="b_equiv, b_gram, b_diff_dep"
    )
    assert CONSTRUCTION_VECTOR_WARNING in report.warnings
    assert blocked_warning in report.warnings
    assert report.warnings[0] == CONSTRUCTION_VECTOR_WARNING
    assert len(set(report.warnings)) == len(report.warnings)


def test_a_backend_slot_reports_absent_geometry_before_it_names_a_backend():
    report = audit_construction(
        _evidence(grouped=_grouped()), _spec(), axis=AXIS
    )

    for slot in ("b_equiv", "b_gram"):
        result = report.components[slot]
        assert result.status is DiagnosticStatus.NOT_APPLICABLE
        assert result.reason_code == "evidence_view_not_supplied"
        assert result.value is None
        assert result.details["required_view"] == "paired_texts"
        assert "backend" not in (result.reason or "").lower()

    dependency = report.components["b_diff_dep"]
    assert dependency.status is DiagnosticStatus.BLOCKED
    assert dependency.reason_code == "dependency_parser_backend_unavailable"

    only_dependency = BACKEND_BLOCKED_WARNING.format(slots="b_diff_dep")
    assert only_dependency in report.warnings


def test_audit_construction_refuses_a_caller_supplied_backend_slot():
    with pytest.raises(ValueError) as excinfo:
        audit_construction(
            _full_evidence(),
            _spec(),
            axis=AXIS,
            diagnostics=(_UnimplementedEmbeddingSlot(),),
        )

    assert "no implementation in this release" in str(excinfo.value)
    assert "P2C-06" in str(excinfo.value)


def test_audit_construction_refuses_a_non_slot_or_duplicated_diagnostic():
    with pytest.raises(ValueError) as excinfo:
        audit_construction(
            _full_evidence(),
            _spec(),
            axis=AXIS,
            diagnostics=(LengthDisparity(), LengthDisparity()),
        )
    assert "duplicate component names" in str(excinfo.value)

    with pytest.raises(TypeError):
        audit_construction(
            _full_evidence(), _spec(), axis=AXIS, diagnostics=("b_min",)
        )


# --------------------------------------------------------------------------
# Mixed vectors, and the rule that a non-ready slot is never a zero
# --------------------------------------------------------------------------


def test_one_call_returns_ready_not_applicable_and_blocked_slots_together():
    evidence = _evidence(
        paired=_paired(),
        grouped=_unbalanced_grouped(),
        templates=_templates(
            rows=(("feminine", "tpl-a"), ("feminine", "tpl-b")),
        ),
    )
    report = audit_construction(
        evidence,
        _spec(),
        axis=AXIS,
        diagnostics=(MinimalPairResidual(identity_mask=_mask()),),
    )

    assert _statuses(report) == {
        "b_min": "ready",
        "b_diff_len": "ready",
        "b_frame": "blocked",
        "b_opt": "not_applicable",
        "b_temp": "blocked",
        "b_equiv": "blocked",
        "b_gram": "blocked",
        "b_diff_dep": "blocked",
    }
    assert _reason_codes(report)["b_frame"] == "missing_frame_predicate"
    assert _reason_codes(report)["b_opt"] == "evidence_view_not_supplied"
    assert _reason_codes(report)["b_temp"] == "empty_declared_group"
    assert report.status is ReportStatus.PARTIAL

    ready = {
        name
        for name, result in report.components.items()
        if result.status is DiagnosticStatus.READY
    }
    assert ready == {"b_min", "b_diff_len"}


def test_non_ready_slots_are_never_serialized_as_zero():
    evidence = _evidence(
        paired=_paired(),
        grouped=_unbalanced_grouped(),
        templates=_templates(
            rows=(("feminine", "tpl-a"), ("feminine", "tpl-b")),
        ),
    )
    report = audit_construction(
        evidence,
        _spec(),
        axis=AXIS,
        diagnostics=(MinimalPairResidual(identity_mask=_mask()),),
    )

    payload = report.to_dict()
    round_tripped = json.loads(report.to_json())
    assert round_tripped == payload

    non_ready = [
        name
        for name, result in report.components.items()
        if result.status is not DiagnosticStatus.READY
    ]
    assert len(non_ready) == 6
    for name in non_ready:
        for source in (payload, round_tripped):
            serialized = source["components"][name]
            assert serialized["status"] != "ready"
            assert serialized["value"] is None
            assert not isinstance(serialized["value"], (int, float))
            assert serialized["reason_code"] is not None
            assert '"value": null' in json.dumps(serialized, indent=1)

    for name in ("b_min", "b_diff_len"):
        assert isinstance(payload["components"][name]["value"], float)
        assert payload["components"][name]["reason_code"] is None


def test_component_result_refuses_a_numeric_sentinel_for_a_non_ready_slot():
    with pytest.raises(ValueError) as excinfo:
        ComponentResult(
            component="b_opt",
            status=DiagnosticStatus.NOT_APPLICABLE,
            value=0.0,
            reason_code="evidence_view_not_supplied",
            reason="No option evidence was supplied.",
        )

    assert "not a numeric sentinel" in str(excinfo.value)


def test_report_json_is_strict_and_deterministic():
    report = audit_construction(
        _full_evidence(), _spec(), axis=AXIS, diagnostics=_full_diagnostics()
    )

    first = report.to_json()
    second = report.to_json()
    assert first == second
    assert "NaN" not in first
    assert "Infinity" not in first
    assert json.loads(first)["schema_version"] == report.to_dict()["schema_version"]


def test_the_construction_report_publishes_no_aggregate_score():
    report = audit_construction(
        _full_evidence(), _spec(), axis=AXIS, diagnostics=_full_diagnostics()
    )

    payload = report.to_dict()
    assert set(payload) == {
        "schema_version",
        "status",
        "spec",
        "components",
        "warnings",
        "provenance",
    }
    for forbidden in ("total", "aggregate", "mean", "score", "b_constr"):
        assert forbidden not in payload["provenance"]
        assert forbidden not in payload["components"]

    with pytest.raises(TypeError):
        float(report)  # type: ignore[arg-type]


def test_an_applicability_override_can_never_manufacture_a_value():
    spec = _spec(
        component_overrides={
            "b_min": ComponentOverride(
                component="b_min",
                status="not_applicable",
                reason="This corpus has no minimal-pair design.",
                declared_reason_code="no_minimal_pair_design",
            )
        }
    )
    report = audit_construction(
        _full_evidence(), spec, axis=AXIS, diagnostics=_full_diagnostics()
    )

    result = report.components["b_min"]
    assert result.status is DiagnosticStatus.NOT_APPLICABLE
    assert result.value is None
    assert result.reason_code == "applicability_override"
    assert result.reason == "This corpus has no minimal-pair design."
    assert result.details["applicability_override"] is True
    assert result.details["declared_reason_code"] == "no_minimal_pair_design"
    assert result.details["override_source"] == "spec"
    assert report.components["b_diff_len"].status is DiagnosticStatus.READY


def test_a_failing_slot_is_reported_as_failed_without_killing_the_report():
    report = audit_construction(
        _full_evidence(),
        _spec(),
        axis=AXIS,
        diagnostics=_full_diagnostics() + (_ExplodingTemplateImbalance(),),
    )

    failed = report.components["b_temp"]
    assert failed.status is DiagnosticStatus.FAILED
    assert failed.value is None
    assert failed.reason_code == "computation_failed"
    assert failed.details["exception_type"] == "OverflowError"
    assert set(report.components) == set(CONSTRUCTION_SLOTS)
    assert report.components["b_min"].status is DiagnosticStatus.READY
    assert report.to_dict()["components"]["b_temp"]["value"] is None


# --------------------------------------------------------------------------
# b_min
# --------------------------------------------------------------------------


def test_b_min_measures_zero_residual_for_an_identity_only_substitution():
    evidence = _evidence(
        paired=_paired(
            (
                ("mp-001", "he is a nurse who works nights",
                 "she is a nurse who works nights"),
                ("mp-002", "his colleagues call him the janitor",
                 "her colleagues call her the janitor"),
            )
        )
    )
    report = audit_construction(
        evidence,
        _spec(),
        axis=AXIS,
        diagnostics=(MinimalPairResidual(identity_mask=_mask()),),
    )

    result = report.components["b_min"]
    assert result.status is DiagnosticStatus.READY
    assert result.value == 0.0
    assert result.details["pair_count"] == 2
    assert result.details["non_zero_ratio"] == 0.0
    assert result.details["maximum_normalized_edit_distance"] == 0.0
    assert result.details["identity_term_count"] == 5
    assert result.details["unit"] == "normalized_token_edit_distance"
    assert result.details["estimator"] == "uniform_mass_per_pair"
    assert result.details["pairing_basis"] == (
        "one declared gender-term substitution per pair"
    )
    mask_provenance = result.to_dict()["provenance"]["identity_mask"]
    assert mask_provenance["placeholder"] == "[ID]"
    assert mask_provenance["identity_terms"] == ["he", "her", "him", "his", "she"]


def test_b_min_grows_when_an_edit_survives_the_identity_mask():
    identity_only = _evidence(
        paired=_paired(
            (("mp-001", "he is a brilliant engineer", "she is a brilliant engineer"),)
        )
    )
    with_residue = _evidence(
        paired=_paired(
            (("mp-001", "he is a brilliant engineer", "she is an engineer"),)
        )
    )
    diagnostics = (MinimalPairResidual(identity_mask=_mask()),)

    clean = audit_construction(
        identity_only, _spec(), axis=AXIS, diagnostics=diagnostics
    ).components["b_min"]
    residual = audit_construction(
        with_residue, _spec(), axis=AXIS, diagnostics=diagnostics
    ).components["b_min"]

    assert clean.value == 0.0
    # "a brilliant" -> "an" is two token edits over the five-token longer side.
    assert residual.value == pytest.approx(0.4, rel=0, abs=1e-15)
    assert residual.value > clean.value
    assert residual.details["non_zero_ratio"] == 1.0


def test_b_min_is_blocked_without_a_declared_identity_mask():
    report = audit_construction(_evidence(paired=_paired()), _spec(), axis=AXIS)

    result = report.components["b_min"]
    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "missing_identity_mask"
    assert "never inferred" in result.reason
    assert result.to_dict()["value"] is None


def test_b_min_blocks_a_pair_that_masks_down_to_two_empty_token_sequences():
    evidence = _evidence(
        paired=_paired(
            (
                ("mp-001", "he is a nurse", "she is a nurse"),
                ("mp-002", "...", "!!!"),
            )
        )
    )
    report = audit_construction(
        evidence,
        _spec(),
        axis=AXIS,
        diagnostics=(MinimalPairResidual(identity_mask=_mask()),),
    )

    result = report.components["b_min"]
    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "degenerate_masked_pair"
    assert list(result.details["degenerate_pair_ids"]) == ["mp-002"]
    assert result.details["degenerate_pair_count"] == 1


# --------------------------------------------------------------------------
# b_diff_len
# --------------------------------------------------------------------------


def test_b_diff_len_normalizes_by_the_sample_weighted_pooled_mean():
    report = audit_construction(
        _evidence(grouped=_unbalanced_grouped()), _spec(), axis=AXIS
    )

    result = report.components["b_diff_len"]
    details = result.to_dict()["details"]
    assert result.status is DiagnosticStatus.READY

    group_means = details["group_mean_lengths"]
    assert group_means == {"feminine": 3.0, "masculine": 13.0}
    assert details["group_sample_counts"] == {"feminine": 8, "masculine": 2}
    assert details["group_token_totals"] == {"feminine": 24, "masculine": 26}
    assert details["max_absolute_gap"] == 10.0
    assert details["sample_count"] == 10
    assert details["total_token_count"] == 50

    sample_weighted = details["total_token_count"] / details["sample_count"]
    unweighted = math.fsum(group_means.values()) / len(group_means)
    assert sample_weighted == 5.0
    assert unweighted == 8.0
    assert sample_weighted != unweighted

    assert details["pooled_mean_length"] == sample_weighted
    assert details["denominator_rule"] == "sample_weighted_pooled_mean"
    assert result.value == pytest.approx(10.0 / sample_weighted, rel=0, abs=1e-15)
    assert result.value == 2.0
    assert result.value != pytest.approx(10.0 / unweighted, rel=0, abs=1e-12)
    assert details["widest_pair"] == ["feminine", "masculine"]
    assert details["unit"] == "ratio_to_pooled_mean_token_count"


def test_b_diff_len_blocks_a_declared_group_with_no_texts():
    grouped = _grouped(
        rows=(("feminine", "one short line"), ("feminine", "another short line")),
    )
    report = audit_construction(_evidence(grouped=grouped), _spec(), axis=AXIS)

    result = report.components["b_diff_len"]
    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "empty_declared_group"
    assert list(result.details["empty_groups"]) == ["masculine"]
    assert list(result.details["support"]) == ["feminine", "masculine"]
    assert "rather than reported as zero" in result.reason


def test_b_diff_len_ready_always_carries_the_tokenization_divergence_warning():
    report = audit_construction(
        _evidence(grouped=_unbalanced_grouped()), _spec(), axis=AXIS
    )

    assert report.components["b_diff_len"].status is DiagnosticStatus.READY
    assert TOKENIZATION_DIVERGENCE_WARNING in report.warnings
    assert report.components["b_diff_len"].details["paper_alignment"] == (
        "generalized_surface_tokenization"
    )


# --------------------------------------------------------------------------
# b_opt
# --------------------------------------------------------------------------


def test_b_opt_refuses_to_run_without_an_explicit_role_contrast():
    # Option roles must be supplied explicitly. With option geometry present
    # but no declared contrast the slot is blocked; with no option view at all
    # it is not applicable (the next test). Neither path produces a value, and
    # neither reads a role from option position.
    report = audit_construction(_evidence(options=_options()), _spec(), axis=AXIS)

    result = report.components["b_opt"]
    assert result.status is not DiagnosticStatus.READY
    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "missing_option_role_contrast"
    assert "never inferred from position" in result.reason
    assert result.to_dict()["value"] is None


def test_b_opt_is_not_applicable_without_any_option_evidence():
    report = audit_construction(_evidence(grouped=_grouped()), _spec(), axis=AXIS)

    result = report.components["b_opt"]
    assert result.status is DiagnosticStatus.NOT_APPLICABLE
    assert result.value is None
    assert result.reason_code == "evidence_view_not_supplied"
    assert result.details["required_view"] == "option_items"


def test_b_opt_reads_declared_roles_and_never_option_position():
    rows = (
        ("q1", "stereotype", "one two three four"),
        ("q1", "anti_stereotype", "solo"),
        ("q2", "anti_stereotype", "one two three four five"),
        ("q2", "stereotype", "one two"),
    )
    shuffled = (rows[2], rows[0], rows[3], rows[1])

    # A positional reading of the supplied rows -- first row minus second row
    # within each item -- would answer +3.0 on this fixture.
    positional = math.fsum(
        (
            len(rows[0][2].split()) - len(rows[1][2].split()),
            len(rows[2][2].split()) - len(rows[3][2].split()),
        )
    ) / 2
    assert positional == 3.0

    diagnostics = (OptionLengthBias(role_contrast=_contrast()),)
    first = audit_construction(
        _evidence(options=_options(rows)), _spec(), axis=AXIS, diagnostics=diagnostics
    ).components["b_opt"]
    second = audit_construction(
        _evidence(options=_options(shuffled)),
        _spec(),
        axis=AXIS,
        diagnostics=diagnostics,
    ).components["b_opt"]

    # stereotype minus anti_stereotype is (4 - 1) + (2 - 5) = 0 over two items.
    assert first.status is DiagnosticStatus.READY
    assert first.value == 0.0
    assert first.value != positional
    assert second.value == first.value
    assert first.details["role_orientation"] == "stereotype_minus_anti_stereotype"
    assert first.details["stereotype_role"] == "stereotype"
    assert first.details["anti_stereotype_role"] == "anti_stereotype"
    assert first.details["item_count"] == 2


def test_b_opt_reports_a_signed_negative_value_that_survives_serialization():
    report = audit_construction(
        _evidence(options=_options()),
        _spec(),
        axis=AXIS,
        diagnostics=(OptionLengthBias(role_contrast=_contrast()),),
    )

    result = report.components["b_opt"]
    # (3 - 6) + (1 - 4) = -6 over two items.
    assert result.status is DiagnosticStatus.READY
    assert result.value == -3.0
    assert result.details["directionality"] == "signed"
    assert result.details["mean_signed_length_difference"] == -3.0
    assert result.details["anti_stereotype_longer_ratio"] == 1.0
    assert result.details["stereotype_longer_ratio"] == 0.0

    round_tripped = json.loads(report.to_json())
    assert round_tripped["components"]["b_opt"]["value"] == -3.0


def test_b_opt_blocks_when_the_contrast_names_an_undeclared_role():
    contrast = OptionRoleContrast(
        stereotype_role="pro_stereotype",
        anti_stereotype_role="anti_stereotype",
    )
    report = audit_construction(
        _evidence(options=_options()),
        _spec(),
        axis=AXIS,
        diagnostics=(OptionLengthBias(role_contrast=contrast),),
    )

    result = report.components["b_opt"]
    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "option_role_not_declared"
    assert list(result.details["missing_roles"]) == ["pro_stereotype"]
    assert list(result.details["declared_roles"]) == list(ROLES)


def test_b_opt_refuses_an_item_that_lacks_one_of_the_contrasted_roles():
    rows = (
        ("q1", "stereotype", "one two three"),
        ("q1", "anti_stereotype", "one two"),
        ("q2", "stereotype", "only the stereotype row exists"),
    )
    report = audit_construction(
        _evidence(options=_options(rows)),
        _spec(),
        axis=AXIS,
        diagnostics=(OptionLengthBias(role_contrast=_contrast()),),
    )

    result = report.components["b_opt"]
    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "incomplete_option_contrast"
    assert list(result.details["incomplete_item_ids"]) == ["q2"]
    assert result.details["incomplete_item_count"] == 1


# --------------------------------------------------------------------------
# b_frame
# --------------------------------------------------------------------------


def test_b_frame_is_blocked_without_a_predicate_and_ready_with_one():
    grouped = _grouped(
        rows=(
            ("feminine", "as a nurse she stayed late"),
            ("feminine", "the clinic opened early"),
            ("feminine", "the ward was quiet"),
            ("feminine", "records were filed"),
            ("masculine", "i am the engineer on call"),
            ("masculine", "the report went out"),
        )
    )
    evidence = _evidence(grouped=grouped)

    blocked = audit_construction(evidence, _spec(), axis=AXIS).components["b_frame"]
    assert blocked.status is DiagnosticStatus.BLOCKED
    assert blocked.value is None
    assert blocked.reason_code == "missing_frame_predicate"
    assert "the predicate definition is the estimand" in blocked.reason

    ready = audit_construction(
        evidence,
        _spec(),
        axis=AXIS,
        diagnostics=(FramingDisparity(predicate=SELF_IDENTIFICATION_FRAME),),
    ).components["b_frame"]
    details = ready.to_dict()["details"]
    assert ready.status is DiagnosticStatus.READY
    # feminine 1/4 = 0.25 against masculine 1/2 = 0.5.
    assert ready.value == 0.25
    assert details["group_frame_counts"] == {"feminine": 1, "masculine": 1}
    assert details["group_frame_rates"] == {"feminine": 0.25, "masculine": 0.5}
    assert details["frame_name"] == "self_identification"
    assert details["predicate_kind"] == "declared_patterns"
    assert details["replayable"] is True
    assert details["paper_alignment"] == "paper_exact"
    assert details["unit"] == "proportion"
    assert details["widest_pair"] == ["feminine", "masculine"]
    predicate_provenance = ready.to_dict()["provenance"]["frame_predicate"]
    assert predicate_provenance["patterns"] == ["i am", "i'm", "as a", "as an"]
    assert predicate_provenance["match_mode"] == "substring"


def test_b_frame_accepts_a_declared_regex_predicate():
    predicate = FramePredicate(
        frame_name="explicit_self_reference",
        definition="The text opens with a first-person clause.",
        patterns=(r"^i\s",),
        match_mode="regex",
    )
    grouped = _grouped(
        rows=(
            ("feminine", "i lead the ward"),
            ("feminine", "the clinic opened early"),
            ("masculine", "the report went out"),
            ("masculine", "the engineer signed off"),
        )
    )
    report = audit_construction(
        _evidence(grouped=grouped),
        _spec(),
        axis=AXIS,
        diagnostics=(FramingDisparity(predicate=predicate),),
    )

    result = report.components["b_frame"]
    assert result.status is DiagnosticStatus.READY
    assert result.value == 0.5
    assert result.details["paper_alignment"] == "generalized_frame_predicate"
    assert result.details["replayable"] is True
    assert INJECTED_PREDICATE_WARNING not in report.warnings


def test_a_mis_cased_substring_pattern_is_refused_rather_than_matching_nothing():
    """``case_fold`` must mean the same thing in both match modes.

    ``FramePredicate.matches`` folds the *haystack* only, so under substring
    matching a declared pattern carrying an upper-case character could never
    fire: every group's frame rate would be 0/n and the slot would publish a
    ready ``0.0`` -- with ``replayable: true`` and a predicate digest -- from
    evidence it never actually examined. The predicate is the estimand, so a
    mis-cased declaration is refused at construction rather than silently
    measured as an absence of the frame.
    """

    with pytest.raises(ValueError, match="lower-case"):
        FramePredicate(
            frame_name="self_identification",
            definition="First-person self identification.",
            patterns=("As a", "I am"),
            match_mode="substring",
            case_fold=True,
        )

    # Both escape hatches keep working, and both really match the same text.
    case_sensitive = FramePredicate(
        frame_name="self_identification",
        definition="First-person self identification.",
        patterns=("As a", "I am"),
        match_mode="substring",
        case_fold=False,
    )
    folded_regex = FramePredicate(
        frame_name="self_identification",
        definition="First-person self identification.",
        patterns=("As a", "I am"),
        match_mode="regex",
        case_fold=True,
    )
    assert case_sensitive.matches("As a nurse I work hard") is True
    assert folded_regex.matches("As a nurse I work hard") is True
    assert folded_regex.matches("as a nurse i work hard") is True

    # The accepted lower-case declaration measures what the refused one would
    # have reported as a successful zero.
    grouped = _grouped(
        rows=(
            ("feminine", "As a nurse I work hard"),
            ("feminine", "I am tired"),
            ("masculine", "Ordinary text"),
            ("masculine", "More text"),
        )
    )
    result = audit_construction(
        _evidence(grouped=grouped),
        _spec(),
        axis=AXIS,
        diagnostics=(
            FramingDisparity(
                predicate=FramePredicate(
                    frame_name="self_identification",
                    definition="First-person self identification.",
                    patterns=("as a", "i am"),
                    match_mode="substring",
                    case_fold=True,
                )
            ),
        ),
    ).components["b_frame"]

    assert result.status is DiagnosticStatus.READY
    assert result.value == 1.0
    assert result.details["group_frame_counts"] == {"feminine": 2, "masculine": 0}


def test_b_frame_with_an_injected_predicate_warns_and_records_replayable_false():
    predicate = InjectedFramePredicate(
        frame_name="first_person_opening",
        definition="A caller-owned callable that flags first-person openings.",
        predicate_id="first-person-opening",
        predicate_version="1.0.0",
        predicate=lambda text: text.startswith("i "),
    )
    grouped = _grouped(
        rows=(
            ("feminine", "i lead the ward"),
            ("feminine", "the clinic opened early"),
            ("masculine", "the report went out"),
            ("masculine", "the engineer signed off"),
        )
    )
    report = audit_construction(
        _evidence(grouped=grouped),
        _spec(),
        axis=AXIS,
        diagnostics=(FramingDisparity(predicate=predicate),),
    )

    result = report.components["b_frame"]
    serialized = json.loads(report.to_json())["components"]["b_frame"]
    assert result.status is DiagnosticStatus.READY
    assert result.value == 0.5
    assert serialized["details"]["replayable"] is False
    assert serialized["details"]["predicate_kind"] == "injected_callable"
    assert "paper_alignment" not in serialized["details"]
    assert serialized["provenance"]["frame_predicate"]["predicate_id"] == (
        "first-person-opening"
    )
    assert "predicate" not in serialized["provenance"]["frame_predicate"]
    assert INJECTED_PREDICATE_WARNING in report.warnings
    assert any("not replayable" in item for item in result.assumptions)


def test_b_frame_reports_a_failed_slot_when_an_injected_predicate_misbehaves():
    predicate = InjectedFramePredicate(
        frame_name="broken",
        definition="A callable that returns a non-boolean.",
        predicate_id="broken",
        predicate_version="0.0.1",
        predicate=lambda text: len(text),
    )
    report = audit_construction(
        _evidence(grouped=_grouped()),
        _spec(),
        axis=AXIS,
        diagnostics=(FramingDisparity(predicate=predicate),),
    )

    result = report.components["b_frame"]
    assert result.status is DiagnosticStatus.FAILED
    assert result.value is None
    assert result.reason_code == "frame_predicate_failed"
    assert result.details["exception_type"] == "TypeError"
    assert result.details["failed_text_index"] == 0


def test_b_frame_blocks_a_declared_group_with_no_texts():
    grouped = _grouped(
        rows=(("feminine", "as a nurse she stayed"), ("feminine", "quiet ward")),
    )
    report = audit_construction(
        _evidence(grouped=grouped),
        _spec(),
        axis=AXIS,
        diagnostics=(FramingDisparity(predicate=SELF_IDENTIFICATION_FRAME),),
    )

    result = report.components["b_frame"]
    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "empty_declared_group"
    assert list(result.details["empty_groups"]) == ["masculine"]


# --------------------------------------------------------------------------
# b_temp
# --------------------------------------------------------------------------


def test_b_temp_reports_both_the_imbalance_and_the_coverage_ratio():
    report = audit_construction(
        _evidence(templates=_templates()), _spec(), axis=AXIS
    )

    result = report.components["b_temp"]
    details = result.to_dict()["details"]
    assert result.status is DiagnosticStatus.READY
    # feminine contributes three unique templates, masculine one.
    assert result.value == 2.0
    assert details["group_unique_template_counts"] == {"feminine": 3, "masculine": 1}
    assert details["group_instance_counts"] == {"feminine": 4, "masculine": 2}
    assert details["max_unique_count"] == 3
    assert details["min_unique_count"] == 1
    assert details["coverage_ratio"] == 3.0
    assert details["coverage_denominator_group"] == "masculine"
    assert details["coverage_ratio_definition"] == (
        "max_unique / min_unique_at_or_above_ratio_min_count"
    )
    assert details["group_duplication_rates"] == {"feminine": 0.25, "masculine": 0.5}
    assert details["ratio_min_count"] == 1
    assert details["template_identity_rule"] == "exact normalized template string"
    assert details["unit"] == "unique_template_count_difference"


def test_b_temp_blocks_a_declared_group_with_no_templates():
    templates = _templates(
        rows=(("feminine", "tpl-a"), ("feminine", "tpl-b"), ("feminine", "tpl-c")),
    )
    assert templates.group_instance_counts["masculine"] == 0
    assert templates.group_unique_template_counts["masculine"] == 0

    report = audit_construction(
        _evidence(templates=templates), _spec(), axis=AXIS
    )

    result = report.components["b_temp"]
    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "empty_declared_group"
    # The group neither vanishes nor turns into a successful zero.
    assert list(result.details["empty_groups"]) == ["masculine"]
    assert list(result.details["support"]) == ["feminine", "masculine"]
    assert "template instantiations" in result.reason
    assert report.to_dict()["components"]["b_temp"]["value"] is None


def test_b_temp_blocks_when_no_group_reaches_the_coverage_denominator():
    report = audit_construction(
        _evidence(templates=_templates()),
        _spec(),
        axis=AXIS,
        diagnostics=(TemplateImbalance(ratio_min_count=4),),
    )

    result = report.components["b_temp"]
    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "template_coverage_denominator_unavailable"
    assert result.details["ratio_min_count"] == 4
    assert "ratio_min_count=4" in result.reason


def test_b_temp_keeps_a_defined_coverage_ratio_at_a_raised_threshold():
    report = audit_construction(
        _evidence(templates=_templates()),
        _spec(),
        axis=AXIS,
        diagnostics=(TemplateImbalance(ratio_min_count=3),),
    )

    result = report.components["b_temp"]
    assert result.status is DiagnosticStatus.READY
    assert result.value == 2.0
    assert result.details["coverage_ratio"] == 1.0
    assert result.details["coverage_denominator_group"] == "feminine"


def test_no_slot_value_depends_on_the_dataset_name_or_task_family():
    evidence = _full_evidence()
    diagnostics = _full_diagnostics()

    first = audit_construction(
        evidence, _spec(), axis=AXIS, diagnostics=diagnostics
    )
    second = audit_construction(
        evidence,
        _spec(target_name="a-completely-different-benchmark", task_family="qa"),
        axis=AXIS,
        diagnostics=diagnostics,
    )

    assert _statuses(first) == _statuses(second)
    assert _reason_codes(first) == _reason_codes(second)
    assert {name: result.value for name, result in first.components.items()} == {
        name: result.value for name, result in second.components.items()
    }
    assert first.warnings == second.warnings


# --------------------------------------------------------------------------
# Golden fixture
# --------------------------------------------------------------------------


def _golden_fixture_path():
    return (
        Path(__file__).parent
        / "data"
        / "golden"
        / "diagnostics"
        / "b_constr"
        / "synthetic_gender_reference_mixed_v1.json"
    )


def test_golden_mixed_construction_vector_fixture_is_portable_and_replays():
    fixture_text = _golden_fixture_path().read_text(encoding="utf-8")
    assert "/Users/" not in fixture_text
    assert "Downloads" not in fixture_text

    fixture = json.loads(fixture_text)
    assert fixture["fixture_id"] == "synthetic_gender_reference_mixed_v1"
    assert fixture["oracle"]["replayable_in_ci"] is True
    assert fixture["expected"]["slot_order"] == list(CONSTRUCTION_SLOTS)
    axis = fixture["axis"]

    paired_rows = fixture["evidence"]["paired_texts"]
    paired = PairedTexts(
        axis=axis,
        pair_ids=[row[0] for row in paired_rows["rows"]],
        conditions=[row[1] for row in paired_rows["rows"]],
        texts=[row[2] for row in paired_rows["rows"]],
        condition_roles=paired_rows["condition_roles"],
        pairing_basis=paired_rows["pairing_basis"],
        source=paired_rows["source"],
    )
    grouped_rows = fixture["evidence"]["grouped_texts"]
    grouped = GroupedTexts(
        axis=axis,
        groups=[row[0] for row in grouped_rows["rows"]],
        texts=[row[1] for row in grouped_rows["rows"]],
        declared_groups=grouped_rows["declared_groups"],
        source=grouped_rows["source"],
    )
    template_rows = fixture["evidence"]["template_groups"]
    templates = TemplateGroups(
        axis=axis,
        groups=[row[0] for row in template_rows["rows"]],
        template_ids=[row[1] for row in template_rows["rows"]],
        declared_groups=template_rows["declared_groups"],
        template_identity_rule=template_rows["template_identity_rule"],
        source=template_rows["source"],
    )
    assert fixture["evidence"]["option_items"] is None

    evidence = DatasetEvidence(
        target_name=fixture["dataset"],
        paired_texts={axis: paired},
        grouped_texts={axis: grouped},
        template_groups={axis: templates},
    )
    spec = DatasetAuditSpec(
        target_name=fixture["dataset"],
        target_kind="benchmark_dataset",
        task_family="free_text",
        design_stance=fixture["design_stance"],
        references={},
        requested_components=CONSTRUCTION_SLOTS,
        protected_axes=(axis,),
    )
    report = audit_construction(
        evidence,
        spec,
        axis=axis,
        diagnostics=(
            MinimalPairResidual(
                identity_mask=IdentityMaskConfig(
                    identity_terms=fixture["parameters"]["identity_terms"]
                )
            ),
            FramingDisparity(predicate=SELF_IDENTIFICATION_FRAME),
            TemplateImbalance(
                ratio_min_count=fixture["parameters"]["ratio_min_count"]
            ),
        ),
    )

    payload = report.to_dict()
    assert payload["status"] == fixture["expected"]["report_status"]
    assert payload["provenance"]["slot_order"] == fixture["expected"]["slot_order"]
    assert len(payload["warnings"]) == fixture["expected"]["warning_count"]
    assert set(payload["components"]) == set(fixture["expected"]["components"])

    for slot in fixture["expected"]["slot_order"]:
        expected = fixture["expected"]["components"][slot]
        serialized = payload["components"][slot]
        assert serialized["status"] == expected["status"], slot
        assert serialized["reason_code"] == expected["reason_code"], slot
        if expected["value"] is None:
            assert serialized["value"] is None, slot
        else:
            assert serialized["value"] == pytest.approx(
                expected["value"], rel=0, abs=1e-15
            ), slot
        for key, value in expected["details"].items():
            assert serialized["details"][key] == value, (slot, key)

    blocked_warning = BACKEND_BLOCKED_WARNING.format(
        slots=", ".join(fixture["expected"]["backend_blocked_slots"])
    )
    assert blocked_warning in payload["warnings"]
    assert CONSTRUCTION_VECTOR_WARNING in payload["warnings"]
    assert TOKENIZATION_DIVERGENCE_WARNING in payload["warnings"]

    # Independent re-derivation of every ready value from the frozen inputs.
    b_min = fixture["expected"]["components"]["b_min"]
    per_pair = b_min["cross_check"]["per_pair_normalized_edit_distances"]
    assert math.fsum(per_pair) / len(per_pair) == b_min["value"]

    declared = grouped_rows["declared_groups"]
    token_totals: dict[str, int] = {group: 0 for group in declared}
    sample_counts: dict[str, int] = {group: 0 for group in declared}
    for group, text in grouped_rows["rows"]:
        token_totals[group] += len(SURFACE_TOKENS.tokenize(text))
        sample_counts[group] += 1
    means = {
        group: token_totals[group] / sample_counts[group] for group in token_totals
    }
    gap = max(means.values()) - min(means.values())
    weighted = math.fsum(token_totals.values()) / math.fsum(sample_counts.values())
    unweighted = math.fsum(means.values()) / len(means)
    diff_len = fixture["expected"]["components"]["b_diff_len"]
    assert gap / weighted == diff_len["value"]
    assert unweighted == diff_len["cross_check"]["unweighted_mean_of_group_means"]
    assert gap / unweighted == diff_len["cross_check"][
        "value_under_unweighted_denominator"
    ]
    assert gap / unweighted != gap / weighted

    frame_hits: dict[str, int] = {group: 0 for group in sample_counts}
    for group, text in grouped_rows["rows"]:
        lowered = text.lower()
        if any(
            pattern in lowered for pattern in fixture["parameters"]["frame_patterns"]
        ):
            frame_hits[group] += 1
    rates = {group: frame_hits[group] / sample_counts[group] for group in frame_hits}
    assert max(rates.values()) - min(rates.values()) == (
        fixture["expected"]["components"]["b_frame"]["value"]
    )

    unique_templates: dict[str, set[str]] = {
        group: set() for group in template_rows["declared_groups"]
    }
    for group, template_id in template_rows["rows"]:
        unique_templates[group].add(template_id)
    counts = {group: len(ids) for group, ids in unique_templates.items()}
    b_temp = fixture["expected"]["components"]["b_temp"]
    assert float(max(counts.values()) - min(counts.values())) == b_temp["value"]
    assert max(counts.values()) / min(counts.values()) == (
        b_temp["details"]["coverage_ratio"]
    )

    # A not-applicable or blocked slot is null in the fixture and in the report.
    for slot, expected in fixture["expected"]["components"].items():
        if expected["status"] != "ready":
            assert expected["value"] is None, slot
            assert payload["components"][slot]["value"] is None, slot


# --- regression: a blocked slot must not advertise something that does not exist


def test_backend_protocols_are_real_types():
    """The refusal metadata names a protocol; that name must resolve.

    A blocked slot is an extension point only if a caller can see what to
    implement. Citing `EmbeddingBackend` while defining no such type made the
    reason metadata unactionable.
    """
    # Imported from the module, not the package surface: D032 keeps
    # unimplemented machinery off the public API, so these stay internal until
    # P2C-06 gives them an injection point.
    from fairlms.diagnostics import CONSTRUCTION_BACKEND_REQUIREMENTS
    from fairlms.diagnostics.construction import (
        DependencyParserBackend,
        EmbeddingBackend,
        GrammarCheckerBackend,
    )

    named = {EmbeddingBackend, GrammarCheckerBackend, DependencyParserBackend}
    by_name = {cls.__name__: cls for cls in named}
    for slot, requirement in CONSTRUCTION_BACKEND_REQUIREMENTS.items():
        assert requirement["required_protocol"] in by_name, slot


def test_blocked_slots_do_not_advertise_an_install_that_does_not_exist():
    """`required_extra` pointed at fairlms[construction-backends], which is not
    declared in pyproject.toml. Telling a user to install a nonexistent extra is
    worse than telling them the backend is not implemented yet."""
    import tomllib
    from pathlib import Path

    from fairlms.diagnostics import CONSTRUCTION_BACKEND_REQUIREMENTS

    root = Path(__file__).resolve().parents[1]
    declared = set(
        tomllib.loads((root / "pyproject.toml").read_text())
        .get("project", {})
        .get("optional-dependencies", {})
    )
    for slot, requirement in CONSTRUCTION_BACKEND_REQUIREMENTS.items():
        extra = requirement.get("required_extra")
        if extra is not None:
            name = extra.partition("[")[2].rstrip("]")
            assert name in declared, f"{slot} advertises undeclared extra {extra!r}"
        assert requirement["availability"] == "not_implemented_in_this_release"
