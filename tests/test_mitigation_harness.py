"""The before/after harness, and the aggregations it refuses to perform."""

import json

import pytest

from fairlms.metrics import GroupPredictions, ScorePair
from fairlms.mitigation import ComparisonReport, MetricDelta, compare_before_after


class TestReporting:
    def test_runs_a_declared_metric_set_against_both_systems(self):
        report = compare_before_after(
            None,
            None,
            metrics={"accuracy_disparity": ScorePair([1.0, 0.0], [0.0, 1.0])},
        )
        assert len(report.fairness) == 1
        delta = report.fairness[0]
        assert delta.metric == "accuracy_disparity"
        assert delta.before is not None and delta.after is not None
        assert delta.delta == pytest.approx(0.0)

    def test_delta_is_after_minus_before(self):
        delta = MetricDelta(metric="m", bias_type="intrinsic", before=0.5, after=0.2)
        assert delta.delta == pytest.approx(-0.3)

    def test_a_failed_metric_is_recorded_not_dropped(self):
        # A partial comparison must be visibly partial.
        report = compare_before_after(
            None, None, metrics={"equal_opportunity_gap": "not valid evidence"}
        )
        delta = report.fairness[0]
        assert delta.error is not None
        assert delta.before is None and delta.delta is None

    def test_unknown_metric_names_are_refused_up_front(self):
        with pytest.raises(KeyError, match="unknown metric"):
            compare_before_after(None, None, metrics={"not_a_metric": []})


class TestSeparationOfConcerns:
    def _report(self):
        return compare_before_after(
            None,
            None,
            metrics={
                "accuracy_disparity": ScorePair([1.0, 0.0], [0.0, 1.0]),
                "equal_opportunity_gap": GroupPredictions(
                    [1, 1, 0], [1, 0, 0], ["A", "B", "A"]
                ),
            },
            utility={"accuracy": lambda model: 0.9},
        )

    def test_fairness_and_utility_are_reported_separately(self):
        report = self._report()
        payload = report.to_dict()
        assert "fairness" in payload and "utility" in payload
        assert [d.metric for d in report.utility] == ["accuracy"]
        # Utility never appears among the fairness deltas.
        assert "accuracy" not in [d.metric for d in report.fairness]

    def test_intrinsic_and_extrinsic_are_reported_separately(self):
        report = self._report()
        payload = report.to_dict()["fairness"]
        assert set(payload) == {"intrinsic", "extrinsic"}
        assert all(d.bias_type == "extrinsic" for d in report.extrinsic)
        assert all(d.bias_type == "intrinsic" for d in report.intrinsic)

    def test_there_is_no_composite_effectiveness_score(self):
        report = self._report()
        # Not float-convertible, and no aggregate anywhere in the payload.
        with pytest.raises(TypeError):
            float(report)
        # Checked against the data, not the prose: the explanatory note names
        # the thing it is refusing to compute.
        payload = report.to_dict()
        payload.pop("note")
        serialized = json.dumps(payload)
        for forbidden in ("overall", "composite", "effectiveness", "total_score"):
            assert forbidden not in serialized
        assert set(payload) == {"fairness", "utility", "provenance"}

    def test_the_report_says_why_it_does_not_aggregate(self):
        assert "no composite" in self._report().to_dict()["note"].lower()

    def test_report_serializes_deterministically(self):
        report = self._report()
        assert json.loads(report.to_json()) == json.loads(report.to_json())

    def test_an_empty_report_is_representable(self):
        assert ComparisonReport().to_dict()["utility"] == []
