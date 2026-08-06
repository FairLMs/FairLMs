"""Typed evidence and explicit schema adapters for dataset diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence

from fairllms.diagnostics._utils import (
    freeze_json_mapping,
    normalize_string_sequence,
    require_nonempty_string,
    thaw_json,
)

if TYPE_CHECKING:  # pragma: no cover - imported only by type checkers
    import pandas as pd


def _validate_support(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError("support must be a sequence, not a bare string.")
    try:
        support = tuple(value)
    except TypeError as exc:
        raise TypeError("support must be a sequence of category labels.") from exc
    if len(support) < 2:
        raise ValueError("support must contain at least two categories.")
    for label in support:
        require_nonempty_string(label, "support label")
    if len(set(support)) != len(support):
        raise ValueError("support must not contain duplicate category labels.")
    return tuple(sorted(support))


def _adapter_provenance(
    provenance: Optional[Mapping[str, Any]],
    *,
    adapter: str,
    group_field: str,
    score_field: Optional[str] = None,
    value_map: Optional[Sequence[Mapping[str, Any]]],
) -> Mapping[str, Any]:
    if provenance is None:
        merged = {}
    elif isinstance(provenance, Mapping):
        merged = dict(provenance)
    else:
        raise TypeError(
            f"provenance must be a mapping, got {type(provenance).__name__}."
        )
    reserved = {"adapter", "field_mapping", "value_map", "value_map_used"}
    overlap = reserved.intersection(merged)
    if overlap:
        raise ValueError(
            "provenance uses reserved adapter key(s): " + ", ".join(sorted(overlap))
        )
    field_mapping = {"group": group_field}
    if score_field is not None:
        field_mapping["score"] = score_field
    return {
        **merged,
        "adapter": adapter,
        "field_mapping": field_mapping,
        "value_map": value_map,
        "value_map_used": value_map is not None,
    }


def _paired_adapter_provenance(
    provenance: Optional[Mapping[str, Any]],
    *,
    adapter: str,
    pair_id_field: str,
    condition_field: str,
    score_field: str,
    condition_map: Optional[Sequence[Mapping[str, Any]]],
) -> Mapping[str, Any]:
    if provenance is None:
        merged = {}
    elif isinstance(provenance, Mapping):
        merged = dict(provenance)
    else:
        raise TypeError(
            f"provenance must be a mapping, got {type(provenance).__name__}."
        )
    reserved = {
        "adapter",
        "field_mapping",
        "condition_map",
        "condition_map_used",
    }
    overlap = reserved.intersection(merged)
    if overlap:
        raise ValueError(
            "provenance uses reserved paired-adapter key(s): "
            + ", ".join(sorted(overlap))
        )
    return {
        **merged,
        "adapter": adapter,
        "field_mapping": {
            "pair_id": pair_id_field,
            "condition": condition_field,
            "score": score_field,
        },
        "condition_map": condition_map,
        "condition_map_used": condition_map is not None,
    }


def _portable_raw_key(value: Any, *, path: str) -> tuple[Any, tuple[int, str]]:
    """Return a portable JSON scalar and a deterministic heterogeneous sort key."""
    if isinstance(value, Enum):
        raise TypeError(
            f"{path} must be a JSON scalar (null, boolean, string, or finite "
            "number), not an enum."
        )
    if value is None:
        return None, (0, "")
    if isinstance(value, bool):
        return value, (1, "1" if value else "0")
    if isinstance(value, Integral):
        normalized = int(value)
        return normalized, (2, str(normalized))
    if isinstance(value, Real):
        try:
            normalized = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path} must be representable as a finite float."
            ) from exc
        if not math.isfinite(normalized):
            raise ValueError(f"{path} must not be NaN or infinity.")
        return normalized, (3, normalized.hex())
    if isinstance(value, str):
        return value, (4, value)
    raise TypeError(
        f"{path} must be a JSON scalar (null, boolean, string, or finite "
        f"number), got {type(value).__name__}."
    )


def _prepare_value_map(
    value_map: Optional[Mapping[Any, str]],
    *,
    support: Sequence[str],
) -> tuple[Optional[Mapping[Any, str]], Optional[tuple[Mapping[str, Any], ...]]]:
    """Validate and defensively copy a raw-to-canonical category mapping."""
    if value_map is None:
        return None, None
    if not isinstance(value_map, Mapping):
        raise TypeError("value_map must be a mapping when provided.")

    lookup: dict[tuple[int, str], str] = {}
    serialized_entries: list[tuple[tuple[int, str], Mapping[str, Any]]] = []
    for raw, canonical in value_map.items():
        portable_raw, sort_key = _portable_raw_key(raw, path="value_map raw key")
        if sort_key in lookup:
            raise ValueError(
                "value_map contains raw keys that collide after JSON-scalar "
                f"normalization: {raw!r}."
            )
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError(
                f"value_map output for raw key {raw!r} must be a non-empty " "string."
            )
        if canonical not in support:
            raise ValueError(
                f"value_map output {canonical!r} for raw key {raw!r} is outside "
                f"the explicit support {list(support)!r}."
            )
        lookup[sort_key] = canonical
        serialized_entries.append(
            (sort_key, {"raw": portable_raw, "canonical": canonical})
        )

    serialized_entries.sort(key=lambda item: item[0])
    return (
        MappingProxyType(lookup),
        tuple(entry for _, entry in serialized_entries),
    )


@dataclass(frozen=True, kw_only=True)
class RepresentationEvidence:
    """Observed category counts for one explicit protected axis."""

    axis: str
    counts: Mapping[str, int]
    source: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))
        object.__setattr__(
            self, "source", require_nonempty_string(self.source, "source")
        )
        if not isinstance(self.counts, Mapping):
            raise TypeError(
                "counts must be a mapping of category label -> integer count, "
                f"got {type(self.counts).__name__}."
            )
        if len(self.counts) < 2:
            raise ValueError("counts must contain at least two categories.")

        counts = {}
        for label, raw in self.counts.items():
            require_nonempty_string(label, "count category")
            if isinstance(raw, bool) or not isinstance(raw, Integral):
                raise TypeError(
                    f"count for {label!r} must be an integer, "
                    f"got {type(raw).__name__}."
                )
            count = int(raw)
            if count < 0:
                raise ValueError(f"count for {label!r} must be non-negative.")
            counts[label] = count
        if sum(counts.values()) <= 0:
            raise ValueError("counts must contain at least one observed item.")
        object.__setattr__(
            self, "counts", MappingProxyType(dict(sorted(counts.items())))
        )
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    @property
    def support(self) -> tuple[str, ...]:
        """Sorted category support, including explicit zero-count cells."""
        return tuple(self.counts)

    @property
    def total(self) -> int:
        """Total number of represented items."""
        return sum(self.counts.values())

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "axis": self.axis,
            "counts": dict(self.counts),
            "source": self.source,
            "provenance": thaw_json(self.provenance),
        }

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        axis: str,
        group_field: str,
        support: Sequence[str],
        source: str,
        value_map: Optional[Mapping[Any, str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "RepresentationEvidence":
        """Count one explicitly mapped category per record.

        The adapter never guesses fields, drops missing values, or infers the
        full support from observed rows.
        """
        support_labels = _validate_support(support)
        group_field = require_nonempty_string(group_field, "group_field")
        if isinstance(records, (str, bytes, Mapping)):
            raise TypeError("records must be an iterable of mapping rows.")
        try:
            iterator = iter(records)
        except TypeError as exc:
            raise TypeError("records must be an iterable of mapping rows.") from exc
        prepared_value_map, serialized_value_map = _prepare_value_map(
            value_map,
            support=support_labels,
        )

        counts = {label: 0 for label in support_labels}
        row_count = 0
        for index, record in enumerate(iterator):
            row_count += 1
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"record at row {index} must be a mapping, "
                    f"got {type(record).__name__}."
                )
            if group_field not in record:
                raise ValueError(
                    f"record at row {index} is missing mapped field "
                    f"{group_field!r}."
                )
            raw = record[group_field]
            if raw is None and (
                prepared_value_map is None or (0, "") not in prepared_value_map
            ):
                raise ValueError(
                    f"record at row {index} has a missing value for "
                    f"{group_field!r}."
                )
            if prepared_value_map is not None:
                _, lookup_key = _portable_raw_key(
                    raw,
                    path=f"record at row {index} value for {group_field!r}",
                )
                present = lookup_key in prepared_value_map
                if not present:
                    raise ValueError(
                        f"record at row {index} has unmapped value {raw!r} for "
                        f"{group_field!r}."
                    )
                label = prepared_value_map[lookup_key]
            else:
                label = raw
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    f"record at row {index} maps to a non-string or empty "
                    f"category {label!r}."
                )
            if label not in counts:
                raise ValueError(
                    f"record at row {index} maps to {label!r}, which is outside "
                    f"the explicit support {list(support_labels)!r}."
                )
            counts[label] += 1
        if row_count == 0:
            raise ValueError("records must contain at least one row.")

        return cls(
            axis=axis,
            counts=counts,
            source=source,
            provenance=_adapter_provenance(
                provenance,
                adapter="records",
                group_field=group_field,
                value_map=serialized_value_map,
            ),
        )

    @classmethod
    def from_dataframe(
        cls,
        frame: "pd.DataFrame",
        *,
        axis: str,
        group_column: str,
        support: Sequence[str],
        source: str,
        value_map: Optional[Mapping[Any, str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "RepresentationEvidence":
        """Adapt one explicitly named pandas column without guessing schema."""
        import pandas as pd

        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"frame must be a pandas DataFrame, got {type(frame).__name__}."
            )
        group_column = require_nonempty_string(group_column, "group_column")
        if group_column not in frame.columns:
            raise ValueError(f"frame is missing mapped group column {group_column!r}.")
        selected = frame[group_column]
        if not isinstance(selected, pd.Series):
            raise ValueError(
                f"frame mapped group column {group_column!r} must be unique."
            )
        record_evidence = cls.from_records(
            ({group_column: value} for value in selected.tolist()),
            axis=axis,
            group_field=group_column,
            support=support,
            source=source,
            value_map=value_map,
        )
        merged_provenance = _adapter_provenance(
            provenance,
            adapter="dataframe",
            group_field=group_column,
            value_map=record_evidence.provenance["value_map"],
        )
        return cls(
            axis=record_evidence.axis,
            counts=record_evidence.counts,
            source=record_evidence.source,
            provenance=merged_provenance,
        )


def _normalize_scores(value: Any) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("scores must be an ordered sequence of finite real numbers.")
    raw_scores = tuple(value)
    if not raw_scores:
        raise ValueError("scores must contain at least one value.")

    scores = []
    for index, raw in enumerate(raw_scores):
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise TypeError(
                f"score at row {index} must be a real number, "
                f"got {type(raw).__name__}."
            )
        try:
            score = float(raw)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"score at row {index} must be representable as a finite float."
            ) from exc
        if not math.isfinite(score):
            raise ValueError(f"score at row {index} must be finite.")
        scores.append(score)
    return tuple(scores)


def _normalize_score_range(value: Any) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("score_range must be an ordered pair of finite numbers.")
    endpoints = tuple(value)
    if len(endpoints) != 2:
        raise ValueError("score_range must contain exactly two endpoints.")

    normalized = []
    for name, raw in zip(("lower", "upper"), endpoints):
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise TypeError(
                f"score_range {name} endpoint must be a real number, "
                f"got {type(raw).__name__}."
            )
        try:
            endpoint = float(raw)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"score_range {name} endpoint must be representable as a finite float."
            ) from exc
        if not math.isfinite(endpoint):
            raise ValueError(f"score_range {name} endpoint must be finite.")
        normalized.append(endpoint)

    lower, upper = normalized
    if lower > upper:
        raise ValueError(
            "score_range lower endpoint must not exceed the upper endpoint."
        )
    return lower, upper


def _normalize_condition_roles(value: Any) -> tuple[str, str]:
    roles = normalize_string_sequence(
        value,
        field_name="condition_roles",
        allow_empty=False,
    )
    if len(roles) != 2:
        raise ValueError("condition_roles must contain exactly two roles.")
    if roles[0] == roles[1]:
        raise ValueError("condition_roles must contain two distinct roles.")
    return roles


def _normalize_pair_ids(
    value: Any,
) -> tuple[tuple[Any, ...], tuple[tuple[int, str], ...]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("pair_ids must be an ordered sequence of JSON scalars.")
    raw_pair_ids = tuple(value)
    if not raw_pair_ids:
        raise ValueError("pair_ids must contain at least one value.")

    portable_ids = []
    lookup_keys = []
    for index, raw in enumerate(raw_pair_ids):
        portable, lookup_key = _portable_raw_key(
            raw,
            path=f"pair_id at row {index}",
        )
        if portable is None:
            raise ValueError(f"pair_id at row {index} must not be missing.")
        if isinstance(portable, bool):
            raise TypeError(f"pair_id at row {index} must not be a boolean.")
        if isinstance(portable, str) and not portable.strip():
            raise ValueError(f"pair_id at row {index} must not be empty.")
        portable_ids.append(portable)
        lookup_keys.append(lookup_key)
    return tuple(portable_ids), tuple(lookup_keys)


@dataclass(frozen=True, kw_only=True)
class ScoredGroups:
    """Finite row-level scores paired with one explicitly declared group."""

    axis: str
    groups: Sequence[str]
    scores: Sequence[float]
    score_name: str
    source: str
    score_range: Optional[Sequence[float]] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))
        object.__setattr__(
            self,
            "score_name",
            require_nonempty_string(self.score_name, "score_name"),
        )
        object.__setattr__(
            self, "source", require_nonempty_string(self.source, "source")
        )

        groups = normalize_string_sequence(
            self.groups,
            field_name="groups",
            allow_empty=False,
        )
        scores = _normalize_scores(self.scores)
        if len(groups) != len(scores):
            raise ValueError(
                "groups and scores must contain the same number of rows; "
                f"got {len(groups)} groups and {len(scores)} scores."
            )
        support = tuple(sorted(set(groups)))
        if len(support) < 2:
            raise ValueError("groups must contain at least two observed categories.")

        score_range = _normalize_score_range(self.score_range)
        if score_range is not None:
            lower, upper = score_range
            for index, score in enumerate(scores):
                if score < lower or score > upper:
                    raise ValueError(
                        f"score at row {index} ({score!r}) is outside the "
                        f"declared score_range [{lower!r}, {upper!r}]."
                    )

        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "score_range", score_range)
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    @property
    def support(self) -> tuple[str, ...]:
        """Sorted observed group support."""
        return tuple(sorted(set(self.groups)))

    @property
    def total(self) -> int:
        """Number of scored rows."""
        return len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "axis": self.axis,
            "groups": list(self.groups),
            "scores": list(self.scores),
            "score_name": self.score_name,
            "source": self.source,
            "score_range": (
                None if self.score_range is None else list(self.score_range)
            ),
            "support": list(self.support),
            "provenance": thaw_json(self.provenance),
        }

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        axis: str,
        group_field: str,
        score_field: str,
        support: Sequence[str],
        score_name: str,
        source: str,
        value_map: Optional[Mapping[Any, str]] = None,
        score_range: Optional[Sequence[float]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "ScoredGroups":
        """Adapt explicitly mapped rows without coercing or dropping values."""
        support_labels = _validate_support(support)
        group_field = require_nonempty_string(group_field, "group_field")
        score_field = require_nonempty_string(score_field, "score_field")
        if group_field == score_field:
            raise ValueError("group_field and score_field must name distinct fields.")
        if isinstance(records, (str, bytes, Mapping)):
            raise TypeError("records must be an iterable of mapping rows.")
        try:
            iterator = iter(records)
        except TypeError as exc:
            raise TypeError("records must be an iterable of mapping rows.") from exc
        prepared_value_map, serialized_value_map = _prepare_value_map(
            value_map,
            support=support_labels,
        )

        groups = []
        scores = []
        for index, record in enumerate(iterator):
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"record at row {index} must be a mapping, "
                    f"got {type(record).__name__}."
                )
            for field_name in (group_field, score_field):
                if field_name not in record:
                    raise ValueError(
                        f"record at row {index} is missing mapped field "
                        f"{field_name!r}."
                    )

            raw_group = record[group_field]
            if raw_group is None and (
                prepared_value_map is None or (0, "") not in prepared_value_map
            ):
                raise ValueError(
                    f"record at row {index} has a missing value for "
                    f"{group_field!r}."
                )
            if prepared_value_map is not None:
                _, lookup_key = _portable_raw_key(
                    raw_group,
                    path=f"record at row {index} value for {group_field!r}",
                )
                if lookup_key not in prepared_value_map:
                    raise ValueError(
                        f"record at row {index} has unmapped value "
                        f"{raw_group!r} for {group_field!r}."
                    )
                group = prepared_value_map[lookup_key]
            else:
                group = raw_group
            if not isinstance(group, str) or not group.strip():
                raise ValueError(
                    f"record at row {index} maps to a non-string or empty "
                    f"group {group!r}."
                )
            if group not in support_labels:
                raise ValueError(
                    f"record at row {index} maps to {group!r}, which is "
                    f"outside the explicit support {list(support_labels)!r}."
                )

            raw_score = record[score_field]
            if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
                raise TypeError(
                    f"score at row {index} must be a real number, "
                    f"got {type(raw_score).__name__}."
                )
            try:
                score = float(raw_score)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"score at row {index} must be representable as a finite float."
                ) from exc
            if not math.isfinite(score):
                raise ValueError(f"score at row {index} must be finite.")
            groups.append(group)
            scores.append(score)

        if not groups:
            raise ValueError("records must contain at least one row.")
        observed_support = set(groups)
        if observed_support != set(support_labels):
            raise ValueError(
                "every explicit support category must have at least one scored "
                "row; missing categories: "
                f"{sorted(set(support_labels) - observed_support)!r}."
            )

        return cls(
            axis=axis,
            groups=groups,
            scores=scores,
            score_name=score_name,
            source=source,
            score_range=score_range,
            provenance=_adapter_provenance(
                provenance,
                adapter="records",
                group_field=group_field,
                score_field=score_field,
                value_map=serialized_value_map,
            ),
        )

    @classmethod
    def from_dataframe(
        cls,
        frame: "pd.DataFrame",
        *,
        axis: str,
        group_column: str,
        score_column: str,
        support: Sequence[str],
        score_name: str,
        source: str,
        value_map: Optional[Mapping[Any, str]] = None,
        score_range: Optional[Sequence[float]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "ScoredGroups":
        """Adapt two explicit DataFrame columns without guessing or coercion."""
        import pandas as pd

        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"frame must be a pandas DataFrame, got {type(frame).__name__}."
            )
        group_column = require_nonempty_string(group_column, "group_column")
        score_column = require_nonempty_string(score_column, "score_column")
        if group_column == score_column:
            raise ValueError("group_column and score_column must be distinct.")
        for role, column in (("group", group_column), ("score", score_column)):
            if column not in frame.columns:
                raise ValueError(f"frame is missing mapped {role} column {column!r}.")
            if not isinstance(frame[column], pd.Series):
                raise ValueError(
                    f"frame mapped {role} column {column!r} must be unique."
                )

        record_evidence = cls.from_records(
            (
                {group_column: group, score_column: score}
                for group, score in zip(
                    frame[group_column].tolist(),
                    frame[score_column].tolist(),
                )
            ),
            axis=axis,
            group_field=group_column,
            score_field=score_column,
            support=support,
            score_name=score_name,
            source=source,
            value_map=value_map,
            score_range=score_range,
        )
        merged_provenance = _adapter_provenance(
            provenance,
            adapter="dataframe",
            group_field=group_column,
            score_field=score_column,
            value_map=record_evidence.provenance["value_map"],
        )
        return cls(
            axis=record_evidence.axis,
            groups=record_evidence.groups,
            scores=record_evidence.scores,
            score_name=record_evidence.score_name,
            source=record_evidence.source,
            score_range=record_evidence.score_range,
            provenance=merged_provenance,
        )


@dataclass(frozen=True, kw_only=True)
class PairedScores:
    """Finite scores for complete two-condition counterfactual pairs."""

    axis: str
    pair_ids: Sequence[Any]
    conditions: Sequence[str]
    scores: Sequence[float]
    condition_roles: Sequence[str]
    score_name: str
    source: str
    pairing_basis: str
    score_range: Optional[Sequence[float]] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))
        object.__setattr__(
            self,
            "score_name",
            require_nonempty_string(self.score_name, "score_name"),
        )
        object.__setattr__(
            self,
            "source",
            require_nonempty_string(self.source, "source"),
        )
        object.__setattr__(
            self,
            "pairing_basis",
            require_nonempty_string(self.pairing_basis, "pairing_basis"),
        )

        roles = _normalize_condition_roles(self.condition_roles)
        conditions = normalize_string_sequence(
            self.conditions,
            field_name="conditions",
            allow_empty=False,
        )
        scores = _normalize_scores(self.scores)
        pair_ids, pair_keys = _normalize_pair_ids(self.pair_ids)
        lengths = {len(pair_ids), len(conditions), len(scores)}
        if len(lengths) != 1:
            raise ValueError(
                "pair_ids, conditions, and scores must contain the same number "
                f"of rows; got {len(pair_ids)}, {len(conditions)}, and "
                f"{len(scores)}."
            )

        score_range = _normalize_score_range(self.score_range)
        if score_range is not None:
            lower, upper = score_range
            for index, score in enumerate(scores):
                if score < lower or score > upper:
                    raise ValueError(
                        f"score at row {index} ({score!r}) is outside the "
                        f"declared score_range [{lower!r}, {upper!r}]."
                    )

        pairs: dict[tuple[int, str], dict[str, Any]] = {}
        for index, (pair_id, pair_key, condition, score) in enumerate(
            zip(pair_ids, pair_keys, conditions, scores)
        ):
            if condition not in roles:
                raise ValueError(
                    f"condition at row {index} ({condition!r}) is outside the "
                    f"declared condition_roles {list(roles)!r}."
                )
            pair = pairs.setdefault(
                pair_key,
                {"pair_id": pair_id, "scores": {}},
            )
            if condition in pair["scores"]:
                raise ValueError(
                    f"pair {pair_id!r} contains duplicate condition role "
                    f"{condition!r}."
                )
            pair["scores"][condition] = score

        canonical_pair_ids = []
        canonical_conditions = []
        canonical_scores = []
        for pair_key in sorted(pairs):
            pair = pairs[pair_key]
            missing = [role for role in roles if role not in pair["scores"]]
            if missing:
                raise ValueError(
                    f"pair {pair['pair_id']!r} is missing condition role(s): "
                    f"{missing!r}."
                )
            for role in roles:
                canonical_pair_ids.append(pair["pair_id"])
                canonical_conditions.append(role)
                canonical_scores.append(pair["scores"][role])

        object.__setattr__(self, "pair_ids", tuple(canonical_pair_ids))
        object.__setattr__(self, "conditions", tuple(canonical_conditions))
        object.__setattr__(self, "scores", tuple(canonical_scores))
        object.__setattr__(self, "condition_roles", roles)
        object.__setattr__(self, "score_range", score_range)
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    @property
    def total(self) -> int:
        """Number of scored rows across both condition roles."""
        return len(self.scores)

    @property
    def pair_count(self) -> int:
        """Number of complete validated pairs."""
        return self.total // 2

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "axis": self.axis,
            "pair_ids": list(self.pair_ids),
            "conditions": list(self.conditions),
            "scores": list(self.scores),
            "condition_roles": list(self.condition_roles),
            "score_name": self.score_name,
            "source": self.source,
            "pairing_basis": self.pairing_basis,
            "score_range": (
                None if self.score_range is None else list(self.score_range)
            ),
            "pair_count": self.pair_count,
            "provenance": thaw_json(self.provenance),
        }

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        axis: str,
        pair_id_field: str,
        condition_field: str,
        score_field: str,
        condition_roles: Sequence[str],
        score_name: str,
        source: str,
        pairing_basis: str,
        condition_map: Optional[Mapping[Any, str]] = None,
        score_range: Optional[Sequence[float]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "PairedScores":
        """Adapt explicit pair/condition/score fields without dropping rows."""
        roles = _normalize_condition_roles(condition_roles)
        pair_id_field = require_nonempty_string(pair_id_field, "pair_id_field")
        condition_field = require_nonempty_string(
            condition_field,
            "condition_field",
        )
        score_field = require_nonempty_string(score_field, "score_field")
        mapped_fields = (pair_id_field, condition_field, score_field)
        if len(set(mapped_fields)) != len(mapped_fields):
            raise ValueError(
                "pair_id_field, condition_field, and score_field must be distinct."
            )
        if isinstance(records, (str, bytes, Mapping)):
            raise TypeError("records must be an iterable of mapping rows.")
        try:
            iterator = iter(records)
        except TypeError as exc:
            raise TypeError("records must be an iterable of mapping rows.") from exc

        prepared_condition_map, serialized_condition_map = _prepare_value_map(
            condition_map,
            support=roles,
        )
        pair_ids = []
        conditions = []
        scores = []
        for index, record in enumerate(iterator):
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"record at row {index} must be a mapping, "
                    f"got {type(record).__name__}."
                )
            for field_name in mapped_fields:
                if field_name not in record:
                    raise ValueError(
                        f"record at row {index} is missing mapped field "
                        f"{field_name!r}."
                    )

            raw_condition = record[condition_field]
            if raw_condition is None and (
                prepared_condition_map is None or (0, "") not in prepared_condition_map
            ):
                raise ValueError(
                    f"record at row {index} has a missing value for "
                    f"{condition_field!r}."
                )
            if prepared_condition_map is not None:
                _, lookup_key = _portable_raw_key(
                    raw_condition,
                    path=(f"record at row {index} value for {condition_field!r}"),
                )
                if lookup_key not in prepared_condition_map:
                    raise ValueError(
                        f"record at row {index} has unmapped value "
                        f"{raw_condition!r} for {condition_field!r}."
                    )
                condition = prepared_condition_map[lookup_key]
            else:
                condition = raw_condition

            pair_ids.append(record[pair_id_field])
            conditions.append(condition)
            scores.append(record[score_field])

        if not pair_ids:
            raise ValueError("records must contain at least one row.")
        return cls(
            axis=axis,
            pair_ids=pair_ids,
            conditions=conditions,
            scores=scores,
            condition_roles=roles,
            score_name=score_name,
            source=source,
            pairing_basis=pairing_basis,
            score_range=score_range,
            provenance=_paired_adapter_provenance(
                provenance,
                adapter="records",
                pair_id_field=pair_id_field,
                condition_field=condition_field,
                score_field=score_field,
                condition_map=serialized_condition_map,
            ),
        )

    @classmethod
    def from_dataframe(
        cls,
        frame: "pd.DataFrame",
        *,
        axis: str,
        pair_id_column: str,
        condition_column: str,
        score_column: str,
        condition_roles: Sequence[str],
        score_name: str,
        source: str,
        pairing_basis: str,
        condition_map: Optional[Mapping[Any, str]] = None,
        score_range: Optional[Sequence[float]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "PairedScores":
        """Adapt three explicit DataFrame columns without guessing schema."""
        import pandas as pd

        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"frame must be a pandas DataFrame, got {type(frame).__name__}."
            )
        pair_id_column = require_nonempty_string(
            pair_id_column,
            "pair_id_column",
        )
        condition_column = require_nonempty_string(
            condition_column,
            "condition_column",
        )
        score_column = require_nonempty_string(score_column, "score_column")
        mapped_columns = (pair_id_column, condition_column, score_column)
        if len(set(mapped_columns)) != len(mapped_columns):
            raise ValueError(
                "pair_id_column, condition_column, and score_column must be "
                "distinct."
            )
        for role, column in (
            ("pair ID", pair_id_column),
            ("condition", condition_column),
            ("score", score_column),
        ):
            if column not in frame.columns:
                raise ValueError(f"frame is missing mapped {role} column {column!r}.")
            if not isinstance(frame[column], pd.Series):
                raise ValueError(
                    f"frame mapped {role} column {column!r} must be unique."
                )

        record_evidence = cls.from_records(
            (
                {
                    pair_id_column: pair_id,
                    condition_column: condition,
                    score_column: score,
                }
                for pair_id, condition, score in zip(
                    frame[pair_id_column].tolist(),
                    frame[condition_column].tolist(),
                    frame[score_column].tolist(),
                )
            ),
            axis=axis,
            pair_id_field=pair_id_column,
            condition_field=condition_column,
            score_field=score_column,
            condition_roles=condition_roles,
            score_name=score_name,
            source=source,
            pairing_basis=pairing_basis,
            condition_map=condition_map,
            score_range=score_range,
        )
        merged_provenance = _paired_adapter_provenance(
            provenance,
            adapter="dataframe",
            pair_id_field=pair_id_column,
            condition_field=condition_column,
            score_field=score_column,
            condition_map=record_evidence.provenance["condition_map"],
        )
        return cls(
            axis=record_evidence.axis,
            pair_ids=record_evidence.pair_ids,
            conditions=record_evidence.conditions,
            scores=record_evidence.scores,
            condition_roles=record_evidence.condition_roles,
            score_name=record_evidence.score_name,
            source=record_evidence.source,
            pairing_basis=record_evidence.pairing_basis,
            score_range=record_evidence.score_range,
            provenance=merged_provenance,
        )


__all__ = ["PairedScores", "RepresentationEvidence", "ScoredGroups"]
