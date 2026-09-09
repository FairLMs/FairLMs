"""Pre-processing mitigators: black-box, applied to the data before training.

These transform evidence, not models. Each records the lexicon, filter or target
distribution it used in provenance, and each **refuses** a row it cannot handle
rather than dropping it: a corpus that silently shrank during augmentation would
change the very distribution the transform claims to be correcting.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from fairlms.mitigation.base import MitigationResult, Mitigator
from fairlms.mitigation.evidence import (
    CorpusWithBenignPool,
    CorpusWithLexicon,
    GroupLabeledRecords,
    PromptSpec,
    TextRecords,
)

__all__ = [
    "CounterfactualDataAugmentation",
    "DebiasingPrompt",
    "GroupLabelReweighting",
    "IdentityTermAugmentation",
]

_ALL_ARCHITECTURES = ("encoder_only", "decoder_only", "encoder_decoder")

#: Word-boundary tokenizer used for surface-form swaps. Deliberately simple and
#: declared: a mitigator must not pull in spaCy to split a string.
_WORD = re.compile(r"\b\w+\b", re.UNICODE)


def _match_case(source: str, replacement: str) -> str:
    """Carry *source*'s casing onto *replacement*."""
    if source.isupper() and len(source) > 1:
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _swap(text: str, table: Dict[str, str]) -> Tuple[str, int]:
    """Return the swapped text and how many terms were replaced."""
    count = 0

    def replace(match: "re.Match") -> str:
        nonlocal count
        word = match.group(0)
        target = table.get(word.lower())
        if target is None:
            return word
        count += 1
        return _match_case(word, target)

    return _WORD.sub(replace, text), count


class CounterfactualDataAugmentation(Mitigator):
    """Swap sensitive surface forms via an explicit lexicon; emit original + swapped.

    This is counterfactual data **augmentation**: the swapped copy is added
    alongside the original. Counterfactual data *substitution*, which replaces
    rather than adds, is deliberately not provided - see the out-of-scope list
    in the package docstring.

    A record containing no lexicon term cannot be rewritten. That is refused,
    not dropped: silently emitting only the augmentable subset would skew the
    corpus toward exactly the rows that mention the protected attribute.

    Parameters
    ----------
    on_unrewritable:
        ``"refuse"`` (default) raises on the first record carrying no lexicon
        term. ``"keep"`` passes it through unswapped, which the caller must ask
        for explicitly and which is recorded in provenance.

    Examples
    --------
    >>> from fairlms.mitigation import (
    ...     CorpusWithLexicon, CounterfactualDataAugmentation, SwapLexicon,
    ...     TextRecords)
    >>> evidence = CorpusWithLexicon(
    ...     records=TextRecords(texts=["He is a nurse."], source="doctest"),
    ...     lexicon=SwapLexicon(
    ...         axis="gender", pairs=[("he", "she")], source="doctest"),
    ... )
    >>> result = CounterfactualDataAugmentation().apply(None, evidence)
    >>> tuple(result.result.texts)
    ('He is a nurse.', 'She is a nurse.')
    """

    name = "counterfactual_data_augmentation"
    category = "pre"
    access = "black_box"
    architectures = _ALL_ARCHITECTURES
    requires = frozenset()
    accepts = (CorpusWithLexicon,)

    def __init__(self, on_unrewritable: str = "refuse"):
        self.on_unrewritable = on_unrewritable

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        records, lexicon = evidence.records, evidence.lexicon
        if self.on_unrewritable not in ("refuse", "keep"):
            raise ValueError(
                "on_unrewritable must be 'refuse' or 'keep'; got "
                f"{self.on_unrewritable!r}."
            )
        table = lexicon.mapping()

        texts: List[str] = []
        ids: List[str] = []
        n_swapped = 0
        n_untouched = 0
        for index, text in enumerate(records.texts):
            row_id = records.ids[index] if records.ids else str(index)
            swapped, count = _swap(text, table)
            texts.append(text)
            ids.append(f"{row_id}:original")
            if count == 0:
                n_untouched += 1
                if self.on_unrewritable == "refuse":
                    raise ValueError(
                        f"{self.name}: record {row_id!r} contains no term from the "
                        f"{lexicon.axis!r} lexicon, so it cannot be rewritten: "
                        f"{text!r}. Extend the lexicon, remove the record "
                        f"deliberately, or pass on_unrewritable='keep'. It will "
                        f"not be dropped silently."
                    )
                texts.append(text)
                ids.append(f"{row_id}:kept")
            else:
                n_swapped += 1
                texts.append(swapped)
                ids.append(f"{row_id}:swapped")

        augmented = TextRecords(
            texts=texts,
            source=f"{records.source}+cda",
            ids=ids,
            provenance={
                "transform": self.name,
                "axis": lexicon.axis,
                "lexicon_source": lexicon.source,
            },
        )
        return self._result(
            augmented,
            axis=lexicon.axis,
            lexicon_source=lexicon.source,
            lexicon_pairs=[list(p) for p in lexicon.pairs],
            n_input_rows=records.n_rows,
            n_output_rows=augmented.n_rows,
            n_swapped=n_swapped,
            n_unrewritable=n_untouched,
            on_unrewritable=self.on_unrewritable,
        )


class GroupLabelReweighting(Mitigator):
    r"""Per-row weights ``w = pi(y, a) / P_hat(y, a)`` toward a declared target.

    ``P_hat`` is the empirical joint of outcome and group in the supplied rows.
    ``pi`` is the target joint. The default target is the **independence**
    product ``P(y) * P(a)``, the standard reweighting choice: it removes the
    label-group association while leaving both marginals as observed.

    Parameters
    ----------
    target:
        ``"independence"`` for ``P(y) P(a)``, or ``"uniform"`` for an equal mass
        in every observed ``(y, a)`` cell.

    Examples
    --------
    >>> from fairlms.mitigation import GroupLabelReweighting, GroupLabeledRecords
    >>> records = GroupLabeledRecords(
    ...     axis="gender",
    ...     groups=["f", "f", "m", "m"],
    ...     labels=["no", "no", "yes", "no"],
    ...     label_name="outcome", source="doctest",
    ... )
    >>> weights = GroupLabelReweighting().apply(None, records).result["weights"]
    >>> len(weights)
    4
    """

    name = "group_label_reweighting"
    category = "pre"
    access = "black_box"
    architectures = _ALL_ARCHITECTURES
    requires = frozenset()
    accepts = (GroupLabeledRecords,)

    def __init__(self, target: str = "independence"):
        self.target = target

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if self.target not in ("independence", "uniform"):
            raise ValueError(
                f"target must be 'independence' or 'uniform'; got {self.target!r}."
            )
        n = evidence.n_rows
        joint: Dict[Tuple[str, str], int] = {}
        label_counts: Dict[str, int] = {}
        group_counts: Dict[str, int] = {}
        for label, group in zip(evidence.labels, evidence.groups):
            joint[(label, group)] = joint.get((label, group), 0) + 1
            label_counts[label] = label_counts.get(label, 0) + 1
            group_counts[group] = group_counts.get(group, 0) + 1

        if self.target == "independence":
            pi = {
                cell: (label_counts[cell[0]] / n) * (group_counts[cell[1]] / n)
                for cell in joint
            }
        else:
            pi = {cell: 1.0 / len(joint) for cell in joint}

        weights = [
            pi[(label, group)] / (joint[(label, group)] / n)
            for label, group in zip(evidence.labels, evidence.groups)
        ]
        return self._result(
            {
                "weights": weights,
                "target": self.target,
                "cells": {
                    f"{label}|{group}": {
                        "observed": count / n,
                        "target": pi[(label, group)],
                        "weight": pi[(label, group)] / (count / n),
                    }
                    for (label, group), count in sorted(joint.items())
                },
            },
            axis=evidence.axis,
            label_name=evidence.label_name,
            n_rows=n,
            source=evidence.source,
        )


class IdentityTermAugmentation(Mitigator):
    """Add benign identity-term examples to break the term-to-label shortcut.

    A classifier trained on a corpus where an identity term appears mostly in
    toxic examples learns the term as the signal. Adding benign examples that
    carry the same terms breaks that correlation at the data level.

    The benign pool is supplied by the caller and labelled with the declared
    ``benign_label``: which outcome counts as benign is a property of the task,
    not something to infer.

    Parameters
    ----------
    benign_label:
        The outcome label attached to every added example.

    Examples
    --------
    >>> from fairlms.mitigation import (
    ...     CorpusWithBenignPool, GroupLabeledRecords, IdentityTermAugmentation,
    ...     TextRecords)
    >>> corpus = GroupLabeledRecords(
    ...     axis="religion",
    ...     groups=["muslim", "muslim", "christian"],
    ...     labels=["toxic", "toxic", "clean"],
    ...     label_name="toxicity", source="doctest",
    ...     texts=["a", "b", "c"],
    ... )
    >>> pool = TextRecords(
    ...     texts=["My muslim neighbour bakes bread."], source="doctest")
    >>> out = IdentityTermAugmentation(benign_label="clean").apply(
    ...     None, CorpusWithBenignPool(corpus=corpus, pool=pool))
    >>> out.result.n_rows
    4
    """

    name = "identity_term_augmentation"
    category = "pre"
    access = "black_box"
    architectures = _ALL_ARCHITECTURES
    requires = frozenset()
    accepts = (CorpusWithBenignPool,)

    def __init__(self, benign_label: str = "", benign_group: str = ""):
        self.benign_label = benign_label
        self.benign_group = benign_group

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        corpus, pool = evidence.corpus, evidence.pool
        if not self.benign_label:
            raise ValueError(
                f"{self.name}: benign_label must be declared. Which outcome counts "
                f"as benign is a property of the task; observed labels are "
                f"{sorted(set(corpus.labels))!r}."
            )
        if self.benign_label not in set(corpus.labels):
            raise ValueError(
                f"{self.name}: benign_label {self.benign_label!r} does not occur in "
                f"the corpus; observed labels are {sorted(set(corpus.labels))!r}."
            )
        group = self.benign_group or corpus.groups[0]
        if group not in set(corpus.groups):
            raise ValueError(
                f"{self.name}: benign_group {group!r} does not occur in the corpus; "
                f"observed groups are {sorted(set(corpus.groups))!r}."
            )

        augmented = GroupLabeledRecords(
            axis=corpus.axis,
            groups=list(corpus.groups) + [group] * pool.n_rows,
            labels=list(corpus.labels) + [self.benign_label] * pool.n_rows,
            label_name=corpus.label_name,
            source=f"{corpus.source}+identity_terms",
            texts=list(corpus.texts) + list(pool.texts),
        )
        return self._result(
            augmented,
            axis=corpus.axis,
            benign_label=self.benign_label,
            benign_group=group,
            pool_source=pool.source,
            n_input_rows=corpus.n_rows,
            n_added=pool.n_rows,
            n_output_rows=augmented.n_rows,
        )


class DebiasingPrompt(Mitigator):
    """Prepend system/task instruction templates to queries.

    Pre-processing, but **generative models only**: an instruction has nowhere
    to go in a masked-token or classification forward pass. Declared as
    decoder-only and encoder-decoder, and refused elsewhere.

    Examples
    --------
    >>> from fairlms.mitigation import DebiasingPrompt, PromptSpec
    >>> spec = PromptSpec(
    ...     templates=["Answer without stereotyping. {query}"],
    ...     queries=["Describe a nurse."],
    ... )
    >>> prompts, = DebiasingPrompt().apply(None, spec).result["prompts"]
    >>> tuple(prompts)
    ('Answer without stereotyping. Describe a nurse.',)
    """

    name = "debiasing_prompt"
    category = "pre"
    access = "black_box"
    # Generative-only, and that restriction lives here rather than in
    # `requires`: this rewrites query strings and reads nothing from a model, so
    # declaring a capability would falsely demand one be supplied.
    architectures = ("decoder_only", "encoder_decoder")
    requires = frozenset()
    accepts = (PromptSpec,)

    def __init__(self) -> None:
        pass

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if not evidence.queries:
            raise ValueError(
                f"{self.name}: PromptSpec carries no queries to rewrite. Build "
                f"PromptSpec(templates=[...], queries=[...])."
            )
        prompts = [list(evidence.render(query)) for query in evidence.queries]
        return self._result(
            {
                "prompts": prompts,
                "templates": list(evidence.templates),
                "queries": list(evidence.queries),
            },
            n_queries=len(evidence.queries),
            n_templates=len(evidence.templates),
            attribute=evidence.attribute,
        )
