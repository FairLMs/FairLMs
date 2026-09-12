# Changelog

## 0.5.0.dev0

Corrected StereoSet roles/context and CAT LMS comparisons; connected public
SelfDebiasing generation to probability damping; required projection layer and
pair identity; measured INLP after the final projection; preserved API model
selection and stopped successful retries. XNLI now produces deduplicated
counterfactual evidence. Added configured before/after evaluation, JSON-safe
undefined scores, run provenance, revision controls, real offline model tests,
source archive validation, and corpus checksums/attribution. No upstream release
or paper update is implied by this development version.

**Bundled corpora consolidated.** 84 files under `fairlms/definition/` that were
byte-identical to a copy under `fairlms/data/` were deleted, removing
359,582,594 bytes. `fairlms/data/` is now the single source, and
`fairlms/data/checksums.json` pins the bytes that remain. Loaders resolve
`fairlms/data/` first and still fall back to legacy locations, so externally
maintained checkouts are unaffected. Historical MCD outputs from earlier
development are no longer distributed; they were never validation results for
this version.


Versions follow the scheme described in the
[README](https://github.com/michaellarionov/FairLMs#versioning): while the
package is pre-1.0, the minor version moves on behaviour changes and the patch
version on additions and fixes. Metric definitions can change between minor
versions, so pin a version when reporting a score. See
[Citing FairLMs](citation.md).

## Unreleased

**Mismatched model heads are refused.** Every metric now declares
`required_task`, and `fairlms.metrics.resolve.check_task` compares it against
the `task` a checkpoint was loaded with. Passing a `task="encoder"` model to
`crows_pairs_score` previously failed with an `AttributeError` on a missing
`.logits`, and the reverse mismatch could return numbers read from a randomly
initialized head; both now raise a `TypeError` that names the required task,
the actual task and the fix. Raw `(tokenizer, model)` tuples carry no task and
are unaffected. The 9 metrics that score precomputed predictions or call an API
declare `required_task = None`.

**`stereotypical_divergence` accepts a real custom scorer.** `metric_fn` was
dispatched by function `__name__` through a two-entry table, so any callable
other than `pronoun_accuracy` or `age_accuracy` failed with a bare `KeyError`.
It now takes an accompanying `predict_fn`, and an unpaired custom scorer raises
a `ValueError` naming both the built-ins and the escape hatch.

**`stereotypical_divergence` validates its label vocabulary.** Its scorers
return 0.5 for a gold label they do not recognise, so a wholly wrong vocabulary
(`["negative"]` where `["male", "female"]` was meant) produced
`m_stereo == m_anti == 0.5` and a divergence of exactly 0.0, indistinguishable
from a real finding of parity. A complete mismatch is now an error; individual
unknown labels still score as chance.

**`counterfactual_auc` refuses inputs it cannot estimate.** The underlying
`compute_auc` returns `0.0` when it cannot fit a probe, but as a score `0.0` is
the most extreme possible finding. String labels (which count as neither class),
a class with fewer than two members, and a `test_ratio` too small to hold both
classes are now all rejected up front with named errors. `compute_auc` itself is
unchanged, so callers using it directly keep the diagnostic short-circuit.

**Documentation.** A Material for MkDocs site at
<https://michaellarionov.github.io/FairLMs/>, with registry tables generated
from the live registries and checked in CI.

## 0.4.0

The WEAT/SEAT/CEAT permutation seed became explicit constructor configuration
(`seed=`, default `None`). It appears in `get_params()`, so a reported p-value
can be pinned and reproduced instead of depending on ambient RNG state.

## 0.3.1

Added `fairlms.data`: the bundled WEAT and SEAT word sets (`weat_c1`–`weat_c4`,
`seat_c1`–`seat_c4`) as validated `WordSets` containers, with a `WORD_SETS`
registry and `get_word_set` / `list_word_sets` accessors.

## 0.3.0

Completed the rename to `fairlms`. The package, its imports and the repository
all moved from the previous name; there is no compatibility shim.

## 0.2.0

Reusable dataset and score-table diagnostics: the `fairlms.diagnostics` package,
with axis representativeness (`b_rep`) and the four scoring-instrument audits,
the `ready` / `blocked` / `not_applicable` / `failed` applicability model, and
versioned JSON reports.

## 0.1.x

Made the package installable: declared previously undeclared dependencies,
added release metadata, single-sourced the version through
`fairlms/_version.py`, and read it via AST rather than a regex.
