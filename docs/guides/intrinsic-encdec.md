# Intrinsic × encoder-decoder

Four metrics measure intrinsic bias in encoder-decoder models. All need
`task="seq2seq"`, and all but one work by generating a translation or completion
and analysing the *text the model chose to produce*:

| Metric | Reads | Evidence |
|---|---|---|
| `lexical_frequency_proportion` | word-frequency profile of generated text | sentences |
| `morphological_choice_divergence` | morphological complexity of generated text | sentences |
| `stereotypical_divergence` | task performance on stereotype vs anti-stereotype sets | `StereotypeLabelled` |
| `stereotypical_value_attribution` | Shapley attribution over attention heads | `WordSets` |

## End to end

```python
from fairlms.metrics import (
    LexicalFrequencyProportion,
    MorphologicalChoiceDivergence,
    StereotypicalDivergence,
)
from fairlms.metrics.data import LabeledSentences, StereotypeLabelled
from fairlms.models import HuggingFaceModel

t5 = HuggingFaceModel("t5-small", task="seq2seq")

# --- corpus-level metrics: just a list of sentences --------------------------
sentences = ["The nurse treated the patient.", "The engineer fixed the bridge."]

lfp = LexicalFrequencyProportion(max_new_tokens=24).compute(t5, sentences)
print(lfp.score)              # 0.5    — pb1
print(lfp.details["pb2"])     # 0.1667
print(lfp.details["pb3"])     # 0.3333

mcd = MorphologicalChoiceDivergence(max_new_tokens=24).compute(t5, sentences)
print(mcd.score)              # 0.0    — mean_h
print(mcd.details["mean_d"])  # 0.0

# --- SD: paired labelled sets, measured by task performance -----------------
data = StereotypeLabelled(
    stereotype=LabeledSentences(
        sentences=["The nurse said she was tired.",
                   "The engineer said he was tired."],
        labels=["female", "male"],
    ),
    anti_stereotype=LabeledSentences(
        sentences=["The nurse said he was tired.",
                   "The engineer said she was tired."],
        labels=["male", "female"],
    ),
)

sd = StereotypicalDivergence(max_new_tokens=8).compute(t5, data)
print(sd.score)                    # 0.0
print(sd.details["m_stereo"])      # 0.5 — performance on the stereotype set
print(sd.details["m_anti"])        # 0.5 — performance on the anti-stereotype set
print(sd.details["delta_s"])       # 0.0 — the divergence
```

Real output on `t5-small` with two sentences. `lfp` and `mcd` take a plain list
of strings, so a corpus is a list comprehension away; both `pb1`/`pb2`/`pb3` and
`mean_h`/`mean_d` are always in `details` regardless of which is the headline
`score`.

## Zero is a real value here, and so is a degenerate one

`mcd.score == 0.0` on two sentences means the generated translations had
identical morphological complexity — plausible for a two-sentence sample from a
60M-parameter model, and *not* the same as "no bias". Same for `sd.score == 0.0`
with `m_stereo == m_anti == 0.5`: both sets scored equally, on one sentence
each. These metrics need corpus-scale evidence before the numbers carry
information; the snippet above shows the interface, not a result.

`t5-small` in particular is a weak translator. Use `t5-base`, `google/mt5-base`
or a task-specific checkpoint for anything you intend to report — the metrics
measure the model's output, so a model that generates poorly produces
uninformative scores rather than an error.

## SD is a French-translation cue task, not a generic scorer

Worth knowing before you use it: `stereotypical_divergence` scores whether the
model prefers a gendered French continuation (`Il`/`Lui` vs `Elle`/`Celle-ci`)
for each source sentence, and compares that accuracy between the stereotype and
anti-stereotype sets. So `labels` must be `"male"` or `"female"` — any other
string silently scores `0.5` for that row rather than raising, which is how you
get a suspiciously flat `m_stereo == m_anti == 0.5`.

!!! warning "`metric_fn` is a closed set, not a free hook"
    Despite the signature, `metric_fn` is dispatched by function `__name__`
    through a two-entry table. Only two callables work:

    ```python
    from fairlms.definition.encoder_decoder.intrinsic_bias.stereotypical_association.sd.sd import (
        age_accuracy, pronoun_accuracy,
    )

    StereotypicalDivergence(metric_fn=age_accuracy)   # labels: "young" / "old"
    ```

    Passing a lambda or your own function raises `KeyError: '<lambda>'`, because
    each scorer is paired with a specific prediction routine (French gender cues
    or French age cues). `metric_fn=None` uses `pronoun_accuracy`.

## Head attribution

`stereotypical_value_attribution` is the mechanistic one, and its container is
used unusually: `target_1` / `target_2` hold the stereotypical and
anti-stereotypical **sentences**, and the attribute roles are ignored — pass the
same sentences again if you have no attribute sets.

```python
from fairlms.metrics import StereotypicalValueAttribution
from fairlms.metrics.data import WordSets

stereo = ["The nurse said she was tired."]
anti = ["The nurse said he was tired."]

StereotypicalValueAttribution(n_samples=5, top_pct=0.10).compute(
    t5, WordSets(target_1=stereo, target_2=anti, attribute_1=stereo, attribute_2=anti)
)
```

The stereotype `direction` is derived from the sentences when not supplied, which
also guarantees it matches the model's hidden size. Pass `direction=` only if you
have a precomputed unit vector of length `d_model`.

## Data sources

None of these four has a bundled dataset. The leaf runners under
`fairlms/definition/encoder_decoder/` pull sentences from XSum, Europarl,
WinoBias and XNLI at runtime via `datasets.load_dataset`, which is worth knowing
if you are working offline — the corpora are not vendored, only CrowS-Pairs and
BBQ are.
