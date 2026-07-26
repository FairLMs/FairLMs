"""CrowS-Pairs Score (CPS) runner — uses shared Phase-1 loaders/adapters."""

import os
from pathlib import Path

import pandas as pd

from fairLLMs.datasets import CrowSPairs, StereoSet, XNLIReligionPairs
from fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cps.cps import (  # noqa: E501
    compute_cps,
)
from fairLLMs.models import load_masked_lm
from fairLLMs.utils import artifacts_root, results_to_csv

_MAIN_DIR = Path(__file__).resolve().parent


def run_cps(name, sentence_pairs, tokenizer, model):
    if not sentence_pairs:
        print(f"\n  [skip] {name} — no pairs")
        return {"dataset": name, "cps_%": None, "acc_%": None, "n_pairs": 0}
    print(f"\n  Running CPS for: {name}  (n={len(sentence_pairs)})")
    score, accuracy, per_bias_type = compute_cps(tokenizer, model, sentence_pairs)
    return {"dataset": name, "cps_score": round(score, 2)}


def main():
    loaded = load_masked_lm("bert-base-uncased")
    tokenizer, model = loaded.tokenizer, loaded.model
    print(f"[INFO] Loaded '{loaded.name}' on {loaded.device}")

    print("[INFO] Loading datasets...")
    crows = list(CrowSPairs().load())
    print(f"[INFO] CrowS-Pairs bias_type categories found: {CrowSPairs().bias_types()}")
    print(f"[INFO] CrowS-Pairs total (all categories pooled): {len(crows)} pairs")

    stereoset = list(StereoSet(config="intersentence").load())
    print(f"[INFO] StereoSet total (all categories pooled): {len(stereoset)} pairs")

    xnli = list(XNLIReligionPairs().load())
    print(f"[INFO] XNLI pairs loaded: {len(xnli)}")

    print("\n" + "=" * 55)
    print("  CPS on BERT-base-uncased  ")
    print("=" * 55)

    configs = [
        ("CrowS-Pairs", crows),
        ("StereoSet", stereoset),
        ("XNLI", xnli),
    ]

    results = []
    for name, pairs in configs:
        results.append(run_cps(name, pairs, tokenizer, model))

    # Prefer package artifacts/; also keep a copy next to this leaf for continuity.
    out = results_to_csv(results, "cps_results.csv")
    legacy = os.path.join(_MAIN_DIR, "cps_results.csv")
    pd.DataFrame(results).to_csv(legacy, index=False)
    print(f"Saved: {out}")
    print(f"Saved (legacy leaf path): {legacy}")
    print(f"Artifacts dir: {artifacts_root()}")


if __name__ == "__main__":
    main()
