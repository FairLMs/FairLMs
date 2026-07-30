"""WEAT via the public fairllms API (one Caliskan-style test).

Shows the preferred shape: configuration in the constructor, data as a
validated container passed positionally to ``compute``.
"""

from fairllms.definition.encoder_only.intrinsic_bias.similarity_based.weat.data import C1
from fairllms.metrics import WEAT, WordSets
from fairllms.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="encoder")

    words = WordSets(
        target_1=C1["t1"][:8],
        target_2=C1["t2"][:8],
        attribute_1=C1["a1"][:8],
        attribute_2=C1["a2"][:8],
    )

    result = WEAT(pooling="mean", n_samples=10_000).compute(model, words)

    print(result)
    print(f"score={result.score}")
    print(f"p_value={result.details['p_value']}")


if __name__ == "__main__":
    main()
