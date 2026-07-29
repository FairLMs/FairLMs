"""WEAT via the public fairLLMs API (one Caliskan-style test)."""

from fairLLMs.definition.encoder_only.intrinsic_bias.similarity_based.weat.data import C1
from fairLLMs.metrics import WEAT
from fairLLMs.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="encoder")
    result = WEAT().compute(
        model=model,
        T1_terms=C1["t1"][:8],
        T2_terms=C1["t2"][:8],
        A_terms=C1["a1"][:8],
        B_terms=C1["a2"][:8],
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
