"""CrowS-Pairs Score via the public fairLLMs API."""

from fairLLMs.datasets import CrowSPairs
from fairLLMs.metrics import CrowSPairsScore
from fairLLMs.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="mlm")
    result = CrowSPairsScore().compute(
        model=model,
        dataset=CrowSPairs(n_max=32),
    )
    print(result)
    print(f"score={result.score}")
    if result.by_category:
        print("by_category:", result.by_category)


if __name__ == "__main__":
    main()
