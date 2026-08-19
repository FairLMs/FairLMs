"""CrowS-Pairs Score via the public fairlms API."""

from fairlms.datasets import CrowSPairs
from fairlms.metrics import CrowSPairsScore
from fairlms.models import HuggingFaceModel


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
