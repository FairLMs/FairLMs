"""WEAT via the public fairlms API (one Caliskan-style test).

Shows the preferred shape: configuration in the constructor, data as a
validated container passed positionally to ``compute``.
"""

from fairlms.data import weat_c1
from fairlms.metrics import WEAT
from fairlms.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="encoder")

    result = WEAT(pooling="mean", n_samples=10_000).compute(model, weat_c1)

    print(result)
    print(f"score={result.score}")
    print(f"p_value={result.details['p_value']}")


if __name__ == "__main__":
    main()
