"""Source-only cross-condition baseline.

This script is a placeholder for the first DVPF-inspired experiment:
train a regressor on labeled source operating conditions and evaluate it on a
held-out target condition. Target labels should be used only for evaluation.
"""


def main() -> None:
    raise NotImplementedError(
        "Source-only cross-domain training is not implemented yet. "
        "Next step: define operating-condition intervals and build source/target splits."
    )


if __name__ == "__main__":
    main()
