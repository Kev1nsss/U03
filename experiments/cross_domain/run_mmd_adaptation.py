"""MMD-based cross-condition adaptation experiment placeholder.

MMD encourages source-domain and target-domain learned features to have similar
distributions. The target domain uses only X during training, not y.
"""


def main() -> None:
    raise NotImplementedError(
        "MMD adaptation is not implemented yet. "
        "Next step: implement condition splits and source-only baseline first."
    )


if __name__ == "__main__":
    main()
