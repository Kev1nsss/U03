"""CORAL-based cross-condition adaptation experiment placeholder.

CORAL aligns the covariance statistics of source-domain and target-domain
features. The target domain uses only X during training, not y.
"""


def main() -> None:
    raise NotImplementedError(
        "CORAL adaptation is not implemented yet. "
        "Next step: implement condition splits and source-only baseline first."
    )


if __name__ == "__main__":
    main()
