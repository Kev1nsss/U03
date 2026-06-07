import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from experiments.cross_domain.common import build_parser, run_cross_domain_experiment


def main():
    parser = build_parser("source_only", default_alignment_weight=0.0)
    args = parser.parse_args()
    run_cross_domain_experiment(PROJECT_ROOT, "source_only", args)


if __name__ == "__main__":
    main()
