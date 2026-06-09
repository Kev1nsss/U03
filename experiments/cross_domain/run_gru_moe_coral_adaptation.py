import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from experiments.cross_domain.sequence_common import build_gru_moe_parser, run_gru_moe_cross_domain_experiment


def main():
    parser = build_gru_moe_parser("gru_moe_coral", default_alignment_weight=0.1)
    args = parser.parse_args()
    run_gru_moe_cross_domain_experiment(PROJECT_ROOT, "gru_moe_coral", args)


if __name__ == "__main__":
    main()
