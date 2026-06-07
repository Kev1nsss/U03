import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from unit03_soft_sensor.config import ExperimentConfig
from experiments.cross_domain.common import DEFAULT_TARGET_CONDITION, make_condition_output_dirs, save_condition_overview


def main():
    config = ExperimentConfig()
    dirs = make_condition_output_dirs(PROJECT_ROOT)
    table = save_condition_overview(config, DEFAULT_TARGET_CONDITION, dirs)
    print("Saved clear condition figures to:", dirs["figures"])
    print("Saved condition intervals to:", dirs["metrics"] / "condition_intervals.csv")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
