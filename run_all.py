import argparse
import os
import subprocess
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"

SRC = Path(__file__).resolve().parent / "src"

ANALYSIS_STEPS = [
    "01_build_charls.py",
    "02_build_elsa.py",
    "03_clustering.py",
    "04_survival_multistate.py",
    "05_prediction_sensitivity.py",
]

FIGURE_STEPS = [f"fig{k}.py" for k in range(1, 7)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", action="store_true", help="Also generate main figures 1-6")
    args = parser.parse_args()

    steps = ANALYSIS_STEPS + (FIGURE_STEPS if args.figures else [])
    for script in steps:
        res = subprocess.run([sys.executable, "-B", str(SRC / script)], check=False)
        if res.returncode != 0:
            sys.exit(res.returncode)


if __name__ == "__main__":
    main()
