#!/usr/bin/env python3
"""One‑click driver for the RC‑Mamba project.

This script orchestrates the typical workflow for running experiments and
generating a paper draft.  It performs hyperparameter sweeps (unless
`--skip-sweeps` is passed), aggregates results, writes LaTeX tables, and
produces plots.  This is a minimal stub; you can extend it to call the
functions provided in the `paper` folder (e.g. results_table.py, results_grid.py,
capture_versions.py) and to build the PDF automatically.
"""

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all steps for RC‑Mamba experiments.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for sweeps.")
    parser.add_argument("--preset", type=str, default="camera_ready", help="Hyperparameter sweep preset.")
    parser.add_argument("--skip-sweeps", action="store_true", help="Skip running sweeps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    # Step 1: run sweeps
    if not args.skip_sweeps:
        sweep_script = project_root / "scripts" / "run_sweep.py"
        subprocess.run([
            "python", str(sweep_script),
            "--output-dir", str(results_dir),
            "--preset", args.preset,
        ], check=True)
    # Step 2: generate tables and plots (placeholders)
    # You can plug in the paper helper scripts here.  For now we just print a message.
    print("Generating tables and plots... (not implemented in this stub)")
    # Step 3: build PDF (requires LaTeX environment)
    print("Build the paper using the Makefile in the paper directory.")


if __name__ == "__main__":
    main()