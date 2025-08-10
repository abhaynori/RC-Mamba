#!/usr/bin/env python3
"""Hyperparameter sweep runner for RC‑Mamba.

This script performs a simple grid search over a set of hyperparameters and
writes aggregate results to CSV files.  It is intentionally minimal and
intended as a starting point; for large sweeps consider using a job
scheduler or the `accelerate` library to distribute work across GPUs.

Example usage:
    python scripts/run_sweep.py --output-dir results --preset camera_ready
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hyperparameter sweeps.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write CSV results.")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Name of a predefined sweep preset (e.g. camera_ready).  If provided, overrides individual hyperparameters.",
    )
    parser.add_argument("--alpha", nargs="*", type=float, default=None, help="π‑DPO alpha values.")
    parser.add_argument("--beta", nargs="*", type=float, default=None, help="DPO temperature values.")
    parser.add_argument("--fsq", nargs="*", type=int, default=None, help="FSQ levels to sweep.")
    parser.add_argument("--K", nargs="*", type=int, default=None, help="Retrieval K values.")
    parser.add_argument("--lam", nargs="*", type=float, default=None, help="MMR lambda values.")
    parser.add_argument("--ent", nargs="*", type=float, default=None, help="Entropy thresholds for retrieval.")
    return parser.parse_args()


def get_preset(name: str) -> Dict[str, List[float]]:
    # Example camera_ready preset; adjust as needed
    if name == "camera_ready":
        return {
            "alpha": [2.0, 3.0, 4.0, 6.0],
            "beta": [0.05, 0.1, 0.2, 0.3],
            "fsq": [8, 16, 32],
            "K": [2, 4, 6, 8, 12],
            "lam": [0.4, 0.5, 0.6, 0.7],
            "ent": [5.0, 5.5, 6.0, 6.5],
        }
    raise ValueError(f"Unknown preset: {name}")


def run_dummy_experiment(params: Dict[str, float]) -> Dict[str, float]:
    """Simulate an experiment with the given hyperparameters.

    In a real implementation this would train or evaluate a model and return
    metrics.  Here we return random numbers for demonstration.
    """
    rng = np.random.default_rng(sum(hash(str(v)) for v in params.values()))
    return {
        "ppl": float(rng.uniform(5, 50)),
        "recall": float(rng.uniform(0.0, 1.0)),
        "metric": float(rng.uniform(0.0, 1.0)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Determine hyperparameter grid
    if args.preset is not None:
        grid = get_preset(args.preset)
    else:
        grid = {
            "alpha": args.alpha or [2.0],
            "beta": args.beta or [0.1],
            "fsq": args.fsq or [16],
            "K": args.K or [4],
            "lam": args.lam or [0.5],
            "ent": args.ent or [5.5],
        }
    # Cartesian product of hyperparameters
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        result = run_dummy_experiment(params)
        # Write to a CSV per hyperparameter category if needed; here we write a single summary file
        out_path = args.output_dir / "sweep_results.csv"
        header = False
        if not out_path.exists():
            header = True
        with out_path.open("a") as f:
            if header:
                f.write(",".join(keys + list(result.keys())) + "\n")
            f.write(",".join(str(params[k]) for k in keys) + "," + ",".join(str(result[k]) for k in result.keys()) + "\n")
    print(f"Sweep completed.  Results written to {out_path}")


if __name__ == "__main__":
    main()