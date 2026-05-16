#!/usr/bin/env python3
"""
threshold_sensitivity_analysis.py
Computes how the functional correctness metric changes as thresholds vary.
Produces results/threshold_sensitivity.json with a 7x6 grid of pass rates
and identifies the "robustness zone" where findings hold across threshold changes.
"""

import os
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
INPUT_FILE = RESULTS_DIR / "human_baseline_results.json"
OUTPUT_FILE = RESULTS_DIR / "threshold_sensitivity.json"

SECURITY_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
BP_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def load_tasks() -> list:
    if not INPUT_FILE.exists():
        print(f"WARNING: {INPUT_FILE} not found. Generating baseline...")
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "run_human_baseline.py")], check=False
        )
    if INPUT_FILE.exists():
        with open(INPUT_FILE) as f:
            data = json.load(f)
        return data.get("per_task", [])
    return []


def compute_pass_rate(tasks: list, sec_thresh: float, bp_thresh: float) -> float:
    """Compute functional correctness rate at given thresholds."""
    if not tasks:
        return 0.0
    passed = sum(
        1 for t in tasks
        if t.get("syntax_valid", False)
        and t.get("schema_valid", False)
        and t.get("security_score", 0.0) >= sec_thresh
        and t.get("bp_score", 0.0) >= bp_thresh
    )
    return round(passed / len(tasks), 4)


def find_robustness_zone(grid: list, sec_thresholds: list, bp_thresholds: list,
                          baseline_rate: float, tolerance: float = 0.05) -> dict:
    """
    Find the range of thresholds where the pass rate stays within ±tolerance
    of the baseline (default thresholds sec=0.5, bp=0.5).
    """
    baseline_sec_idx = sec_thresholds.index(0.5) if 0.5 in sec_thresholds else 3
    baseline_bp_idx = bp_thresholds.index(0.5) if 0.5 in bp_thresholds else 2

    robust_cells = []
    for i, sec in enumerate(sec_thresholds):
        for j, bp in enumerate(bp_thresholds):
            rate = grid[i][j]
            if abs(rate - baseline_rate) <= tolerance:
                robust_cells.append({"security_threshold": sec, "bp_threshold": bp, "rate": rate})

    # Find contiguous robustness zone
    robust_sec = sorted(set(c["security_threshold"] for c in robust_cells))
    robust_bp = sorted(set(c["bp_threshold"] for c in robust_cells))

    return {
        "baseline_rate": baseline_rate,
        "tolerance": tolerance,
        "n_robust_cells": len(robust_cells),
        "n_total_cells": len(sec_thresholds) * len(bp_thresholds),
        "robustness_fraction": round(len(robust_cells) / (len(sec_thresholds) * len(bp_thresholds)), 4),
        "robust_security_range": [min(robust_sec), max(robust_sec)] if robust_sec else [0.5, 0.5],
        "robust_bp_range": [min(robust_bp), max(robust_bp)] if robust_bp else [0.5, 0.5],
        "robust_cells": robust_cells,
    }


def sensitivity_to_threshold_changes(grid: np.ndarray,
                                      sec_thresholds: list,
                                      bp_thresholds: list) -> dict:
    """Compute partial derivatives (finite differences) to assess sensitivity."""
    grid = np.array(grid)

    # Sensitivity to security threshold (vertical gradient)
    d_security = np.diff(grid, axis=0) / np.diff(sec_thresholds)[:, None]
    # Sensitivity to bp threshold (horizontal gradient)
    d_bp = np.diff(grid, axis=1) / np.diff(bp_thresholds)[None, :]

    return {
        "max_security_sensitivity": round(float(np.abs(d_security).max()), 4),
        "mean_security_sensitivity": round(float(np.abs(d_security).mean()), 4),
        "max_bp_sensitivity": round(float(np.abs(d_bp).max()), 4),
        "mean_bp_sensitivity": round(float(np.abs(d_bp).mean()), 4),
        "dominant_threshold": "security" if float(np.abs(d_security).mean()) > float(np.abs(d_bp).mean()) else "bp",
        "grid_range": round(float(grid.max() - grid.min()), 4),
        "grid_std": round(float(grid.std()), 4),
    }


def compute_by_group(tasks: list, group_key: str,
                      sec_thresholds: list, bp_thresholds: list) -> dict:
    """Compute threshold grid for each subgroup."""
    from collections import defaultdict
    groups = defaultdict(list)
    for t in tasks:
        groups[t.get(group_key, "unknown")].append(t)

    result = {}
    for grp, grp_tasks in sorted(groups.items()):
        grid = [
            [compute_pass_rate(grp_tasks, sec, bp) for bp in bp_thresholds]
            for sec in sec_thresholds
        ]
        baseline = compute_pass_rate(grp_tasks, 0.5, 0.5)
        result[grp] = {
            "n": len(grp_tasks),
            "baseline_rate_at_0.5_0.5": baseline,
            "grid": grid,
            "sensitivity": sensitivity_to_threshold_changes(
                grid, sec_thresholds, bp_thresholds
            ),
        }
    return result


def main():
    print("=== Threshold Sensitivity Analysis ===")

    tasks = load_tasks()
    if not tasks:
        print("ERROR: No task data available.")
        return

    print(f"Loaded {len(tasks)} tasks.")
    print(f"Security thresholds: {SECURITY_THRESHOLDS}")
    print(f"BP thresholds: {BP_THRESHOLDS}")
    print(f"Grid size: {len(SECURITY_THRESHOLDS)} x {len(BP_THRESHOLDS)}\n")

    # Compute full grid
    grid = []
    for sec in SECURITY_THRESHOLDS:
        row = []
        for bp in BP_THRESHOLDS:
            rate = compute_pass_rate(tasks, sec, bp)
            row.append(rate)
        grid.append(row)

    # Baseline at default thresholds
    baseline_rate = compute_pass_rate(tasks, 0.5, 0.5)

    # Robustness zone
    robustness = find_robustness_zone(grid, SECURITY_THRESHOLDS, BP_THRESHOLDS, baseline_rate)

    # Sensitivity analysis
    sensitivity = sensitivity_to_threshold_changes(grid, SECURITY_THRESHOLDS, BP_THRESHOLDS)

    # By language and difficulty
    by_language = compute_by_group(tasks, "language", SECURITY_THRESHOLDS, BP_THRESHOLDS)
    by_difficulty = compute_by_group(tasks, "difficulty", SECURITY_THRESHOLDS, BP_THRESHOLDS)

    output = {
        "meta": {
            "n_tasks": len(tasks),
            "security_thresholds": SECURITY_THRESHOLDS,
            "bp_thresholds": BP_THRESHOLDS,
            "default_security_threshold": 0.5,
            "default_bp_threshold": 0.5,
        },
        "grid": {
            "rows": "security_threshold",
            "cols": "bp_threshold",
            "security_thresholds": SECURITY_THRESHOLDS,
            "bp_thresholds": BP_THRESHOLDS,
            "pass_rates": grid,
            "baseline_rate": baseline_rate,
        },
        "robustness_zone": robustness,
        "sensitivity": sensitivity,
        "by_language": by_language,
        "by_difficulty": by_difficulty,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    # Print grid
    print(f"Pass Rate Grid (rows=security_threshold, cols=bp_threshold):")
    header = "sec\\bp  " + "  ".join(f"{bp:.1f}" for bp in BP_THRESHOLDS)
    print(f"  {header}")
    for i, sec in enumerate(SECURITY_THRESHOLDS):
        row_str = "  ".join(f"{grid[i][j]:.3f}" for j in range(len(BP_THRESHOLDS)))
        print(f"  {sec:.1f}  | {row_str}")

    print(f"\nBaseline rate (sec=0.5, bp=0.5): {baseline_rate:.3f}")
    print(f"Robustness zone: {robustness['robustness_fraction']:.1%} of grid cells "
          f"within ±{robustness['tolerance']} of baseline")
    print(f"Robust security range: {robustness['robust_security_range']}")
    print(f"Robust BP range:       {robustness['robust_bp_range']}")
    print(f"\nSensitivity:")
    print(f"  Dominant threshold: {sensitivity['dominant_threshold']}")
    print(f"  Grid range (max-min): {sensitivity['grid_range']:.4f}")
    print(f"  Grid std dev:         {sensitivity['grid_std']:.4f}")

    print(f"\nResults written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
