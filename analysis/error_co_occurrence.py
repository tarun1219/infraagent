#!/usr/bin/env python3
"""
error_co_occurrence.py
Analyzes which failure modes tend to co-occur using phi coefficient (binary correlation).
Produces results/error_co_occurrence.json with the correlation matrix.
"""

import os
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
INPUT_FILE = RESULTS_DIR / "human_baseline_results.json"
OUTPUT_FILE = RESULTS_DIR / "error_co_occurrence.json"

# Failure mode columns derived from baseline results
FAILURE_MODES = [
    "syntax_fail",
    "schema_fail",
    "security_fail",
    "bp_fail",
    "functional_fail",
]

SECURITY_THRESHOLD = 0.5
BP_THRESHOLD = 0.5


def load_tasks() -> list:
    if not INPUT_FILE.exists():
        print(f"WARNING: {INPUT_FILE} not found. Running baseline first...")
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "run_human_baseline.py")], check=False
        )
    if INPUT_FILE.exists():
        with open(INPUT_FILE) as f:
            data = json.load(f)
        return data.get("per_task", [])
    return []


def extract_failure_vectors(tasks: list) -> dict:
    """Convert tasks to binary failure indicator vectors."""
    vectors = {mode: [] for mode in FAILURE_MODES}
    for t in tasks:
        syntax_fail = not t.get("syntax_valid", True)
        schema_fail = not t.get("schema_valid", True)
        security_fail = t.get("security_score", 1.0) < SECURITY_THRESHOLD
        bp_fail = t.get("bp_score", 1.0) < BP_THRESHOLD
        functional_fail = not t.get("functional_correct", True)

        vectors["syntax_fail"].append(int(syntax_fail))
        vectors["schema_fail"].append(int(schema_fail))
        vectors["security_fail"].append(int(security_fail))
        vectors["bp_fail"].append(int(bp_fail))
        vectors["functional_fail"].append(int(functional_fail))

    return {k: np.array(v) for k, v in vectors.items()}


def phi_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Compute phi coefficient (binary Pearson correlation) between two binary arrays."""
    # Phi = (n11*n00 - n10*n01) / sqrt(n1.*n0.*n.1*n.0)
    n = len(x)
    if n == 0:
        return 0.0

    n11 = float(np.sum((x == 1) & (y == 1)))
    n10 = float(np.sum((x == 1) & (y == 0)))
    n01 = float(np.sum((x == 0) & (y == 1)))
    n00 = float(np.sum((x == 0) & (y == 0)))

    n1x = n11 + n10  # row sum x=1
    n0x = n01 + n00  # row sum x=0
    nx1 = n11 + n01  # col sum y=1
    nx0 = n10 + n00  # col sum y=0

    denom = np.sqrt(n1x * n0x * nx1 * nx0)
    if denom == 0:
        return 0.0

    phi = (n11 * n00 - n10 * n01) / denom
    return round(float(phi), 4)


def chi2_p_value(x: np.ndarray, y: np.ndarray) -> float:
    """Compute chi-squared p-value for two binary variables."""
    from scipy.stats import chi2_contingency
    n11 = int(np.sum((x == 1) & (y == 1)))
    n10 = int(np.sum((x == 1) & (y == 0)))
    n01 = int(np.sum((x == 0) & (y == 1)))
    n00 = int(np.sum((x == 0) & (y == 0)))

    table = np.array([[n00, n01], [n10, n11]])
    if table.sum() == 0 or 0 in table.sum(axis=0) or 0 in table.sum(axis=1):
        return 1.0
    try:
        _, p, _, _ = chi2_contingency(table, correction=False)
        return round(float(p), 6)
    except Exception:
        return 1.0


def build_correlation_matrix(vectors: dict) -> tuple:
    """Build phi coefficient matrix and p-value matrix."""
    modes = list(vectors.keys())
    n = len(modes)
    phi_matrix = np.eye(n)
    pval_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                phi_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            elif i < j:
                phi = phi_coefficient(vectors[modes[i]], vectors[modes[j]])
                p = chi2_p_value(vectors[modes[i]], vectors[modes[j]])
                phi_matrix[i, j] = phi
                phi_matrix[j, i] = phi
                pval_matrix[i, j] = p
                pval_matrix[j, i] = p

    return modes, phi_matrix, pval_matrix


def find_strong_associations(modes: list, phi_matrix: np.ndarray,
                              pval_matrix: np.ndarray,
                              phi_threshold: float = 0.3,
                              alpha: float = 0.05) -> list:
    """Find pairs with strong and significant associations."""
    associations = []
    n = len(modes)
    for i in range(n):
        for j in range(i + 1, n):
            phi = phi_matrix[i, j]
            p = pval_matrix[i, j]
            if abs(phi) >= phi_threshold and p < alpha:
                strength = "strong" if abs(phi) >= 0.5 else "moderate"
                direction = "positive" if phi > 0 else "negative"
                associations.append({
                    "mode_a": modes[i],
                    "mode_b": modes[j],
                    "phi": round(float(phi), 4),
                    "p_value": round(float(p), 6),
                    "significant": True,
                    "strength": strength,
                    "direction": direction,
                })
    return sorted(associations, key=lambda x: abs(x["phi"]), reverse=True)


def failure_co_occurrence_counts(vectors: dict) -> dict:
    """Count how often each pair of failures co-occurs."""
    modes = list(vectors.keys())
    counts = {}
    n = len(vectors[modes[0]])

    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            a, b = modes[i], modes[j]
            both = int(np.sum((vectors[a] == 1) & (vectors[b] == 1)))
            a_only = int(np.sum((vectors[a] == 1) & (vectors[b] == 0)))
            b_only = int(np.sum((vectors[a] == 0) & (vectors[b] == 1)))
            neither = int(np.sum((vectors[a] == 0) & (vectors[b] == 0)))

            counts[f"{a}_AND_{b}"] = {
                "both_fail": both,
                "a_only_fail": a_only,
                "b_only_fail": b_only,
                "both_pass": neither,
                "conditional_p_a_given_b": round(both / (both + b_only), 4) if (both + b_only) > 0 else 0.0,
                "conditional_p_b_given_a": round(both / (both + a_only), 4) if (both + a_only) > 0 else 0.0,
            }
    return counts


def failure_rate_summary(vectors: dict) -> dict:
    """Overall failure rate per mode."""
    return {
        mode: {
            "n_fail": int(vec.sum()),
            "n_pass": int((vec == 0).sum()),
            "fail_rate": round(float(vec.mean()), 4),
        }
        for mode, vec in vectors.items()
    }


def main():
    print("=== Error Co-occurrence Analysis ===")

    tasks = load_tasks()
    if not tasks:
        print("ERROR: No task data available.")
        return

    print(f"Loaded {len(tasks)} tasks.")

    # Build failure vectors
    vectors = extract_failure_vectors(tasks)

    # Failure rate summary
    rate_summary = failure_rate_summary(vectors)
    print("\nFailure rates:")
    for mode, stats_d in rate_summary.items():
        print(f"  {mode:20s}: {stats_d['fail_rate']:.1%} ({stats_d['n_fail']}/{len(tasks)})")

    # Correlation matrix
    modes, phi_matrix, pval_matrix = build_correlation_matrix(vectors)

    # Co-occurrence counts
    co_counts = failure_co_occurrence_counts(vectors)

    # Strong associations
    associations = find_strong_associations(modes, phi_matrix, pval_matrix)

    print(f"\nStrong associations (|phi| >= 0.3, p < 0.05):")
    if associations:
        for a in associations:
            print(f"  {a['mode_a']:20s} <-> {a['mode_b']:20s}: "
                  f"phi={a['phi']:+.3f} p={a['p_value']:.4f} [{a['strength']} {a['direction']}]")
    else:
        print("  None found at threshold.")

    output = {
        "meta": {
            "n_tasks": len(tasks),
            "failure_modes": FAILURE_MODES,
            "security_threshold": SECURITY_THRESHOLD,
            "bp_threshold": BP_THRESHOLD,
        },
        "failure_rates": rate_summary,
        "phi_matrix": {
            "modes": modes,
            "matrix": phi_matrix.round(4).tolist(),
            "interpretation": "phi coefficient: +1=always co-occur, 0=independent, -1=never co-occur",
        },
        "p_value_matrix": {
            "modes": modes,
            "matrix": pval_matrix.round(6).tolist(),
        },
        "strong_associations": associations,
        "co_occurrence_counts": co_counts,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    # Print phi matrix
    print(f"\nPhi Coefficient Matrix:")
    header = f"{'':20s}" + "".join(f"  {m[:8]:>8s}" for m in modes)
    print(f"  {header}")
    for i, mode_i in enumerate(modes):
        row = f"  {mode_i:20s}" + "".join(f"  {phi_matrix[i,j]:+8.3f}" for j in range(len(modes)))
        print(row)

    print(f"\nResults written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
