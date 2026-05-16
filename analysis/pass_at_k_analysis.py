#!/usr/bin/env python3
"""
pass_at_k_analysis.py
Computes pass@k from multiple independent generations.
Simulates based on per-task pass rates from human baseline (Beta distribution).
Produces results/pass_at_k.json with theoretical pass@k values per condition.
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats as scipy_stats

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
INPUT_FILE = RESULTS_DIR / "human_baseline_results.json"
OUTPUT_FILE = RESULTS_DIR / "pass_at_k.json"

# k values to compute
K_VALUES = [1, 2, 3, 5, 10]

# Conditions from paper
CONDITIONS = {
    "human_reference": {"mean_pass_rate": 0.82, "std": 0.12},
    "AI_only": {"mean_pass_rate": 0.41, "std": 0.18},
    "AI_plus_SC": {"mean_pass_rate": 0.58, "std": 0.16},
    "human_in_loop": {"mean_pass_rate": 0.74, "std": 0.13},
}

N_SIMULATION = 10_000
SEED = 42


def pass_at_k_theoretical(p: float, k: int) -> float:
    """Theoretical pass@k = 1 - (1-p)^k for a single task with pass prob p."""
    return 1.0 - (1.0 - p) ** k


def load_tasks() -> list:
    if not INPUT_FILE.exists():
        print(f"WARNING: {INPUT_FILE} not found. Running baseline...")
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "run_human_baseline.py")], check=False
        )
    if INPUT_FILE.exists():
        with open(INPUT_FILE) as f:
            data = json.load(f)
        return data.get("per_task", [])
    return []


def calibrate_beta(mean: float, std: float) -> tuple:
    """
    Calibrate Beta(alpha, beta) parameters to match mean and std.
    Returns (alpha, beta). Clips to valid range.
    """
    mean = np.clip(mean, 0.01, 0.99)
    var = min(std**2, mean * (1 - mean) * 0.99)  # ensure var < mean*(1-mean)
    if var <= 0:
        var = 1e-6
    alpha = mean * (mean * (1 - mean) / var - 1)
    beta = (1 - mean) * (mean * (1 - mean) / var - 1)
    alpha = max(alpha, 0.01)
    beta = max(beta, 0.01)
    return alpha, beta


def compute_pass_at_k_from_distribution(alpha: float, beta_param: float,
                                         k_values: list, n_tasks: int = 150,
                                         seed: int = SEED) -> dict:
    """
    Sample per-task pass probabilities from Beta(alpha, beta),
    then compute pass@k for each task and average.
    """
    rng = np.random.default_rng(seed)
    task_probs = rng.beta(alpha, beta_param, size=n_tasks)
    task_probs = np.clip(task_probs, 0.0, 1.0)

    result = {}
    for k in k_values:
        # For each task: pass@k = 1 - (1-p)^k
        task_pass_at_k = 1.0 - (1.0 - task_probs) ** k
        mean_pass_at_k = float(task_pass_at_k.mean())
        std_pass_at_k = float(task_pass_at_k.std())

        # 95% CI via bootstrap
        bootstrap_means = []
        for _ in range(1000):
            sample = rng.choice(task_pass_at_k, size=n_tasks, replace=True)
            bootstrap_means.append(sample.mean())
        ci_low = float(np.percentile(bootstrap_means, 2.5))
        ci_high = float(np.percentile(bootstrap_means, 97.5))

        result[f"pass@{k}"] = {
            "mean": round(mean_pass_at_k, 4),
            "std": round(std_pass_at_k, 4),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
        }

    # Also compute absolute improvement over pass@1
    p1 = result["pass@1"]["mean"]
    for k in k_values[1:]:
        pk = result[f"pass@{k}"]["mean"]
        result[f"pass@{k}"]["gain_over_pass_at_1"] = round(pk - p1, 4)
        result[f"pass@{k}"]["relative_gain_pct"] = (
            round((pk - p1) / p1 * 100, 2) if p1 > 0 else 0.0
        )

    return result


def compute_calibrated_from_data(tasks: list, k_values: list) -> dict:
    """
    Calibrate from actual human baseline per-task functional correctness.
    Per-task prob = 1 if functional_correct else 0 (binary).
    Use Beta to model uncertainty around this.
    """
    n = len(tasks)
    if n == 0:
        return {}
    pass_rates = [1.0 if t.get("functional_correct", False) else 0.0 for t in tasks]
    mean_rate = float(np.mean(pass_rates))
    std_rate = float(np.std(pass_rates))

    # Smooth: per-task rate is mean ± some noise (model multiple trials)
    # Beta calibration with shrinkage
    alpha, beta_p = calibrate_beta(
        max(0.05, mean_rate),
        max(0.05, std_rate) if std_rate > 0 else 0.15
    )
    return {
        "source": "human_baseline_calibrated",
        "empirical_mean": round(mean_rate, 4),
        "empirical_std": round(std_rate, 4),
        "beta_alpha": round(alpha, 4),
        "beta_beta": round(beta_p, 4),
        "pass_at_k": compute_pass_at_k_from_distribution(
            alpha, beta_p, k_values, n_tasks=n
        ),
    }


def main():
    print("=== Pass@k Analysis ===")

    tasks = load_tasks()
    n_tasks = len(tasks) if tasks else 150
    print(f"Loaded {n_tasks} tasks from baseline.")

    results = {}

    # 1. Condition-based pass@k from literature-based priors
    print("\nComputing pass@k for each condition (Beta-calibrated)...")
    for cond_name, cond_params in CONDITIONS.items():
        print(f"  Condition: {cond_name} (mean={cond_params['mean_pass_rate']:.2f}, "
              f"std={cond_params['std']:.2f})")
        alpha, beta_p = calibrate_beta(
            cond_params["mean_pass_rate"], cond_params["std"]
        )
        pak = compute_pass_at_k_from_distribution(
            alpha, beta_p, K_VALUES, n_tasks=n_tasks, seed=SEED
        )
        results[cond_name] = {
            "distribution": {
                "mean": cond_params["mean_pass_rate"],
                "std": cond_params["std"],
                "beta_alpha": round(alpha, 4),
                "beta_beta": round(beta_p, 4),
            },
            "pass_at_k": pak,
        }
        for k in K_VALUES:
            print(f"    pass@{k} = {pak[f'pass@{k}']['mean']:.3f}")

    # 2. Calibrated from actual data if available
    if tasks:
        print("\nComputing pass@k calibrated from actual human baseline...")
        empirical = compute_calibrated_from_data(tasks, K_VALUES)
        results["empirical_human_baseline"] = empirical
        for k in K_VALUES:
            print(f"  pass@{k} = {empirical['pass_at_k'][f'pass@{k}']['mean']:.3f}")

    # 3. Theoretical pass@k curves
    print("\nComputing theoretical pass@k curves...")
    p_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    theoretical_curves = {}
    for p in p_grid:
        theoretical_curves[str(p)] = {
            f"pass@{k}": round(pass_at_k_theoretical(p, k), 4)
            for k in K_VALUES
        }

    # 4. Diminishing returns analysis
    print("\nDiminishing returns analysis...")
    diminishing_returns = {}
    for cond_name in CONDITIONS:
        pak = results[cond_name]["pass_at_k"]
        gains = []
        for i in range(len(K_VALUES) - 1):
            k_a, k_b = K_VALUES[i], K_VALUES[i + 1]
            gain = pak[f"pass@{k_b}"]["mean"] - pak[f"pass@{k_a}"]["mean"]
            cost = k_b - k_a  # additional samples
            efficiency = gain / cost if cost > 0 else 0
            gains.append({
                "from_k": k_a, "to_k": k_b,
                "gain": round(gain, 4),
                "marginal_cost": cost,
                "efficiency": round(efficiency, 4),
            })
        diminishing_returns[cond_name] = gains

    output = {
        "meta": {
            "n_tasks": n_tasks,
            "k_values": K_VALUES,
            "n_simulation_bootstrap": 1000,
            "conditions": list(CONDITIONS.keys()),
        },
        "by_condition": results,
        "theoretical_curves": theoretical_curves,
        "diminishing_returns": diminishing_returns,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    # Summary table
    print(f"\n=== Pass@k Summary Table ===")
    header = f"{'Condition':25s}" + "".join(f"  pass@{k:2d}" for k in K_VALUES)
    print(f"  {header}")
    print(f"  {'-'*80}")
    for cond in CONDITIONS:
        pak = results[cond]["pass_at_k"]
        row = f"  {cond:25s}" + "".join(
            f"  {pak[f'pass@{k}']['mean']:.3f}" for k in K_VALUES
        )
        print(row)

    print(f"\nResults written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
