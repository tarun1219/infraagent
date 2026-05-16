#!/usr/bin/env python3
"""
power_analysis.py
Statistical power analysis for the IaCBench study design.
Computes power of Wilcoxon signed-rank test for various effect sizes.
Produces results/power_analysis.json
"""

import os
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "power_analysis.json"

# Study parameters
N_TASKS = 150
ALPHA_UNCORRECTED = 0.05
N_COMPARISONS = 10  # Bonferroni correction
ALPHA_BONFERRONI = ALPHA_UNCORRECTED / N_COMPARISONS  # 0.005
POWER_TARGET = 0.80
N_SIMS = 10_000  # Monte Carlo simulations for power estimation

# Reported effect sizes from paper
REPORTED_EFFECT_SIZES = {
    "functional_correctness": 0.31,
    "security_score": 0.52,
}

# Effect sizes to sweep
DELTA_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def cliffs_delta_to_common_language(delta: float) -> float:
    """Convert Cliff's delta to common language effect size (P(X>Y))."""
    return (delta + 1) / 2


def simulate_wilcoxon_power(delta: float, n: int, alpha: float,
                             n_sims: int = N_SIMS,
                             seed: int = 42) -> float:
    """
    Estimate power of Wilcoxon signed-rank test via Monte Carlo simulation.
    delta: Cliff's delta (effect size)
    Simulates paired binary/continuous outcomes consistent with given delta.
    """
    rng = np.random.default_rng(seed)

    # Convert Cliff's delta to probability P(X > Y) for paired comparison
    # P(X > Y) = (delta + 1) / 2  -- common language effect size
    p_x_gt_y = (delta + 1) / 2

    reject_count = 0
    for _ in range(n_sims):
        # Simulate n paired observations
        # Each pair (x_i, y_i): differences drawn from a distribution
        # where P(d_i > 0) = p_x_gt_y
        # Use a shifted normal to achieve this
        shift = stats.norm.ppf(p_x_gt_y) * math.sqrt(2)
        differences = rng.normal(loc=shift, scale=1.0, size=n)

        # Wilcoxon signed-rank test on differences
        if len(np.unique(differences)) < 2:
            continue
        try:
            _, p_val = stats.wilcoxon(differences, alternative="two-sided")
            if p_val < alpha:
                reject_count += 1
        except Exception:
            pass

    return round(reject_count / n_sims, 4)


def analytical_power_normal_approx(delta: float, n: int, alpha: float) -> float:
    """
    Analytical approximation of Wilcoxon signed-rank power using normal approximation.
    Under H1, the test statistic is approximately normal.
    """
    # Convert Cliff's delta to rank-biserial correlation r
    r = delta  # Cliff's delta ≈ rank-biserial r for paired tests

    # Expected value of W under H1 (approximate)
    # Under H0: E[W] = n(n+1)/4, Var[W] = n(n+1)(2n+1)/24
    e_w_h0 = n * (n + 1) / 4
    var_w_h0 = n * (n + 1) * (2 * n + 1) / 24
    sd_w_h0 = math.sqrt(var_w_h0)

    # Under H1: shift in W proportional to effect size
    # Approximate noncentrality: lambda ≈ r * n * (n+1) / (2 * sd_w_h0)
    noncentrality = r * n * (n + 1) / (2 * sd_w_h0 + 1e-10)

    # Two-sided critical value
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Power = P(|Z + lambda| > z_crit)
    power = stats.norm.sf(z_crit - abs(noncentrality)) + stats.norm.cdf(-z_crit - abs(noncentrality))
    return round(float(np.clip(power, 0, 1)), 4)


def minimum_detectable_effect(n: int, alpha: float, power: float = 0.80) -> float:
    """
    Find minimum detectable Cliff's delta at given n, alpha, power.
    Uses binary search on the analytical approximation.
    """
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        p = analytical_power_normal_approx(mid, n, alpha)
        if p < power:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


def effect_size_interpretation(delta: float) -> str:
    if delta < 0.147:
        return "negligible"
    elif delta < 0.330:
        return "small"
    elif delta < 0.474:
        return "medium"
    else:
        return "large"


def main():
    print("=== Statistical Power Analysis ===")
    print(f"Study design: n={N_TASKS} tasks, alpha_Bonferroni={ALPHA_BONFERRONI:.4f}")
    print(f"Monte Carlo simulations: {N_SIMS:,}\n")

    # Sweep over effect sizes
    print("Computing power for delta sweep (Monte Carlo + analytical)...")
    power_sweep = []
    for delta in DELTA_SWEEP:
        print(f"  delta={delta:.1f} ...", end=" ", flush=True)
        mc_power = simulate_wilcoxon_power(delta, N_TASKS, ALPHA_BONFERRONI, n_sims=N_SIMS)
        analytical = analytical_power_normal_approx(delta, N_TASKS, ALPHA_BONFERRONI)
        print(f"MC={mc_power:.3f}  Analytical={analytical:.3f}")
        power_sweep.append({
            "cliff_delta": delta,
            "interpretation": effect_size_interpretation(delta),
            "p_x_gt_y": round((delta + 1) / 2, 4),
            "power_mc": mc_power,
            "power_analytical": analytical,
            "adequately_powered": mc_power >= POWER_TARGET,
        })

    # Reported effect sizes
    print("\nComputing power for reported effect sizes...")
    reported_power = {}
    for condition, delta in REPORTED_EFFECT_SIZES.items():
        mc_power = simulate_wilcoxon_power(delta, N_TASKS, ALPHA_BONFERRONI, n_sims=N_SIMS)
        analytical = analytical_power_normal_approx(delta, N_TASKS, ALPHA_BONFERRONI)
        reported_power[condition] = {
            "cliff_delta": delta,
            "interpretation": effect_size_interpretation(delta),
            "power_mc": mc_power,
            "power_analytical": analytical,
            "adequately_powered": mc_power >= POWER_TARGET,
        }
        print(f"  {condition}: delta={delta}, MC power={mc_power:.3f}, "
              f"Analytical={analytical:.3f}")

    # Minimum detectable effect
    print(f"\nComputing minimum detectable effect at n={N_TASKS}, "
          f"alpha={ALPHA_BONFERRONI:.4f}, power={POWER_TARGET}...")
    mde_bonferroni = minimum_detectable_effect(N_TASKS, ALPHA_BONFERRONI, POWER_TARGET)
    mde_uncorrected = minimum_detectable_effect(N_TASKS, ALPHA_UNCORRECTED, POWER_TARGET)
    print(f"  MDE (Bonferroni alpha={ALPHA_BONFERRONI:.4f}): delta={mde_bonferroni:.4f} "
          f"({effect_size_interpretation(mde_bonferroni)})")
    print(f"  MDE (uncorrected alpha={ALPHA_UNCORRECTED:.4f}): delta={mde_uncorrected:.4f} "
          f"({effect_size_interpretation(mde_uncorrected)})")

    # Sample size sensitivity
    sample_sizes = [50, 75, 100, 125, 150, 175, 200]
    delta_ref = 0.31  # functional correctness effect
    sample_size_power = []
    for n in sample_sizes:
        p = analytical_power_normal_approx(delta_ref, n, ALPHA_BONFERRONI)
        sample_size_power.append({
            "n": n,
            "power_at_delta_0.31": round(p, 4),
            "adequately_powered": p >= POWER_TARGET,
        })

    output = {
        "meta": {
            "n_tasks": N_TASKS,
            "alpha_uncorrected": ALPHA_UNCORRECTED,
            "n_comparisons_bonferroni": N_COMPARISONS,
            "alpha_bonferroni": ALPHA_BONFERRONI,
            "power_target": POWER_TARGET,
            "n_monte_carlo_sims": N_SIMS,
        },
        "power_by_effect_size": power_sweep,
        "reported_effect_sizes": reported_power,
        "minimum_detectable_effect": {
            "at_alpha_bonferroni": {
                "alpha": ALPHA_BONFERRONI,
                "n": N_TASKS,
                "power": POWER_TARGET,
                "mde_cliff_delta": mde_bonferroni,
                "interpretation": effect_size_interpretation(mde_bonferroni),
            },
            "at_alpha_uncorrected": {
                "alpha": ALPHA_UNCORRECTED,
                "n": N_TASKS,
                "power": POWER_TARGET,
                "mde_cliff_delta": mde_uncorrected,
                "interpretation": effect_size_interpretation(mde_uncorrected),
            },
        },
        "sample_size_sensitivity": {
            "delta_ref": delta_ref,
            "alpha": ALPHA_BONFERRONI,
            "results": sample_size_power,
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Summary Table ===")
    print(f"  {'Delta':>6}  {'Interp':>12}  {'MC Power':>9}  {'Adequate':>9}")
    print(f"  {'-'*50}")
    for row in power_sweep:
        print(f"  {row['cliff_delta']:>6.1f}  {row['interpretation']:>12s}  "
              f"{row['power_mc']:>9.3f}  {'YES' if row['adequately_powered'] else 'NO':>9}")

    print(f"\nResults written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
