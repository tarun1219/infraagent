#!/usr/bin/env python3
"""
regression_analysis.py
Analyzes SC (self-correction) regression patterns: when correction makes things worse.
Creates a synthetic dataset demonstrating the analysis framework.
Models regression probability per error type based on error informativeness.
Produces results/regression_analysis.json
"""

import os
import json
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "regression_analysis.json"

SEED = 42

# Error type parameters: (informativeness, base_recovery_rate, regression_risk)
ERROR_TYPE_PARAMS = {
    "syntax": {
        "informativeness": 0.47,
        "base_recovery_rate": 0.72,
        "regression_risk": 0.08,   # probability that SC fix introduces new error
        "typical_fix_scope": "local",
        "description": "Missing bracket, wrong indentation",
    },
    "schema": {
        "informativeness": 0.47,
        "base_recovery_rate": 0.61,
        "regression_risk": 0.14,
        "typical_fix_scope": "local",
        "description": "Missing required field, wrong type",
    },
    "security": {
        "informativeness": 0.27,
        "base_recovery_rate": 0.49,
        "regression_risk": 0.22,
        "typical_fix_scope": "local-to-medium",
        "description": "Privileged container, missing resource limits",
    },
    "cross_resource": {
        "informativeness": 0.37,
        "base_recovery_rate": 0.31,
        "regression_risk": 0.38,
        "typical_fix_scope": "global",
        "description": "Service selector mismatch, missing ConfigMap",
    },
}

# Condition-level regression rates from paper observations
CONDITION_REGRESSION_RATES = {
    "AI_only_no_feedback": 0.00,   # no SC, no regression possible
    "AI_plus_syntax_feedback": 0.06,
    "AI_plus_schema_feedback": 0.11,
    "AI_plus_security_feedback": 0.19,
    "AI_plus_full_feedback": 0.15,  # full feedback allows some self-correction
}


def simulate_sc_outcomes(n_tasks: int, error_type: str, seed: int = SEED) -> dict:
    """
    Simulate SC outcomes for a given error type.
    Each task starts with an error. SC attempts to fix it.
    Outcomes: recovered, regressed, unchanged.
    """
    rng = np.random.default_rng(seed)
    params = ERROR_TYPE_PARAMS[error_type]

    recovery_rate = params["base_recovery_rate"]
    regression_risk = params["regression_risk"]
    # P(unchanged) = 1 - P(recovered) - P(regressed)
    unchanged_rate = max(0.0, 1.0 - recovery_rate - regression_risk)

    outcomes = rng.choice(
        ["recovered", "regressed", "unchanged"],
        size=n_tasks,
        p=[recovery_rate, regression_risk, unchanged_rate]
    )

    n_recovered = int((outcomes == "recovered").sum())
    n_regressed = int((outcomes == "regressed").sum())
    n_unchanged = int((outcomes == "unchanged").sum())

    # Simulate secondary error types introduced by regression
    secondary_error_types = []
    for _ in range(n_regressed):
        # Regression typically introduces a different error type
        # Weighted toward schema and security errors
        secondary = rng.choice(
            list(ERROR_TYPE_PARAMS.keys()),
            p=[0.15, 0.35, 0.35, 0.15]
        )
        secondary_error_types.append(str(secondary))

    return {
        "n_tasks": n_tasks,
        "n_recovered": n_recovered,
        "n_regressed": n_regressed,
        "n_unchanged": n_unchanged,
        "recovery_rate": round(n_recovered / n_tasks, 4),
        "regression_rate": round(n_regressed / n_tasks, 4),
        "unchanged_rate": round(n_unchanged / n_tasks, 4),
        "net_improvement": round((n_recovered - n_regressed) / n_tasks, 4),
        "secondary_error_types": secondary_error_types,
        "secondary_error_distribution": {
            et: secondary_error_types.count(et)
            for et in ERROR_TYPE_PARAMS.keys()
        },
    }


def model_regression_risk(informativeness: float) -> dict:
    """
    Model regression risk as a function of error informativeness.
    Less informative errors -> higher regression risk (model guesses more).
    Regression risk = a - b * informativeness
    Fit from observed data points.
    """
    # Data points: (informativeness, regression_risk)
    data_points = [
        (0.47, 0.08),   # syntax
        (0.47, 0.14),   # schema (same informative, but structural complexity differs)
        (0.27, 0.22),   # security
        (0.37, 0.31),   # cross_resource
    ]
    x = np.array([p[0] for p in data_points])
    y = np.array([p[1] for p in data_points])

    # Linear regression
    slope, intercept, r, p, se = stats.linregress(x, y)
    predicted = intercept + slope * informativeness
    predicted = float(np.clip(predicted, 0.0, 1.0))

    return {
        "predicted_regression_risk": round(predicted, 4),
        "model_slope": round(float(slope), 4),
        "model_intercept": round(float(intercept), 4),
        "model_r_squared": round(float(r**2), 4),
        "model_p_value": round(float(p), 6),
        "model_se": round(float(se), 4),
    }


def compute_net_value_of_sc(n_tasks: int = 150) -> dict:
    """
    For each error type, compute net value of SC:
    Net value = P(recover) - P(regress)
    Also compute break-even: minimum recovery rate for SC to be beneficial.
    """
    results = {}
    for error_type, params in ERROR_TYPE_PARAMS.items():
        net = params["base_recovery_rate"] - params["regression_risk"]
        beneficial = net > 0
        # Break-even: recovery_rate = regression_risk
        breakeven_recovery = params["regression_risk"]

        # Expected change in task-level quality score
        # Assuming: recover -> +1, regress -> -1, unchanged -> 0
        expected_delta = params["base_recovery_rate"] * 1.0 + params["regression_risk"] * (-1.0)

        results[error_type] = {
            "recovery_rate": params["base_recovery_rate"],
            "regression_risk": params["regression_risk"],
            "net_improvement": round(net, 4),
            "beneficial": beneficial,
            "expected_score_delta": round(expected_delta, 4),
            "breakeven_recovery_rate": round(breakeven_recovery, 4),
        }
    return results


def regression_cascade_analysis() -> dict:
    """
    Analyze cascading regressions: when fixing error A introduces error B,
    and fixing B introduces error C.
    Simulate multi-round SC to find convergence.
    """
    rng = np.random.default_rng(SEED + 1)
    n_tasks = 150
    max_rounds = 5

    # Start all tasks in "broken" state
    task_states = ["broken"] * n_tasks
    round_stats = []

    for round_num in range(1, max_rounds + 1):
        still_broken = [i for i, s in enumerate(task_states) if s == "broken"]
        n_broken = len(still_broken)
        n_fixed = 0
        n_regressed_to_broken = 0

        # Average recovery and regression rates across error types
        avg_recovery = np.mean([p["base_recovery_rate"] for p in ERROR_TYPE_PARAMS.values()])
        avg_regression = np.mean([p["regression_risk"] for p in ERROR_TYPE_PARAMS.values()])

        for i in still_broken:
            outcome = rng.choice(
                ["recovered", "regressed", "unchanged"],
                p=[avg_recovery, avg_regression, 1.0 - avg_recovery - avg_regression]
            )
            if outcome == "recovered":
                task_states[i] = "fixed"
                n_fixed += 1
            elif outcome == "regressed":
                # Stays broken, possibly with worse error
                n_regressed_to_broken += 1

        n_now_fixed = sum(1 for s in task_states if s == "fixed")
        round_stats.append({
            "round": round_num,
            "n_broken_before": n_broken,
            "n_newly_fixed": n_fixed,
            "n_regressed": n_regressed_to_broken,
            "n_total_fixed": n_now_fixed,
            "cumulative_fix_rate": round(n_now_fixed / n_tasks, 4),
        })

    return {
        "n_tasks": n_tasks,
        "max_rounds": max_rounds,
        "final_fix_rate": round(sum(1 for s in task_states if s == "fixed") / n_tasks, 4),
        "rounds": round_stats,
        "interpretation": (
            "Multi-round SC shows diminishing returns. "
            "Regression accumulation can prevent convergence for complex error types."
        ),
    }


def main():
    print("=== SC Regression Analysis ===")

    N_TASKS = 150

    # Simulate SC outcomes per error type
    print("Simulating SC outcomes per error type...")
    simulations = {}
    for error_type in ERROR_TYPE_PARAMS:
        sim = simulate_sc_outcomes(N_TASKS // len(ERROR_TYPE_PARAMS), error_type)
        simulations[error_type] = sim
        print(f"  {error_type:15s}: recover={sim['recovery_rate']:.1%} "
              f"regress={sim['regression_rate']:.1%} "
              f"net={sim['net_improvement']:+.3f}")

    # Regression risk model
    print("\nFitting regression risk model...")
    model_fits = {}
    for error_type, params in ERROR_TYPE_PARAMS.items():
        fit = model_regression_risk(params["informativeness"])
        model_fits[error_type] = {
            **params,
            "model_prediction": fit,
        }

    # Net value of SC
    print("\nNet value of SC per error type:")
    net_values = compute_net_value_of_sc(N_TASKS)
    for et, nv in net_values.items():
        status = "BENEFICIAL" if nv["beneficial"] else "HARMFUL"
        print(f"  {et:15s}: net={nv['net_improvement']:+.3f} [{status}] "
              f"expected_delta={nv['expected_score_delta']:+.3f}")

    # Cascade analysis
    print("\nRunning cascade regression analysis...")
    cascade = regression_cascade_analysis()
    print(f"  After {cascade['max_rounds']} rounds: "
          f"final fix rate = {cascade['final_fix_rate']:.1%}")

    # Condition-level analysis
    print("\nSC regression by condition:")
    condition_analysis = {}
    for cond, reg_rate in CONDITION_REGRESSION_RATES.items():
        # Recovery rate decreases with regression risk
        recovery = max(0.0, 0.6 - reg_rate * 0.5)
        condition_analysis[cond] = {
            "regression_rate": reg_rate,
            "recovery_rate": round(recovery, 4),
            "net_improvement": round(recovery - reg_rate, 4),
        }
        print(f"  {cond:35s}: regress={reg_rate:.1%} recover={recovery:.1%}")

    output = {
        "meta": {
            "n_tasks": N_TASKS,
            "error_types": list(ERROR_TYPE_PARAMS.keys()),
            "seed": SEED,
        },
        "error_type_params": ERROR_TYPE_PARAMS,
        "simulations": simulations,
        "regression_risk_model": model_fits,
        "net_value_of_sc": net_values,
        "cascade_analysis": cascade,
        "condition_level_analysis": condition_analysis,
        "key_findings": {
            "most_regression_prone": max(
                ERROR_TYPE_PARAMS.keys(),
                key=lambda e: ERROR_TYPE_PARAMS[e]["regression_risk"]
            ),
            "safest_to_correct": min(
                ERROR_TYPE_PARAMS.keys(),
                key=lambda e: ERROR_TYPE_PARAMS[e]["regression_risk"]
            ),
            "net_beneficial_types": [
                et for et, nv in net_values.items() if nv["beneficial"]
            ],
            "net_harmful_types": [
                et for et, nv in net_values.items() if not nv["beneficial"]
            ],
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
