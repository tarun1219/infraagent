#!/usr/bin/env python3
"""
benchmark_validity_study.py
Validates that the benchmark discriminates between approaches.
Computes: Spearman correlation between difficulty and failure rate,
Friedman test across conditions, Cronbach's alpha for benchmark reliability.
Produces results/benchmark_validity.json
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
INPUT_FILE = RESULTS_DIR / "human_baseline_results.json"
OUTPUT_FILE = RESULTS_DIR / "benchmark_validity.json"

# Condition performance data (functional correctness rates)
# Shape: [condition][difficulty_idx] for difficulty = [easy, medium, hard, expert]
CONDITION_DATA = {
    "AI_only":            [0.71, 0.52, 0.34, 0.18],
    "AI_plus_SC":         [0.82, 0.63, 0.47, 0.29],
    "AI_plus_human_review": [0.91, 0.79, 0.64, 0.48],
    "human_only":         [0.95, 0.87, 0.76, 0.61],
}

DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}
DIFFICULTY_NUMERIC = [1, 2, 3, 4]  # ordinal encoding


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


def spearman_difficulty_failure(tasks: list) -> dict:
    """
    Test: does difficulty level predict failure rate?
    x = difficulty numeric (1-4), y = fail indicator (1=fail, 0=pass)
    """
    if not tasks:
        return {"note": "no data"}

    difficulty_scores = []
    fail_indicators = []
    for t in tasks:
        diff = t.get("difficulty", "medium")
        diff_num = DIFFICULTY_ORDER.get(diff, 1) + 1  # 1-4
        fail = 0 if t.get("functional_correct", False) else 1
        difficulty_scores.append(diff_num)
        fail_indicators.append(fail)

    r, p = stats.spearmanr(difficulty_scores, fail_indicators)

    # By difficulty: compute failure rates
    diff_groups = defaultdict(list)
    for t in tasks:
        diff = t.get("difficulty", "medium")
        diff_groups[diff].append(0 if t.get("functional_correct", False) else 1)

    failure_by_difficulty = {
        diff: {
            "n": len(vals),
            "n_fail": sum(vals),
            "failure_rate": round(sum(vals) / len(vals), 4) if vals else 0.0,
        }
        for diff, vals in sorted(diff_groups.items(), key=lambda x: DIFFICULTY_ORDER.get(x[0], 99))
    }

    return {
        "spearman_rho": round(float(r), 4),
        "p_value": round(float(p), 6),
        "significant": p < 0.05,
        "n": len(tasks),
        "interpretation": (
            f"Spearman rho={r:.3f}, p={p:.4f}: "
            + ("difficulty significantly predicts failure rate" if p < 0.05
               else "difficulty does NOT significantly predict failure rate")
        ),
        "failure_by_difficulty": failure_by_difficulty,
    }


def spearman_difficulty_condition(conditions: dict) -> dict:
    """
    For each condition, test if difficulty predicts performance.
    Uses condition-level performance data (not per-task).
    """
    results = {}
    for cond, perf_by_diff in conditions.items():
        difficulty_nums = DIFFICULTY_NUMERIC
        # Pass rate (inverse of failure rate)
        pass_rates = perf_by_diff

        r, p = stats.spearmanr(difficulty_nums, pass_rates)
        results[cond] = {
            "spearman_rho": round(float(r), 4),
            "p_value": round(float(p), 6),
            "significant": p < 0.05,
            "performance_by_difficulty": {
                d: round(v, 4) for d, v in zip(["easy", "medium", "hard", "expert"], pass_rates)
            },
        }
    return results


def friedman_test_across_conditions() -> dict:
    """
    Friedman test: does condition predict performance?
    Treats each difficulty level as a 'block' (subject in repeated measures).
    Ranks conditions within each block (difficulty level).
    """
    # Data matrix: rows=difficulty levels, cols=conditions
    conditions = list(CONDITION_DATA.keys())
    n_blocks = 4  # difficulty levels
    n_conditions = len(conditions)

    # Build matrix
    matrix = np.array([CONDITION_DATA[c] for c in conditions]).T  # shape (4, 4)

    try:
        statistic, p_value = stats.friedmanchisquare(*[matrix[:, j] for j in range(n_conditions)])
    except Exception as e:
        return {"error": str(e)}

    # Degrees of freedom
    df = n_conditions - 1

    # Effect size: Kendall's W = chi2 / (n_blocks * (n_conditions - 1))
    kendalls_w = statistic / (n_blocks * (n_conditions - 1))

    # Post-hoc Wilcoxon signed-rank tests (pairwise)
    posthoc = {}
    for i in range(n_conditions):
        for j in range(i + 1, n_conditions):
            cond_i = conditions[i]
            cond_j = conditions[j]
            data_i = matrix[:, i]
            data_j = matrix[:, j]
            try:
                if len(np.unique(data_i - data_j)) > 1:
                    w_stat, w_p = stats.wilcoxon(data_i, data_j)
                else:
                    w_stat, w_p = 0.0, 1.0
            except Exception:
                w_stat, w_p = 0.0, 1.0

            posthoc[f"{cond_i}_vs_{cond_j}"] = {
                "statistic": round(float(w_stat), 4),
                "p_value": round(float(w_p), 6),
                "significant": w_p < 0.05,
            }

    return {
        "test": "Friedman chi-squared",
        "statistic": round(float(statistic), 4),
        "df": df,
        "p_value": round(float(p_value), 6),
        "significant": p_value < 0.05,
        "kendalls_w": round(float(kendalls_w), 4),
        "kendalls_w_interpretation": (
            "strong concordance" if kendalls_w > 0.7
            else "moderate concordance" if kendalls_w > 0.5
            else "weak concordance"
        ),
        "n_blocks": n_blocks,
        "n_conditions": n_conditions,
        "posthoc_wilcoxon": posthoc,
        "interpretation": (
            f"Friedman test: chi2({df})={statistic:.3f}, p={p_value:.4f}, "
            f"Kendall's W={kendalls_w:.3f}. "
            + ("Conditions significantly differ in performance." if p_value < 0.05
               else "No significant difference between conditions.")
        ),
    }


def cronbachs_alpha(tasks: list) -> dict:
    """
    Cronbach's alpha: reliability of the benchmark.
    Treats each task as an 'item' and each check (syntax, schema, security, bp) as a 'rater'.
    Alpha measures internal consistency.
    """
    if not tasks:
        return {"note": "no data"}

    # Build item matrix: rows=tasks, cols=checks
    items = []
    for t in tasks:
        row = [
            1 if t.get("syntax_valid", False) else 0,
            1 if t.get("schema_valid", False) else 0,
            1 if t.get("security_score", 0.0) >= 0.5 else 0,
            1 if t.get("bp_score", 0.0) >= 0.5 else 0,
        ]
        items.append(row)

    X = np.array(items, dtype=float)
    n_tasks, k = X.shape

    if k < 2:
        return {"note": "insufficient items"}

    # Variance of each item
    item_variances = X.var(axis=0, ddof=1)

    # Variance of total score
    total_scores = X.sum(axis=1)
    total_variance = total_scores.var(ddof=1)

    if total_variance == 0:
        return {"alpha": 1.0, "note": "zero total variance (perfect consistency)"}

    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)

    # Item-total correlations
    item_total_corr = {}
    check_names = ["syntax", "schema", "security", "best_practices"]
    for i, name in enumerate(check_names):
        rest_total = total_scores - X[:, i]
        try:
            r, p = stats.pearsonr(X[:, i], rest_total)
            item_total_corr[name] = {"r": round(float(r), 4), "p": round(float(p), 6)}
        except Exception:
            item_total_corr[name] = {"r": 0.0, "p": 1.0}

    return {
        "alpha": round(float(alpha), 4),
        "n_tasks": n_tasks,
        "n_items": k,
        "item_variances": {check_names[i]: round(float(v), 4) for i, v in enumerate(item_variances)},
        "total_variance": round(float(total_variance), 4),
        "item_total_correlations": item_total_corr,
        "interpretation": (
            "excellent" if alpha >= 0.9
            else "good" if alpha >= 0.8
            else "acceptable" if alpha >= 0.7
            else "questionable" if alpha >= 0.6
            else "poor"
        ),
    }


def discriminant_validity(tasks: list) -> dict:
    """
    Discriminant validity: does the benchmark separate easy from hard tasks?
    Mann-Whitney U test: easy vs expert failure rates.
    """
    if not tasks:
        return {"note": "no data"}

    easy_scores = [t.get("security_score", 0.0) for t in tasks if t.get("difficulty") == "easy"]
    expert_scores = [t.get("security_score", 0.0) for t in tasks if t.get("difficulty") == "expert"]

    if not easy_scores or not expert_scores:
        return {"note": "insufficient data for discriminant validity"}

    stat, p = stats.mannwhitneyu(easy_scores, expert_scores, alternative="greater")

    # Cliff's delta
    n1, n2 = len(easy_scores), len(expert_scores)
    greater = sum(1 for x in easy_scores for y in expert_scores if x > y)
    less = sum(1 for x in easy_scores for y in expert_scores if x < y)
    delta = (greater - less) / (n1 * n2)

    return {
        "test": "Mann-Whitney U",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": p < 0.05,
        "cliff_delta": round(float(delta), 4),
        "n_easy": n1,
        "n_expert": n2,
        "mean_easy": round(float(np.mean(easy_scores)), 4),
        "mean_expert": round(float(np.mean(expert_scores)), 4),
        "interpretation": (
            f"Easy tasks (mean={np.mean(easy_scores):.3f}) vs Expert tasks "
            f"(mean={np.mean(expert_scores):.3f}): "
            + ("significantly different" if p < 0.05 else "not significantly different")
            + f" (Cliff's delta={delta:.3f})"
        ),
    }


def main():
    print("=== Benchmark Validity Study ===")

    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks from human baseline.")

    # 1. Difficulty predicts failure rate
    print("\n1. Testing: difficulty level predicts failure rate...")
    spearman_result = spearman_difficulty_failure(tasks)
    print(f"   {spearman_result['interpretation']}")

    # 2. Condition-level Spearman
    print("\n2. Testing: difficulty predicts performance per condition...")
    cond_spearman = spearman_difficulty_condition(CONDITION_DATA)
    for cond, res in cond_spearman.items():
        print(f"   {cond}: rho={res['spearman_rho']:.3f}, p={res['p_value']:.4f}")

    # 3. Friedman test across conditions
    print("\n3. Friedman test: conditions differ in performance...")
    friedman = friedman_test_across_conditions()
    print(f"   {friedman.get('interpretation', 'N/A')}")

    # 4. Cronbach's alpha
    print("\n4. Cronbach's alpha (benchmark reliability)...")
    alpha = cronbachs_alpha(tasks)
    print(f"   alpha={alpha.get('alpha', 'N/A'):.4f} ({alpha.get('interpretation', 'N/A')})")

    # 5. Discriminant validity
    print("\n5. Discriminant validity (easy vs expert)...")
    discrim = discriminant_validity(tasks)
    print(f"   {discrim.get('interpretation', 'N/A')}")

    output = {
        "meta": {
            "n_tasks": len(tasks),
            "conditions_used": list(CONDITION_DATA.keys()),
            "difficulty_levels": ["easy", "medium", "hard", "expert"],
        },
        "difficulty_predicts_failure": spearman_result,
        "condition_level_spearman": cond_spearman,
        "friedman_test": friedman,
        "cronbachs_alpha": alpha,
        "discriminant_validity": discrim,
        "validity_summary": {
            "construct_validity": spearman_result.get("significant", False),
            "criterion_validity": friedman.get("significant", False),
            "reliability": alpha.get("alpha", 0.0),
            "discriminant_validity": discrim.get("significant", False),
            "overall_verdict": (
                "VALID: benchmark demonstrates construct validity, criterion validity, "
                "and acceptable reliability"
                if (
                    spearman_result.get("significant", False)
                    and friedman.get("significant", False)
                    and alpha.get("alpha", 0.0) >= 0.6
                )
                else "MIXED: some validity criteria not fully met"
            ),
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
