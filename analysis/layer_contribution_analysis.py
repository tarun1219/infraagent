#!/usr/bin/env python3
"""
layer_contribution_analysis.py
Computes the "layer contribution" table from the human baseline results.
Shows how many tasks pass/fail at each validation layer and the false-pass rate
if only syntax were checked.
Produces results/layer_contribution.json
"""

import os
import json
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
INPUT_FILE = RESULTS_DIR / "human_baseline_results.json"
OUTPUT_FILE = RESULTS_DIR / "layer_contribution.json"


def load_results() -> list:
    """Load human baseline results, generating synthetic if missing."""
    if not INPUT_FILE.exists():
        print(f"WARNING: {INPUT_FILE} not found. Running human baseline first...")
        import subprocess, sys
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "run_human_baseline.py")],
            check=False
        )
    if INPUT_FILE.exists():
        with open(INPUT_FILE) as f:
            data = json.load(f)
        return data.get("per_task", [])
    print("ERROR: Could not load or generate human baseline results.")
    return []


def compute_layer_funnel(tasks: list) -> dict:
    """
    Compute layer-by-layer funnel.
    L1: syntax check
    L2: schema validation
    L3: security check (score >= 0.5)
    L4: best practices (bp_score >= 0.5) + all above = functional correct
    """
    n = len(tasks)
    if n == 0:
        return {}

    SECURITY_THRESHOLD = 0.5
    BP_THRESHOLD = 0.5

    l1_pass = [t for t in tasks if t.get("syntax_valid", False)]
    l2_pass = [t for t in l1_pass if t.get("schema_valid", False)]
    l3_pass = [t for t in l2_pass if t.get("security_score", 0.0) >= SECURITY_THRESHOLD]
    l4_pass = [t for t in l3_pass if t.get("bp_score", 0.0) >= BP_THRESHOLD]

    l1_fail = [t for t in tasks if not t.get("syntax_valid", False)]
    l2_fail = [t for t in l1_pass if not t.get("schema_valid", False)]
    l3_fail = [t for t in l2_pass if t.get("security_score", 0.0) < SECURITY_THRESHOLD]
    l4_fail = [t for t in l3_pass if t.get("bp_score", 0.0) < BP_THRESHOLD]

    # False-pass analysis: tasks that pass L1 (syntax) but fail functional
    syntax_pass_func_fail = [
        t for t in tasks
        if t.get("syntax_valid", False) and not t.get("functional_correct", False)
    ]
    # "Silent pass" rate: if we only checked syntax, how many would incorrectly pass?
    n_syntax_only_pass = len(l1_pass)
    n_func_correct = len(l4_pass)
    n_false_passes = len(syntax_pass_func_fail)

    false_pass_rate = n_false_passes / n if n > 0 else 0.0
    syntax_only_miss_rate = n_false_passes / n_syntax_only_pass if n_syntax_only_pass > 0 else 0.0

    return {
        "n_total": n,
        "layer_funnel": {
            "L1_syntax": {
                "pass": len(l1_pass),
                "fail": len(l1_fail),
                "pass_rate": round(len(l1_pass) / n, 4),
                "fail_rate": round(len(l1_fail) / n, 4),
            },
            "L2_schema": {
                "pass": len(l2_pass),
                "fail": len(l2_fail),
                "pass_rate": round(len(l2_pass) / n, 4),
                "fail_rate": round(len(l2_fail) / n, 4),
                "additional_failures_caught": len(l2_fail),
                "cumulative_fail_rate": round((len(l1_fail) + len(l2_fail)) / n, 4),
            },
            "L3_security": {
                "pass": len(l3_pass),
                "fail": len(l3_fail),
                "pass_rate": round(len(l3_pass) / n, 4),
                "fail_rate": round(len(l3_fail) / n, 4),
                "additional_failures_caught": len(l3_fail),
                "cumulative_fail_rate": round(
                    (len(l1_fail) + len(l2_fail) + len(l3_fail)) / n, 4
                ),
            },
            "L4_best_practices": {
                "pass": len(l4_pass),
                "fail": len(l4_fail),
                "pass_rate": round(len(l4_pass) / n, 4),
                "fail_rate": round(len(l4_fail) / n, 4),
                "additional_failures_caught": len(l4_fail),
                "cumulative_fail_rate": round(
                    (len(l1_fail) + len(l2_fail) + len(l3_fail) + len(l4_fail)) / n, 4
                ),
            },
        },
        "false_pass_analysis": {
            "n_syntax_pass": n_syntax_only_pass,
            "n_functional_correct": n_func_correct,
            "n_false_passes_if_syntax_only": n_false_passes,
            "false_pass_rate_overall": round(false_pass_rate, 4),
            "syntax_only_miss_rate": round(syntax_only_miss_rate, 4),
            "interpretation": (
                f"{false_pass_rate:.1%} of all tasks would be incorrectly "
                f"accepted if only syntax were checked. "
                f"Among syntax-valid outputs, {syntax_only_miss_rate:.1%} are "
                f"actually non-functional."
            ),
        },
    }


def compute_layer_contribution_by_group(tasks: list, group_key: str) -> dict:
    """Compute layer funnel broken down by language or difficulty."""
    groups = defaultdict(list)
    for t in tasks:
        groups[t.get(group_key, "unknown")].append(t)
    return {grp: compute_layer_funnel(items) for grp, items in sorted(groups.items())}


def compute_layer_marginal_value(funnel: dict) -> dict:
    """Compute the marginal value each layer adds beyond L1."""
    lf = funnel.get("layer_funnel", {})
    n = funnel.get("n_total", 1)

    l1_catch = lf.get("L1_syntax", {}).get("fail", 0)
    l2_catch = lf.get("L2_schema", {}).get("additional_failures_caught", 0)
    l3_catch = lf.get("L3_security", {}).get("additional_failures_caught", 0)
    l4_catch = lf.get("L4_best_practices", {}).get("additional_failures_caught", 0)

    total_caught = l1_catch + l2_catch + l3_catch + l4_catch

    return {
        "L1_syntax_contribution": round(l1_catch / n, 4) if n else 0,
        "L2_schema_contribution": round(l2_catch / n, 4) if n else 0,
        "L3_security_contribution": round(l3_catch / n, 4) if n else 0,
        "L4_bp_contribution": round(l4_catch / n, 4) if n else 0,
        "total_issues_caught_rate": round(total_caught / n, 4) if n else 0,
        "L2_marginal_gain_over_L1": round(l2_catch / n, 4) if n else 0,
        "L3_marginal_gain_over_L2": round(l3_catch / n, 4) if n else 0,
        "L4_marginal_gain_over_L3": round(l4_catch / n, 4) if n else 0,
    }


def main():
    print("=== Layer Contribution Analysis ===")

    tasks = load_results()
    if not tasks:
        print("ERROR: No task data available.")
        return

    print(f"Loaded {len(tasks)} tasks from human baseline.")

    # Overall funnel
    overall = compute_layer_funnel(tasks)
    marginal = compute_layer_marginal_value(overall)

    # By language
    by_language = compute_layer_contribution_by_group(tasks, "language")
    by_difficulty = compute_layer_contribution_by_group(tasks, "difficulty")

    output = {
        "meta": {
            "n_tasks": len(tasks),
            "security_threshold": 0.5,
            "bp_threshold": 0.5,
        },
        "overall_funnel": overall,
        "marginal_layer_value": marginal,
        "by_language": {lang: compute_layer_marginal_value(f) for lang, f in by_language.items()},
        "by_difficulty": {diff: compute_layer_marginal_value(f) for diff, f in by_difficulty.items()},
        "full_by_language": by_language,
        "full_by_difficulty": by_difficulty,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    lf = overall.get("layer_funnel", {})
    print(f"\n=== Layer Funnel (N={len(tasks)}) ===")
    for layer, data in lf.items():
        print(f"  {layer:25s}: pass={data['pass']:3d} ({data['pass_rate']:.1%})  "
              f"fail={data['fail']:3d} ({data['fail_rate']:.1%})  "
              f"additional_caught={data.get('additional_failures_caught', data['fail']):3d}")

    fpa = overall.get("false_pass_analysis", {})
    print(f"\n=== False-Pass Analysis ===")
    print(f"  If only syntax checked: {fpa['n_false_passes_if_syntax_only']} tasks "
          f"would be wrongly accepted ({fpa['false_pass_rate_overall']:.1%})")
    print(f"  Syntax-valid miss rate: {fpa['syntax_only_miss_rate']:.1%}")

    print(f"\n=== Marginal Layer Value ===")
    for k, v in marginal.items():
        print(f"  {k:40s}: {v:.4f}")

    print(f"\nResults written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
