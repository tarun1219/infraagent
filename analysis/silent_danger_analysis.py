#!/usr/bin/env python3
"""
silent_danger_analysis.py
Quantifies the "silent danger" problem:
Files that pass syntax+schema checks but have poor security scores.
Produces results/silent_danger.json
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
INPUT_FILE = RESULTS_DIR / "human_baseline_results.json"
OUTPUT_FILE = RESULTS_DIR / "silent_danger.json"

SECURITY_THRESHOLD = 0.5
SCHEMA_REQUIRED = True


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


def is_silently_dangerous(task: dict, security_threshold: float = SECURITY_THRESHOLD) -> bool:
    """
    A task is 'silently dangerous' if:
    - syntax is valid (passes basic checks), AND
    - schema is valid (would be accepted by validators), BUT
    - security_score < threshold (has real security problems)
    """
    return (
        task.get("syntax_valid", False)
        and task.get("schema_valid", False)
        and task.get("security_score", 1.0) < security_threshold
    )


def compute_silent_danger_stats(tasks: list,
                                 security_threshold: float = SECURITY_THRESHOLD) -> dict:
    """Compute silent danger statistics for a group of tasks."""
    n = len(tasks)
    if n == 0:
        return {"n": 0}

    n_syntax_valid = sum(1 for t in tasks if t.get("syntax_valid", False))
    n_schema_valid = sum(
        1 for t in tasks
        if t.get("syntax_valid", False) and t.get("schema_valid", False)
    )
    n_silently_dangerous = sum(
        1 for t in tasks if is_silently_dangerous(t, security_threshold)
    )

    # Rate among all tasks
    rate_overall = n_silently_dangerous / n

    # Rate among syntax-valid tasks
    rate_among_syntax_valid = (
        n_silently_dangerous / n_syntax_valid if n_syntax_valid > 0 else 0.0
    )

    # Rate among schema-valid tasks (surface area exposed to deployment)
    rate_among_schema_valid = (
        n_silently_dangerous / n_schema_valid if n_schema_valid > 0 else 0.0
    )

    # Mean security score of silently dangerous tasks
    sd_tasks = [t for t in tasks if is_silently_dangerous(t, security_threshold)]
    mean_security_sd = (
        float(np.mean([t["security_score"] for t in sd_tasks])) if sd_tasks else 0.0
    )
    mean_security_overall = float(np.mean([t.get("security_score", 0.0) for t in tasks]))

    # Security score distribution among syntax+schema valid
    deployable = [
        t for t in tasks
        if t.get("syntax_valid", False) and t.get("schema_valid", False)
    ]
    sec_scores_deployable = [t.get("security_score", 0.0) for t in deployable]

    return {
        "n_total": n,
        "n_syntax_valid": n_syntax_valid,
        "n_schema_valid": n_schema_valid,
        "n_silently_dangerous": n_silently_dangerous,
        "rate_of_total": round(rate_overall, 4),
        "rate_among_syntax_valid": round(rate_among_syntax_valid, 4),
        "rate_among_schema_valid": round(rate_among_schema_valid, 4),
        "mean_security_score_overall": round(mean_security_overall, 4),
        "mean_security_score_silently_dangerous": round(mean_security_sd, 4),
        "security_score_distribution_deployable": {
            "mean": round(float(np.mean(sec_scores_deployable)), 4) if sec_scores_deployable else 0.0,
            "std": round(float(np.std(sec_scores_deployable)), 4) if sec_scores_deployable else 0.0,
            "min": round(float(min(sec_scores_deployable)), 4) if sec_scores_deployable else 0.0,
            "max": round(float(max(sec_scores_deployable)), 4) if sec_scores_deployable else 0.0,
            "pct_below_0.5": round(
                sum(1 for s in sec_scores_deployable if s < 0.5) / len(sec_scores_deployable), 4
            ) if sec_scores_deployable else 0.0,
            "pct_below_0.3": round(
                sum(1 for s in sec_scores_deployable if s < 0.3) / len(sec_scores_deployable), 4
            ) if sec_scores_deployable else 0.0,
        },
    }


def compute_threshold_sensitivity(tasks: list) -> list:
    """How does silent danger rate change with security threshold?"""
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    return [
        {
            "threshold": t,
            "n_silently_dangerous": sum(
                1 for task in tasks if is_silently_dangerous(task, t)
            ),
            "rate": round(
                sum(1 for task in tasks if is_silently_dangerous(task, t)) / len(tasks), 4
            ) if tasks else 0.0,
        }
        for t in thresholds
    ]


def main():
    print("=== Silent Danger Analysis ===")
    print(f"Security threshold: {SECURITY_THRESHOLD}")
    print("Definition: syntax_valid AND schema_valid BUT security_score < threshold\n")

    tasks = load_tasks()
    if not tasks:
        print("ERROR: No task data available.")
        return

    print(f"Loaded {len(tasks)} tasks.")

    # Overall stats
    overall = compute_silent_danger_stats(tasks)
    print(f"\nOverall silent danger:")
    print(f"  Silently dangerous tasks:  {overall['n_silently_dangerous']}/{overall['n_total']} "
          f"({overall['rate_of_total']:.1%})")
    print(f"  Among syntax-valid:        {overall['rate_among_syntax_valid']:.1%}")
    print(f"  Among schema-valid (deployable): {overall['rate_among_schema_valid']:.1%}")

    # By language
    lang_groups = defaultdict(list)
    diff_groups = defaultdict(list)
    for t in tasks:
        lang_groups[t.get("language", "unknown")].append(t)
        diff_groups[t.get("difficulty", "unknown")].append(t)

    by_language = {lang: compute_silent_danger_stats(grp) for lang, grp in sorted(lang_groups.items())}
    by_difficulty = {diff: compute_silent_danger_stats(grp) for diff, grp in sorted(diff_groups.items())}

    print(f"\nBy language (silent danger rate among deployable):")
    for lang, stats_d in by_language.items():
        print(f"  {lang:15s}: {stats_d['rate_among_schema_valid']:.1%} "
              f"({stats_d['n_silently_dangerous']}/{stats_d['n_schema_valid']})")

    print(f"\nBy difficulty (silent danger rate among deployable):")
    for diff, stats_d in by_difficulty.items():
        print(f"  {diff:8s}: {stats_d['rate_among_schema_valid']:.1%} "
              f"({stats_d['n_silently_dangerous']}/{stats_d['n_schema_valid']})")

    # Threshold sensitivity
    threshold_sens = compute_threshold_sensitivity(tasks)
    print(f"\nThreshold sensitivity:")
    for row in threshold_sens:
        print(f"  sec_threshold={row['threshold']:.1f}: "
              f"{row['n_silently_dangerous']:3d} silently dangerous ({row['rate']:.1%})")

    # Examples of silently dangerous tasks (top 5 worst security scores)
    sd_tasks = [t for t in tasks if is_silently_dangerous(t)]
    sd_tasks_sorted = sorted(sd_tasks, key=lambda t: t.get("security_score", 1.0))
    worst_examples = [
        {
            "task_id": t.get("task_id", "?"),
            "language": t.get("language", "?"),
            "difficulty": t.get("difficulty", "?"),
            "security_score": t.get("security_score", 0.0),
            "syntax_valid": t.get("syntax_valid"),
            "schema_valid": t.get("schema_valid"),
        }
        for t in sd_tasks_sorted[:10]
    ]

    output = {
        "meta": {
            "n_tasks": len(tasks),
            "security_threshold": SECURITY_THRESHOLD,
            "definition": "syntax_valid AND schema_valid BUT security_score < threshold",
        },
        "overall": overall,
        "by_language": by_language,
        "by_difficulty": by_difficulty,
        "threshold_sensitivity": threshold_sens,
        "worst_examples": worst_examples,
        "interpretation": {
            "key_finding": (
                f"{overall['rate_among_schema_valid']:.1%} of deployable IaC "
                f"(syntax+schema valid) has poor security, creating a 'silent danger' "
                f"that syntax-only checkers cannot detect."
            ),
            "implication": (
                "Deploying based on syntax/schema validity alone would expose "
                f"{overall['n_silently_dangerous']} tasks worth of insecure infrastructure."
            ),
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
