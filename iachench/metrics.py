"""Canonical scoring logic for IaCBench evaluation.

All metric definitions here match exactly those used in the paper (Table 3).
Do not change thresholds without updating the paper.
"""
from __future__ import annotations
from typing import Dict, Any, List

# Thresholds (Section 5.4 of paper; ablation in Section 7.12)
SECURITY_THRESHOLD = 0.50   # >= 50% to pass Layer 3
BP_THRESHOLD = 0.60         # >= 60% to pass Layer 4


def compute_functional_correctness(validation_result: Dict[str, Any]) -> bool:
    """
    Canonical functional correctness definition (matches paper Table 3).

    A task passes iff ALL four validation layers pass:
    - L1: syntax valid
    - L2: schema valid
    - L3: security_score >= 0.50
    - L4: best_practices_score >= 0.60
    """
    return (
        bool(validation_result.get("syntax_valid", False))
        and bool(validation_result.get("schema_valid", False))
        and float(validation_result.get("security_score", 0.0)) >= SECURITY_THRESHOLD
        and float(validation_result.get("best_practices_score", 0.0)) >= BP_THRESHOLD
    )


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-task results into condition-level metrics (percentages)."""
    if not results:
        return {}
    n = len(results)
    return {
        "syntax":     sum(bool(r.get("syntax_valid", False)) for r in results) / n * 100,
        "schema":     sum(bool(r.get("schema_valid", False)) for r in results) / n * 100,
        "security":   sum(float(r.get("security_score", 0.0)) for r in results) / n * 100,
        "functional": sum(compute_functional_correctness(r) for r in results) / n * 100,
        "n_tasks":    n,
    }


def compute_recovery_rate(
    initial_results: List[Dict[str, Any]],
    corrected_results: List[Dict[str, Any]] = None,
):
    """
    Compute self-correction recovery rate.

    Two calling conventions:

    1. Two-list form (original):
       compute_recovery_rate(initial_results, corrected_results)
       Both lists contain per-task dicts with 'task_id'.

    2. Single-list form (test-friendly):
       compute_recovery_rate([{"initial_passed": bool, "final_passed": bool}, ...])
       Returns None if no initially-failing tasks exist.
    """
    if corrected_results is None:
        # Single-list form
        failing = [r for r in initial_results if not r.get("initial_passed", False)]
        if not failing:
            return None
        recovered = sum(1 for r in failing if r.get("final_passed", False))
        return recovered / len(failing)

    # Two-list form (original)
    initially_failing = [
        r for r in initial_results if not compute_functional_correctness(r)
    ]
    if not initially_failing:
        return 0.0
    failing_ids = {r["task_id"] for r in initially_failing}
    corrected_map = {r["task_id"]: r for r in corrected_results}
    recovered = sum(
        1 for tid in failing_ids
        if tid in corrected_map and compute_functional_correctness(corrected_map[tid])
    )
    return recovered / len(initially_failing)


def compute_metric(
    results: List[Dict[str, Any]],
    by_difficulty: bool = False,
) -> Dict[str, Any]:
    """
    Canonical single-metric function used by all paper figures and tables.

    Args:
        results: List of per-task dicts with keys:
            - functional (bool): overall pass/fail
            - security (bool): security layer pass/fail
            - difficulty (int, optional): required if by_difficulty=True
        by_difficulty: If True, also return per-difficulty breakdown.

    Returns dict with functional_accuracy (%), security_pass_rate (%), n, and
    optionally by_difficulty (dict keyed by difficulty level).
    """
    if not results:
        return {"functional_accuracy": 0.0, "security_pass_rate": 0.0, "n": 0}

    n = len(results)
    out: Dict[str, Any] = {
        "functional_accuracy": sum(1 for r in results if r.get("functional", False)) / n * 100,
        "security_pass_rate":  sum(1 for r in results if r.get("security",   False)) / n * 100,
        "n": n,
    }

    if by_difficulty:
        by_diff: Dict[int, list] = {}
        for r in results:
            diff = r.get("difficulty")
            if diff is not None:
                by_diff.setdefault(diff, []).append(r)
        out["by_difficulty"] = {
            level: compute_metric(level_results)
            for level, level_results in by_diff.items()
        }

    return out


def compute_pass_at_k(
    results_per_task: List[List[bool]],
    k: int,
) -> float:
    """
    Compute pass@k: fraction of tasks that pass in at least one of k attempts.

    Returns percentage (0–100).
    """
    if not results_per_task:
        return 0.0
    n_pass = sum(1 for attempts in results_per_task if any(attempts[:k]))
    return n_pass / len(results_per_task) * 100
