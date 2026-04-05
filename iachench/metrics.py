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
    corrected_results: List[Dict[str, Any]],
) -> float:
    """
    Compute self-correction recovery rate.

    Recovery rate = (tasks that fail initially but pass after SC) / (tasks that fail initially)
    """
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
