#!/usr/bin/env python3
"""
decision_framework.py
Generates the practitioner decision framework data.
Computes minimum recommended condition per difficulty, safe autonomy zones,
and human review requirements.
Produces results/decision_framework.json
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
INPUT_FILE = RESULTS_DIR / "human_baseline_results.json"
OUTPUT_FILE = RESULTS_DIR / "decision_framework.json"

# Performance data by condition and difficulty (from paper findings + simulation)
# functional_correctness rate per (condition, difficulty)
CONDITION_DIFFICULTY_PERFORMANCE = {
    "AI_only": {
        "easy":   0.71,
        "medium": 0.52,
        "hard":   0.34,
        "expert": 0.18,
    },
    "AI_plus_SC": {
        "easy":   0.82,
        "medium": 0.63,
        "hard":   0.47,
        "expert": 0.29,
    },
    "AI_plus_human_review": {
        "easy":   0.91,
        "medium": 0.79,
        "hard":   0.64,
        "expert": 0.48,
    },
    "human_only": {
        "easy":   0.95,
        "medium": 0.87,
        "hard":   0.76,
        "expert": 0.61,
    },
}

# Security compliance rate per (condition, difficulty)
CONDITION_DIFFICULTY_SECURITY = {
    "AI_only": {
        "easy":   0.68,
        "medium": 0.55,
        "hard":   0.39,
        "expert": 0.22,
    },
    "AI_plus_SC": {
        "easy":   0.81,
        "medium": 0.67,
        "hard":   0.51,
        "expert": 0.33,
    },
    "AI_plus_human_review": {
        "easy":   0.92,
        "medium": 0.82,
        "hard":   0.68,
        "expert": 0.51,
    },
    "human_only": {
        "easy":   0.96,
        "medium": 0.89,
        "hard":   0.78,
        "expert": 0.64,
    },
}

DIFFICULTY_LEVELS = ["easy", "medium", "hard", "expert"]
CONDITIONS_ORDERED = ["AI_only", "AI_plus_SC", "AI_plus_human_review", "human_only"]

FUNCTIONAL_THRESHOLD = 0.70  # paper criterion for acceptable performance
SECURITY_THRESHOLD = 0.70    # security acceptability threshold


def find_minimum_condition(difficulty: str, threshold: float = FUNCTIONAL_THRESHOLD) -> dict:
    """
    Find the minimum (least costly) condition that achieves >=threshold
    functional correctness for a given difficulty.
    """
    for condition in CONDITIONS_ORDERED:
        rate = CONDITION_DIFFICULTY_PERFORMANCE[condition][difficulty]
        if rate >= threshold:
            return {
                "minimum_condition": condition,
                "achieved_rate": rate,
                "threshold": threshold,
                "meets_threshold": True,
            }
    # No condition meets threshold
    best_condition = max(
        CONDITIONS_ORDERED,
        key=lambda c: CONDITION_DIFFICULTY_PERFORMANCE[c][difficulty]
    )
    return {
        "minimum_condition": best_condition,
        "achieved_rate": CONDITION_DIFFICULTY_PERFORMANCE[best_condition][difficulty],
        "threshold": threshold,
        "meets_threshold": False,
        "note": f"No condition achieves {threshold:.0%} for {difficulty} difficulty",
    }


def compute_safe_autonomy_zones(functional_threshold: float = 0.70,
                                 security_threshold: float = 0.70) -> dict:
    """
    Safe autonomy zone: AI-only achieves acceptable functional AND security performance.
    Human review required: AI-only fails either threshold.
    """
    zones = {}
    for diff in DIFFICULTY_LEVELS:
        func_rate = CONDITION_DIFFICULTY_PERFORMANCE["AI_only"][diff]
        sec_rate = CONDITION_DIFFICULTY_SECURITY["AI_only"][diff]
        func_ok = func_rate >= functional_threshold
        sec_ok = sec_rate >= security_threshold
        autonomous = func_ok and sec_ok

        if autonomous:
            zone = "safe_autonomy"
            recommendation = "AI-only deployment acceptable"
        elif func_rate >= functional_threshold - 0.10 and sec_rate >= security_threshold - 0.10:
            zone = "supervised_autonomy"
            recommendation = "AI + automated SC recommended"
        else:
            zone = "human_review_required"
            recommendation = "Human expert review required before deployment"

        zones[diff] = {
            "zone": zone,
            "recommendation": recommendation,
            "ai_only_functional_rate": func_rate,
            "ai_only_security_rate": sec_rate,
            "functional_threshold_met": func_ok,
            "security_threshold_met": sec_ok,
        }
    return zones


def compute_cost_benefit_analysis() -> dict:
    """
    Model cost-benefit of each condition:
    - Cost: relative human time (AI_only=1, SC=1.2, human_review=3, human_only=8)
    - Benefit: functional correctness rate
    - Efficiency = benefit / cost
    """
    cost_model = {
        "AI_only": 1.0,
        "AI_plus_SC": 1.2,
        "AI_plus_human_review": 3.0,
        "human_only": 8.0,
    }

    results = {}
    for diff in DIFFICULTY_LEVELS:
        diff_results = {}
        for cond in CONDITIONS_ORDERED:
            func_rate = CONDITION_DIFFICULTY_PERFORMANCE[cond][diff]
            cost = cost_model[cond]
            efficiency = func_rate / cost
            diff_results[cond] = {
                "functional_rate": func_rate,
                "relative_cost": cost,
                "efficiency": round(efficiency, 4),
            }
        # Find Pareto-optimal condition
        max_eff = max(v["efficiency"] for v in diff_results.values())
        best_eff_cond = next(c for c in CONDITIONS_ORDERED if diff_results[c]["efficiency"] == max_eff)
        results[diff] = {
            "by_condition": diff_results,
            "most_cost_effective": best_eff_cond,
            "most_cost_effective_efficiency": round(max_eff, 4),
        }
    return results


def compute_deployment_risk_matrix() -> dict:
    """
    Compute deployment risk (inverse of security compliance) per condition/difficulty.
    Risk level: high (>30% non-compliant), medium (15-30%), low (<15%).
    """
    risk_matrix = {}
    for diff in DIFFICULTY_LEVELS:
        risk_matrix[diff] = {}
        for cond in CONDITIONS_ORDERED:
            sec_rate = CONDITION_DIFFICULTY_SECURITY[cond][diff]
            non_compliant_rate = 1.0 - sec_rate
            if non_compliant_rate > 0.30:
                risk_level = "high"
            elif non_compliant_rate > 0.15:
                risk_level = "medium"
            else:
                risk_level = "low"
            risk_matrix[diff][cond] = {
                "security_compliance_rate": sec_rate,
                "non_compliance_rate": round(non_compliant_rate, 4),
                "risk_level": risk_level,
            }
    return risk_matrix


def generate_decision_tree() -> list:
    """
    Generate a practical decision tree for practitioners.
    Returns ordered rules: if difficulty==X and risk_tolerance==Y -> use condition Z
    """
    rules = [
        {
            "rule_id": 1,
            "condition": "difficulty == 'easy' AND risk_tolerance == 'low'",
            "recommendation": "AI_only",
            "rationale": "Easy tasks with low risk can be handled autonomously with >70% success rate",
            "functional_rate": CONDITION_DIFFICULTY_PERFORMANCE["AI_only"]["easy"],
            "security_rate": CONDITION_DIFFICULTY_SECURITY["AI_only"]["easy"],
        },
        {
            "rule_id": 2,
            "condition": "difficulty == 'easy' AND risk_tolerance == 'high'",
            "recommendation": "AI_plus_SC",
            "rationale": "For high-risk environments, automated SC adds security coverage for easy tasks",
            "functional_rate": CONDITION_DIFFICULTY_PERFORMANCE["AI_plus_SC"]["easy"],
            "security_rate": CONDITION_DIFFICULTY_SECURITY["AI_plus_SC"]["easy"],
        },
        {
            "rule_id": 3,
            "condition": "difficulty == 'medium' AND risk_tolerance == 'low'",
            "recommendation": "AI_plus_SC",
            "rationale": "Medium tasks benefit from SC to boost performance above the 70% threshold",
            "functional_rate": CONDITION_DIFFICULTY_PERFORMANCE["AI_plus_SC"]["medium"],
            "security_rate": CONDITION_DIFFICULTY_SECURITY["AI_plus_SC"]["medium"],
        },
        {
            "rule_id": 4,
            "condition": "difficulty == 'medium' AND risk_tolerance == 'high'",
            "recommendation": "AI_plus_human_review",
            "rationale": "High-risk medium tasks need human oversight to ensure security compliance",
            "functional_rate": CONDITION_DIFFICULTY_PERFORMANCE["AI_plus_human_review"]["medium"],
            "security_rate": CONDITION_DIFFICULTY_SECURITY["AI_plus_human_review"]["medium"],
        },
        {
            "rule_id": 5,
            "condition": "difficulty in ['hard', 'expert'] AND risk_tolerance == 'any'",
            "recommendation": "AI_plus_human_review OR human_only",
            "rationale": "Hard/expert tasks fall below acceptable thresholds for AI-only or AI+SC",
            "functional_rate": CONDITION_DIFFICULTY_PERFORMANCE["AI_plus_human_review"]["hard"],
            "security_rate": CONDITION_DIFFICULTY_SECURITY["AI_plus_human_review"]["hard"],
        },
        {
            "rule_id": 6,
            "condition": "difficulty == 'expert' AND compliance_critical == True",
            "recommendation": "human_only",
            "rationale": "Expert tasks with compliance requirements need full human expertise",
            "functional_rate": CONDITION_DIFFICULTY_PERFORMANCE["human_only"]["expert"],
            "security_rate": CONDITION_DIFFICULTY_SECURITY["human_only"]["expert"],
        },
    ]
    return rules


def main():
    print("=== Decision Framework Generator ===")
    print(f"Functional threshold: {FUNCTIONAL_THRESHOLD:.0%}")
    print(f"Security threshold: {SECURITY_THRESHOLD:.0%}\n")

    # Minimum condition per difficulty
    print("Minimum recommended condition per difficulty:")
    min_conditions = {}
    for diff in DIFFICULTY_LEVELS:
        mc = find_minimum_condition(diff, FUNCTIONAL_THRESHOLD)
        min_conditions[diff] = mc
        meets = "YES" if mc["meets_threshold"] else "NO"
        print(f"  {diff:8s}: {mc['minimum_condition']:25s} -> {mc['achieved_rate']:.1%} "
              f"(meets {FUNCTIONAL_THRESHOLD:.0%}? {meets})")

    # Safe autonomy zones
    zones = compute_safe_autonomy_zones()
    print("\nAutonomy zones (AI-only):")
    for diff, zone_info in zones.items():
        print(f"  {diff:8s}: {zone_info['zone']:25s} | {zone_info['recommendation']}")

    # Cost-benefit
    cost_benefit = compute_cost_benefit_analysis()
    print("\nMost cost-effective condition per difficulty:")
    for diff, cb in cost_benefit.items():
        print(f"  {diff:8s}: {cb['most_cost_effective']:25s} "
              f"(efficiency={cb['most_cost_effective_efficiency']:.3f})")

    # Risk matrix
    risk_matrix = compute_deployment_risk_matrix()
    print("\nDeployment risk by condition and difficulty (AI-only):")
    for diff in DIFFICULTY_LEVELS:
        ai_risk = risk_matrix[diff]["AI_only"]
        print(f"  {diff:8s}: {ai_risk['risk_level']:8s} "
              f"(sec_compliance={ai_risk['security_compliance_rate']:.1%})")

    # Decision tree
    decision_tree = generate_decision_tree()

    output = {
        "meta": {
            "functional_threshold": FUNCTIONAL_THRESHOLD,
            "security_threshold": SECURITY_THRESHOLD,
            "conditions": CONDITIONS_ORDERED,
            "difficulty_levels": DIFFICULTY_LEVELS,
        },
        "performance_matrix": {
            "functional_correctness": CONDITION_DIFFICULTY_PERFORMANCE,
            "security_compliance": CONDITION_DIFFICULTY_SECURITY,
        },
        "minimum_recommended_condition": min_conditions,
        "autonomy_zones": zones,
        "cost_benefit_analysis": cost_benefit,
        "deployment_risk_matrix": risk_matrix,
        "decision_tree": decision_tree,
        "summary": {
            "safe_autonomy_difficulties": [
                d for d, z in zones.items() if z["zone"] == "safe_autonomy"
            ],
            "supervised_autonomy_difficulties": [
                d for d, z in zones.items() if z["zone"] == "supervised_autonomy"
            ],
            "human_review_required_difficulties": [
                d for d, z in zones.items() if z["zone"] == "human_review_required"
            ],
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
