#!/usr/bin/env python3
"""
comprehensive_results_compiler.py
Master script that runs all analyses and compiles into a single results file.
Runs all analyses in order, loads all JSONs, compiles all_results.json,
and prints a summary table of key findings.
"""

import os
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "all_results.json"

TOOL_PATH = "/Users/tarun/Library/Python/3.9/bin:/opt/homebrew/bin:" + os.environ.get("PATH", "")
ENV = {**os.environ, "PATH": TOOL_PATH}

# Analysis scripts in dependency order
ANALYSIS_SCRIPTS = [
    # Name, script path, description, dependencies (result files needed before running)
    ("human_baseline", "run_human_baseline.py",
     "Human reference solution validation", []),
    ("layer_contribution", "layer_contribution_analysis.py",
     "Layer contribution funnel analysis", ["human_baseline_results.json"]),
    ("threshold_sensitivity", "threshold_sensitivity_analysis.py",
     "Threshold sensitivity analysis", ["human_baseline_results.json"]),
    ("power_analysis", "power_analysis.py",
     "Statistical power analysis", []),
    ("pass_at_k", "pass_at_k_analysis.py",
     "Pass@k theoretical analysis", ["human_baseline_results.json"]),
    ("error_co_occurrence", "error_co_occurrence.py",
     "Error co-occurrence (phi coefficients)", ["human_baseline_results.json"]),
    ("silent_danger", "silent_danger_analysis.py",
     "Silent danger quantification", ["human_baseline_results.json"]),
    ("correctability", "correctability_framework.py",
     "Correctability taxonomy framework", []),
    ("decision_framework", "decision_framework.py",
     "Practitioner decision framework", []),
    ("benchmark_validity", "benchmark_validity_study.py",
     "Benchmark validity study", ["human_baseline_results.json"]),
    ("regression_analysis", "regression_analysis.py",
     "SC regression pattern analysis", []),
    ("inter_tool_agreement", "inter_tool_agreement.py",
     "Inter-tool agreement (checkov vs trivy)", []),
]

# Result JSON filenames per analysis
RESULT_FILES = {
    "human_baseline": "human_baseline_results.json",
    "github_baseline": "github_baseline_results.json",
    "layer_contribution": "layer_contribution.json",
    "threshold_sensitivity": "threshold_sensitivity.json",
    "power_analysis": "power_analysis.json",
    "pass_at_k": "pass_at_k.json",
    "error_co_occurrence": "error_co_occurrence.json",
    "silent_danger": "silent_danger.json",
    "correctability": "correctability_framework.json",
    "decision_framework": "decision_framework.json",
    "inter_tool_agreement": "inter_tool_agreement.json",
    "benchmark_validity": "benchmark_validity.json",
    "regression_analysis": "regression_analysis.json",
}


def run_script(name: str, script: str, description: str) -> tuple:
    """Run a single analysis script. Returns (success, duration_s, error_msg)."""
    script_path = SCRIPT_DIR / script
    if not script_path.exists():
        return False, 0.0, f"Script not found: {script_path}"

    print(f"  Running {name} ({description})...", end=" ", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=300, env=ENV,
            cwd=str(SCRIPT_DIR)
        )
        duration = time.time() - t0
        if proc.returncode != 0:
            stderr_snippet = proc.stderr[-500:] if proc.stderr else ""
            print(f"FAILED ({duration:.1f}s)")
            return False, duration, stderr_snippet
        print(f"OK ({duration:.1f}s)")
        return True, duration, None
    except subprocess.TimeoutExpired:
        duration = time.time() - t0
        print(f"TIMEOUT ({duration:.1f}s)")
        return False, duration, "timeout"
    except Exception as e:
        duration = time.time() - t0
        print(f"ERROR ({duration:.1f}s): {e}")
        return False, duration, str(e)


def load_json_safe(path: Path) -> dict:
    """Load JSON file, return {} on error."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"could not load {path}: {e}"}


def extract_key_metrics(all_data: dict) -> dict:
    """Extract the key metrics needed for the paper."""
    metrics = {}

    # Human baseline
    hb = all_data.get("human_baseline", {})
    agg = hb.get("aggregates", {}).get("overall", {})
    metrics["human_baseline"] = {
        "n_tasks": agg.get("n", 0),
        "syntax_valid_rate": agg.get("syntax_valid_rate", 0),
        "schema_valid_rate": agg.get("schema_valid_rate", 0),
        "mean_security_score": agg.get("mean_security_score", 0),
        "mean_bp_score": agg.get("mean_bp_score", 0),
        "functional_correct_rate": agg.get("functional_correct_rate", 0),
    }

    # Layer contribution
    lc = all_data.get("layer_contribution", {})
    overall_funnel = lc.get("overall_funnel", {})
    lf = overall_funnel.get("layer_funnel", {})
    fpa = overall_funnel.get("false_pass_analysis", {})
    metrics["layer_contribution"] = {
        "l1_syntax_pass_rate": lf.get("L1_syntax", {}).get("pass_rate", 0),
        "l2_schema_pass_rate": lf.get("L2_schema", {}).get("pass_rate", 0),
        "l3_security_pass_rate": lf.get("L3_security", {}).get("pass_rate", 0),
        "l4_bp_pass_rate": lf.get("L4_best_practices", {}).get("pass_rate", 0),
        "false_pass_rate_if_syntax_only": fpa.get("false_pass_rate_overall", 0),
        "syntax_only_miss_rate": fpa.get("syntax_only_miss_rate", 0),
    }

    # Threshold sensitivity
    ts = all_data.get("threshold_sensitivity", {})
    metrics["threshold_sensitivity"] = {
        "baseline_rate": ts.get("grid", {}).get("baseline_rate", 0),
        "robustness_fraction": ts.get("robustness_zone", {}).get("robustness_fraction", 0),
        "dominant_threshold": ts.get("sensitivity", {}).get("dominant_threshold", "unknown"),
        "grid_range": ts.get("sensitivity", {}).get("grid_range", 0),
    }

    # Power analysis
    pa = all_data.get("power_analysis", {})
    reported = pa.get("reported_effect_sizes", {})
    mde = pa.get("minimum_detectable_effect", {}).get("at_alpha_bonferroni", {})
    metrics["power_analysis"] = {
        "functional_delta": reported.get("functional_correctness", {}).get("cliff_delta", 0.31),
        "functional_power_mc": reported.get("functional_correctness", {}).get("power_mc", 0),
        "security_delta": reported.get("security_score", {}).get("cliff_delta", 0.52),
        "security_power_mc": reported.get("security_score", {}).get("power_mc", 0),
        "mde_bonferroni": mde.get("mde_cliff_delta", 0),
        "n_tasks": pa.get("meta", {}).get("n_tasks", 150),
        "alpha_bonferroni": pa.get("meta", {}).get("alpha_bonferroni", 0.005),
    }

    # Pass@k
    pak = all_data.get("pass_at_k", {}).get("by_condition", {})
    metrics["pass_at_k"] = {
        cond: {
            f"pass@{k}": pak.get(cond, {}).get("pass_at_k", {}).get(f"pass@{k}", {}).get("mean", 0)
            for k in [1, 3, 5]
        }
        for cond in ["AI_only", "AI_plus_SC", "human_only"]
        if cond in pak
    }

    # Silent danger
    sd = all_data.get("silent_danger", {}).get("overall", {})
    metrics["silent_danger"] = {
        "n_silently_dangerous": sd.get("n_silently_dangerous", 0),
        "rate_of_total": sd.get("rate_of_total", 0),
        "rate_among_schema_valid": sd.get("rate_among_schema_valid", 0),
    }

    # Correctability framework
    cf = all_data.get("correctability", {})
    taxonomy = cf.get("taxonomy_table", [])
    metrics["correctability"] = {
        row["error_class"]: {
            "informativeness": row.get("informativeness", {}).get("mean", 0),
            "actual_sc_recovery_rate": row.get("actual_sc_recovery_rate", 0),
        }
        for row in taxonomy
    }

    # Benchmark validity
    bv = all_data.get("benchmark_validity", {})
    metrics["benchmark_validity"] = {
        "cronbachs_alpha": bv.get("cronbachs_alpha", {}).get("alpha", 0),
        "difficulty_predicts_failure": bv.get("difficulty_predicts_failure", {}).get("spearman_rho", 0),
        "friedman_p_value": bv.get("friedman_test", {}).get("p_value", 1.0),
        "kendalls_w": bv.get("friedman_test", {}).get("kendalls_w", 0),
        "overall_verdict": bv.get("validity_summary", {}).get("overall_verdict", "unknown"),
    }

    # Inter-tool agreement
    ita = all_data.get("inter_tool_agreement", {})
    kappa = ita.get("cohens_kappa", {})
    metrics["inter_tool_agreement"] = {
        "cohens_kappa": kappa.get("kappa", 0),
        "observed_agreement": kappa.get("observed_agreement", 0),
        "interpretation": kappa.get("interpretation", "unknown"),
    }

    # Decision framework
    df = all_data.get("decision_framework", {})
    summary = df.get("summary", {})
    metrics["decision_framework"] = {
        "safe_autonomy_difficulties": summary.get("safe_autonomy_difficulties", []),
        "human_review_required": summary.get("human_review_required_difficulties", []),
    }

    return metrics


def print_summary_table(metrics: dict):
    """Print a formatted summary table of key findings."""
    print("\n" + "=" * 70)
    print("KEY FINDINGS SUMMARY")
    print("=" * 70)

    hb = metrics.get("human_baseline", {})
    print(f"\n[Human Baseline] N={hb.get('n_tasks', '?')}")
    print(f"  Syntax valid:           {hb.get('syntax_valid_rate', 0):.1%}")
    print(f"  Schema valid:           {hb.get('schema_valid_rate', 0):.1%}")
    print(f"  Mean security score:    {hb.get('mean_security_score', 0):.3f}")
    print(f"  Functional correct:     {hb.get('functional_correct_rate', 0):.1%}")

    lc = metrics.get("layer_contribution", {})
    print(f"\n[Layer Contribution]")
    print(f"  False-pass rate (syntax-only check): {lc.get('false_pass_rate_if_syntax_only', 0):.1%}")
    print(f"  Miss rate among syntax-valid:        {lc.get('syntax_only_miss_rate', 0):.1%}")

    pa = metrics.get("power_analysis", {})
    print(f"\n[Power Analysis] n={pa.get('n_tasks', 150)}, alpha={pa.get('alpha_bonferroni', 0.005)}")
    print(f"  Functional delta={pa.get('functional_delta', 0.31)}: "
          f"power={pa.get('functional_power_mc', 0):.3f}")
    print(f"  Security delta={pa.get('security_delta', 0.52)}: "
          f"power={pa.get('security_power_mc', 0):.3f}")
    print(f"  MDE (Bonferroni): delta={pa.get('mde_bonferroni', 0):.4f}")

    sd = metrics.get("silent_danger", {})
    print(f"\n[Silent Danger]")
    print(f"  Silently dangerous (of all): {sd.get('rate_of_total', 0):.1%}")
    print(f"  Among deployable (schema-valid): {sd.get('rate_among_schema_valid', 0):.1%}")

    ts = metrics.get("threshold_sensitivity", {})
    print(f"\n[Threshold Sensitivity]")
    print(f"  Baseline rate: {ts.get('baseline_rate', 0):.3f}")
    print(f"  Robustness zone: {ts.get('robustness_fraction', 0):.1%} of threshold grid")
    print(f"  Dominant threshold: {ts.get('dominant_threshold', '?')}")

    pak = metrics.get("pass_at_k", {})
    print(f"\n[Pass@k]")
    for cond, rates in pak.items():
        line = f"  {cond:25s}:"
        for k in [1, 3, 5]:
            line += f"  @{k}={rates.get(f'pass@{k}', 0):.3f}"
        print(line)

    bv = metrics.get("benchmark_validity", {})
    print(f"\n[Benchmark Validity]")
    print(f"  Cronbach's alpha:          {bv.get('cronbachs_alpha', 0):.4f}")
    print(f"  Difficulty->failure rho:   {bv.get('difficulty_predicts_failure', 0):.4f}")
    print(f"  Friedman p-value:          {bv.get('friedman_p_value', 1.0):.4f}")
    print(f"  Kendall's W:               {bv.get('kendalls_w', 0):.4f}")
    print(f"  Verdict: {bv.get('overall_verdict', 'unknown')}")

    ita = metrics.get("inter_tool_agreement", {})
    print(f"\n[Inter-Tool Agreement (checkov vs trivy)]")
    print(f"  Cohen's kappa:    {ita.get('cohens_kappa', 0):.4f}")
    print(f"  Agreement rate:   {ita.get('observed_agreement', 0):.1%}")
    print(f"  Interpretation:   {ita.get('interpretation', '?')}")

    cf = metrics.get("correctability", {})
    print(f"\n[Correctability Framework]")
    for et, vals in cf.items():
        print(f"  {et:15s}: info={vals.get('informativeness', 0):.3f} "
              f"recovery={vals.get('actual_sc_recovery_rate', 0):.3f}")

    df = metrics.get("decision_framework", {})
    print(f"\n[Decision Framework]")
    print(f"  Safe autonomy (AI-only ok): {df.get('safe_autonomy_difficulties', [])}")
    print(f"  Human review required:      {df.get('human_review_required', [])}")

    print("=" * 70)


def main():
    print("=" * 70)
    print("COMPREHENSIVE RESULTS COMPILER")
    print("IaCBench Analysis Pipeline")
    print("=" * 70)
    print(f"\nScript directory: {SCRIPT_DIR}")
    print(f"Results directory: {RESULTS_DIR}\n")

    # Run all analysis scripts
    run_log = []
    print("Running all analyses:")
    print("-" * 50)
    total_start = time.time()

    for name, script, description, deps in ANALYSIS_SCRIPTS:
        # Check dependencies
        deps_met = all((RESULTS_DIR / dep).exists() for dep in deps)
        if deps and not deps_met:
            missing = [dep for dep in deps if not (RESULTS_DIR / dep).exists()]
            print(f"  Skipping {name}: missing deps {missing}")
            run_log.append({
                "name": name,
                "script": script,
                "status": "skipped",
                "reason": f"missing dependencies: {missing}",
            })
            continue

        success, duration, error = run_script(name, script, description)
        run_log.append({
            "name": name,
            "script": script,
            "status": "ok" if success else "failed",
            "duration_s": round(duration, 2),
            "error": error,
        })

    total_duration = time.time() - total_start
    print(f"\nAll scripts complete in {total_duration:.1f}s")

    # Load all result files
    print("\nLoading result files...")
    all_data = {}
    for analysis_name, result_file in RESULT_FILES.items():
        path = RESULTS_DIR / result_file
        if path.exists():
            all_data[analysis_name] = load_json_safe(path)
            print(f"  Loaded: {result_file}")
        else:
            print(f"  MISSING: {result_file}")
            all_data[analysis_name] = {"status": "not_generated"}

    # Extract key metrics
    print("\nExtracting key metrics...")
    key_metrics = extract_key_metrics(all_data)

    # Compile final output
    output = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %Human:%M:%S"),
            "total_duration_s": round(total_duration, 2),
            "scripts_run": len(run_log),
            "scripts_succeeded": sum(1 for r in run_log if r["status"] == "ok"),
            "scripts_failed": sum(1 for r in run_log if r["status"] == "failed"),
            "scripts_skipped": sum(1 for r in run_log if r["status"] == "skipped"),
            "tool_path": TOOL_PATH,
        },
        "run_log": run_log,
        "key_metrics": key_metrics,
        "full_results": all_data,
    }

    # Fix timestamp formatting
    output["meta"]["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    # Print run log summary
    print("\nRun log:")
    for entry in run_log:
        status = entry["status"].upper()
        dur = f"({entry.get('duration_s', 0):.1f}s)" if "duration_s" in entry else ""
        print(f"  {entry['name']:25s}: {status:8s} {dur}")

    # Print summary table
    print_summary_table(key_metrics)

    print(f"\nAll results written to {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
