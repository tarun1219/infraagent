#!/usr/bin/env python3
"""
Compute statistical significance tests from multi-run experiment results.

Reads results/raw/{model}_{condition}_run*.jsonl and prints:
  - Mean ± 95% CI for each (model, condition) pair
  - Paired t-tests between key condition comparisons
  - Cohen's d effect sizes

Usage:
  python scripts/compute_statistics.py
  python scripts/compute_statistics.py --model deepseek --alpha 0.05
"""
from __future__ import annotations
import argparse
import json
import logging
import math
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Critical t-values for two-tailed test (df = n_runs - 1)
T_CRIT = {4: 2.776, 9: 2.262, 14: 2.145, 19: 2.093}


def ci95(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, half_width, std) for a list of values."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    t = T_CRIT.get(n - 1, 2.0)
    hw = t * std / math.sqrt(n)
    return mean, hw, std


def paired_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Return (t_statistic, cohen_d) for paired samples."""
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1) if n > 1 else 1e-9
    std_d = math.sqrt(var_d)
    t = mean_d / (std_d / math.sqrt(n)) if std_d > 0 else 0.0
    d = mean_d / std_d if std_d > 0 else 0.0
    return t, d


def load_run_results(raw_dir: Path, model: str, condition: str) -> list[float]:
    """Load per-run accuracy scores from JSONL files."""
    cond_slug = condition.replace("+", "_")
    scores = []
    for run_file in sorted(raw_dir.glob(f"{model}_{cond_slug}_run*.jsonl")):
        tasks = [json.loads(line) for line in run_file.read_text().splitlines() if line.strip()]
        if tasks:
            acc = sum(1 for t in tasks if t.get("functional")) / len(tasks)
            scores.append(acc * 100)
    return scores


def print_table(rows: list[tuple]) -> None:
    header = f"{'Model':<12} {'Condition':<15} {'Mean':>7} {'±CI':>7} {'Std':>7}"
    print(header)
    print("-" * len(header))
    for model, cond, mean, hw, std in rows:
        print(f"{model:<12} {cond:<15} {mean:>6.1f}% {hw:>6.2f}  {std:>6.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute statistical significance tests")
    parser.add_argument("--raw_dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--model",   default="all", help="Model alias or 'all'")
    parser.add_argument("--alpha",   type=float, default=0.05)
    args = parser.parse_args()

    if not args.raw_dir.exists():
        logger.warning(
            f"{args.raw_dir} not found. Run scripts/run_experiments.py first to generate raw results.\n"
            "Showing example output from simulate_results.py instead."
        )
        # Load from simulate_results summary
        results_file = Path("results/experiment_results.json")
        if results_file.exists():
            data = json.loads(results_file.read_text())
            stats = data.get("multi_run_statistics", {})
            print("\nPre-computed statistics from experiment_results.json:")
            for model, cond_data in stats.get("conditions", {}).items():
                for cond, s in cond_data.items():
                    print(
                        f"  {model:<12} {cond:<20} "
                        f"mean={s['mean']:.1f}%  "
                        f"95%CI=[{s['ci_lower']:.1f}, {s['ci_upper']:.1f}]  "
                        f"d={s['cohens_d']:.2f}  p={s['p_value']:.4f}"
                    )
        return

    models = ["deepseek", "codellama", "mistral", "phi3", "llama", "qwen", "gpt4o", "claude"]
    if args.model != "all":
        models = [args.model]

    conditions = ["one-shot", "one-shot_rag", "sc_3r", "sc_rag_3r", "sc_rag_5r"]
    rows = []

    for model in models:
        for cond in conditions:
            scores = load_run_results(args.raw_dir, model, cond)
            if len(scores) >= 2:
                mean, hw, std = ci95(scores)
                rows.append((model, cond, mean, hw, std))

    if rows:
        print_table(rows)

        # Key comparison: one-shot+rag vs sc+rag+3r
        print("\nKey paired comparisons (RAG prevention vs SC repair):")
        for model in models:
            a = load_run_results(args.raw_dir, model, "one-shot_rag")
            b = load_run_results(args.raw_dir, model, "sc_rag_3r")
            if len(a) >= 2 and len(a) == len(b):
                t, d = paired_t(a, b)
                t_crit = T_CRIT.get(len(a) - 1, 2.0)
                sig = "*" if abs(t) > t_crit else " "
                print(f"  {model:<12}  t={t:+.3f}  d={d:+.2f}  {sig}")


if __name__ == "__main__":
    main()
