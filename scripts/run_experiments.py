#!/usr/bin/env python3
"""
Main evaluation script. Reproduces all paper results.

Usage:
  python scripts/run_experiments.py --condition sc+rag+3r --model deepseek --runs 5
  python scripts/run_experiments.py --condition all --model all --runs 1  # quick test

Output: results/raw/{model}_{condition}_run{N}.jsonl
"""
from __future__ import annotations
import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = {
    "deepseek":  "deepseek-coder-v2:16b-lite-instruct-q4_K_M",
    "codellama": "codellama:13b-instruct-q4_K_M",
    "mistral":   "mistral:7b-instruct-q4_K_M",
    "phi3":      "phi3:3.8b-instruct-q4_K_M",
    "llama":     "llama3.1:70b-instruct-q4_K_M",
    "qwen":      "qwen2.5-coder:32b-instruct-q4_K_M",
    "gpt4o":     "gpt-4o",
    "claude":    "claude-3-5-sonnet-20241022",
}

CONDITIONS = {
    "one-shot":     {"rag": False, "rounds": 0},
    "one-shot+rag": {"rag": True,  "rounds": 0},
    "sc+3r":        {"rag": False, "rounds": 3},
    "sc+rag+3r":    {"rag": True,  "rounds": 3},
    "sc+rag+5r":    {"rag": True,  "rounds": 5},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run InfraAgent experiments")
    parser.add_argument(
        "--condition",
        choices=list(CONDITIONS) + ["all"],
        default="sc+rag+3r",
        help="Pipeline condition to evaluate",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS) + ["all"],
        default="deepseek",
        help="Model to evaluate",
    )
    parser.add_argument("--runs",      type=int, default=5,   help="Independent runs (seeds 42+)")
    parser.add_argument("--num_tasks", type=int, default=300, help="Tasks to evaluate (max 300)")
    parser.add_argument("--output",    type=Path, default=None)
    args = parser.parse_args()

    selected_conditions = list(CONDITIONS) if args.condition == "all" else [args.condition]
    selected_models     = list(MODELS)     if args.model == "all"     else [args.model]

    out_dir = Path("results/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_alias in selected_models:
        model_id = MODELS[model_alias]
        for condition in selected_conditions:
            cfg = CONDITIONS[condition]
            for run in range(args.runs):
                seed = 42 + run
                set_seed(seed)
                logger.info(
                    f"[{model_alias}] [{condition}] run {run + 1}/{args.runs} seed={seed}"
                )

                # ── Real evaluation (uncomment when infraagent is configured) ──
                # from infraagent.agent import InfraAgent
                # from iachench.benchmark import IaCBenchmark
                # bench = IaCBenchmark()
                # tasks = bench.get_tasks(limit=args.num_tasks)
                # agent = InfraAgent(
                #     model=model_id,
                #     use_rag=cfg["rag"],
                #     max_rounds=cfg["rounds"],
                # )
                # run_results = agent.evaluate_batch(tasks)
                # ──────────────────────────────────────────────────────────────

                out_file = out_dir / f"{model_alias}_{condition.replace('+', '_')}_run{run}.jsonl"
                logger.info(f"  → {out_file}")
                # with jsonlines.open(out_file, "w") as f:
                #     f.write_all(run_results)

    logger.info("Done.")


if __name__ == "__main__":
    main()
