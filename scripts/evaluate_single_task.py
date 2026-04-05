#!/usr/bin/env python3
"""
Evaluate InfraAgent on a single task for debugging and demonstration.

Usage:
  python scripts/evaluate_single_task.py --task examples/example_task_kubernetes.json
  python scripts/evaluate_single_task.py --task examples/example_task_kubernetes.json --model deepseek --condition sc+rag+3r
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

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


def load_task(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def print_result(task: dict, result: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Task: {task['id']} — {task['description']}")
    print(f"Type: {task['type']}  Difficulty: L{task['difficulty']}")
    print("-" * 60)
    print(f"Functional: {'PASS' if result.get('functional') else 'FAIL'}")
    print(f"Security:   {'PASS' if result.get('security') else 'FAIL'}")
    print(f"Round:      {result.get('round', 0)}")
    if result.get("errors"):
        print("\nErrors:")
        for e in result["errors"]:
            print(f"  [{e['layer']}] {e['message']}")
    if result.get("output"):
        print("\nGenerated output:")
        print(result["output"][:2000])
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate InfraAgent on a single task")
    parser.add_argument("--task",      type=Path, required=True, help="Path to task JSON file")
    parser.add_argument("--model",     choices=list(MODELS), default="deepseek")
    parser.add_argument("--condition", choices=list(CONDITIONS), default="sc+rag+3r")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    task = load_task(args.task)
    cfg  = CONDITIONS[args.condition]
    model_id = MODELS[args.model]

    logger.info(f"Task: {task['id']}")
    logger.info(f"Model: {model_id}  condition: {args.condition}")

    # ── Real evaluation (uncomment when infraagent is configured) ──
    # from infraagent.agent import InfraAgent
    # agent = InfraAgent(model=model_id, use_rag=cfg["rag"], max_rounds=cfg["rounds"])
    # result = agent.evaluate(task)
    # print_result(task, result)
    # ──────────────────────────────────────────────────────────────

    # Demo: show task structure
    print("\nTask loaded successfully:")
    print(json.dumps(task, indent=2))
    print(f"\nWould evaluate with: model={model_id}, rag={cfg['rag']}, rounds={cfg['rounds']}")
    print("Uncomment the InfraAgent import block to run live evaluation.")


if __name__ == "__main__":
    main()
