"""
Example 03 — Model Comparison

Generate the same task with multiple models and compare:
  - Functional correctness (all layers pass?)
  - Security score
  - Rounds required to pass
  - Time per task

Uses the same L3 task for all models so results are directly comparable.
Mirrors the model comparison experiment from the paper (Figure 5).

Run:
  python examples/03_compare_models.py
  python examples/03_compare_models.py --models deepseek mistral phi3
  python examples/03_compare_models.py --condition sc+rag+3r
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from infraagent.agent import InfraAgent, AgentResult
from infraagent.generator import ModelID


TASK_INTENT = (
    "Create a Kubernetes Deployment 'cache-service' using redis:7.2-alpine with "
    "a full security context (runAsNonRoot: true, runAsUser: 999, "
    "readOnlyRootFilesystem: true, drop ALL capabilities), "
    "resource limits (CPU: 200m, memory: 256Mi), a liveness probe on TCP port 6379, "
    "and a ConfigMap 'redis-config' mounted at /etc/redis/ with maxmemory 128mb."
)

AVAILABLE_MODELS: Dict[str, ModelID] = {
    "deepseek":  ModelID.DEEPSEEK_CODER,
    "codellama": ModelID.CODELLAMA,
    "mistral":   ModelID.MISTRAL,
    "phi3":      ModelID.PHI3,
    "gpt4o":     ModelID.GPT4O,
    "claude":    ModelID.CLAUDE,
}

CONDITIONS = {
    "one-shot":   {"max_rounds": 0, "use_rag": False},
    "one-shot+rag": {"max_rounds": 0, "use_rag": True},
    "sc+rag+3r":  {"max_rounds": 3, "use_rag": True},
}


def run_model(name: str, model_id: ModelID, max_rounds: int, use_rag: bool,
              stub: bool) -> AgentResult:
    agent = InfraAgent(
        model=model_id,
        max_rounds=max_rounds,
        use_rag=use_rag,
        use_stub=stub,
        verbose=False,
    )
    return agent.run(TASK_INTENT, task_id=f"k8s-l3-compare-{name}")


def print_comparison_table(results: Dict[str, AgentResult]) -> None:
    print(f"\n{'─' * 72}")
    print(f"  {'Model':<16} {'Pass':>5} {'Sec':>6} {'BP':>6} {'Rounds':>7} {'Time':>7}")
    print(f"{'─' * 72}")
    for name, result in results.items():
        r = result.final_report
        pass_str = "✓" if result.success else "✗"
        print(
            f"  {name:<16} {pass_str:>5} "
            f"{r.security_score:>5.2f}  "
            f"{r.best_practice_score:>5.2f}  "
            f"{result.total_rounds_used:>6}  "
            f"{result.total_duration_s:>6.1f}s"
        )
    print(f"{'─' * 72}")

    # Best model
    passed = {n: r for n, r in results.items() if r.success}
    if passed:
        fastest = min(passed, key=lambda n: passed[n].total_duration_s)
        best_sec = max(passed, key=lambda n: passed[n].final_report.security_score)
        print(f"\n  Fastest to pass:          {fastest} ({passed[fastest].total_duration_s:.1f}s)")
        print(f"  Highest security score:   {best_sec} ({passed[best_sec].final_report.security_score:.2f})")
    else:
        print("\n  No model passed on this task with the current configuration.")
        print("  Try: --condition sc+rag+3r  or  --models deepseek gpt4o")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare models on a single task")
    parser.add_argument(
        "--models", nargs="+",
        choices=list(AVAILABLE_MODELS),
        default=["deepseek", "mistral", "phi3"],
        help="Models to compare",
    )
    parser.add_argument(
        "--condition",
        choices=list(CONDITIONS),
        default="sc+rag+3r",
        help="Evaluation condition",
    )
    parser.add_argument("--stub", action="store_true",
                        help="Stub mode: no Ollama required (results are illustrative only)")
    args = parser.parse_args()

    cfg = CONDITIONS[args.condition]
    print("=" * 72)
    print("  InfraAgent — Model Comparison")
    print(f"  Condition: {args.condition}  "
          f"(max_rounds={cfg['max_rounds']}, rag={cfg['use_rag']})")
    print(f"  Models:    {', '.join(args.models)}")
    print("=" * 72)
    print(f"\n  Task (L3 difficulty):")
    import textwrap
    print(textwrap.fill(f"  {TASK_INTENT}", width=70, subsequent_indent="  "))
    print()

    results: Dict[str, AgentResult] = {}
    for model_name in args.models:
        model_id = AVAILABLE_MODELS[model_name]
        print(f"  Running {model_name}...", end="", flush=True)
        t0 = time.perf_counter()
        result = run_model(
            model_name, model_id,
            max_rounds=cfg["max_rounds"],
            use_rag=cfg["use_rag"],
            stub=args.stub,
        )
        elapsed = time.perf_counter() - t0
        status = "PASS" if result.success else "FAIL"
        print(f" [{status}] in {result.total_rounds_used} round(s), {elapsed:.1f}s")
        results[model_name] = result

    print_comparison_table(results)

    # Paper reference numbers
    print("\n  Paper reference (DeepSeek Coder v2, sc+rag+3r, 300 tasks):")
    print("    Functional accuracy: 69.4%  Security pass rate: 59.3%")
    print("    GPT-4o (commercial ceiling): 93.9% functional accuracy")
    print("\n  Run with --models deepseek gpt4o claude to compare all tiers.")


if __name__ == "__main__":
    main()
