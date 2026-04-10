"""
Example 01 — Single Task Generation with RAG

Shows the full pipeline for one Kubernetes task:
  Task description
    → RAG retrieval (top-5 documentation chunks)
    → LLM generation
    → Multi-layer validation
    → Structured result

Run:
  python examples/01_single_task_generation.py
  python examples/01_single_task_generation.py --model phi3 --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from infraagent.agent import InfraAgent
from infraagent.generator import ModelID

MODELS = {
    "deepseek": ModelID.DEEPSEEK_CODER,
    "codellama": ModelID.CODELLAMA,
    "mistral":   ModelID.MISTRAL,
    "phi3":      ModelID.PHI3,
    "gpt4o":     ModelID.GPT4O,
    "claude":    ModelID.CLAUDE,
}

# A realistic L3 task: Deployment + HPA with full security context
TASK_INTENT = (
    "Create a Kubernetes Deployment named 'web-api' for an nginx:1.25.3 container "
    "with a full security context (runAsNonRoot: true, runAsUser: 1000, "
    "readOnlyRootFilesystem: true, drop ALL capabilities), CPU limit 500m, "
    "memory limit 128Mi, liveness and readiness probes on /healthz and /ready. "
    "Add a HorizontalPodAutoscaler using autoscaling/v2 targeting 70% CPU "
    "with minReplicas=2 and maxReplicas=10."
)


def print_section(title: str, content: str, width: int = 72) -> None:
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")
    print(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single task generation example")
    parser.add_argument("--model",   choices=list(MODELS), default="deepseek")
    parser.add_argument("--rounds",  type=int, default=0,
                        help="Self-correction rounds (0 = one-shot, paper default = 3)")
    parser.add_argument("--no-rag",  action="store_true", help="Disable RAG context")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--stub",    action="store_true",
                        help="Use stub mode (no Ollama required — for testing)")
    args = parser.parse_args()

    print("=" * 72)
    print("  InfraAgent — Single Task Generation")
    print(f"  Model:   {args.model}")
    print(f"  RAG:     {'disabled' if args.no_rag else 'enabled'}")
    print(f"  Rounds:  {args.rounds} ({'one-shot' if args.rounds == 0 else 'self-correction'})")
    print("=" * 72)

    # ── Step 1: Show task ──────────────────────────────────────────────────
    print_section("Task", textwrap.fill(TASK_INTENT, width=70))

    # ── Step 2: Initialize agent ──────────────────────────────────────────
    agent = InfraAgent(
        model=MODELS[args.model],
        max_rounds=args.rounds,
        use_rag=not args.no_rag,
        use_stub=args.stub,
        verbose=args.verbose,
    )

    # ── Step 3: Show RAG context (before generation) ──────────────────────
    if not args.no_rag:
        ctx = agent.rag.build_context_string(
            query=TASK_INTENT,
            language_filter="kubernetes",
        )
        if ctx:
            preview = ctx[:800] + ("..." if len(ctx) > 800 else "")
            print_section(
                f"RAG Context (top-{agent.rag.top_k} chunks, "
                f"{len(ctx)} chars retrieved)",
                preview,
            )
        else:
            print_section("RAG Context", "[No relevant documentation retrieved]")

    # ── Step 4: Run agent ──────────────────────────────────────────────────
    print(f"\n  Generating IaC... (this may take 30–60s on Ollama CPU)\n")
    result = agent.run(TASK_INTENT, task_id="k8s-l3-example-01")

    # ── Step 5: Show generated code ────────────────────────────────────────
    code_preview = result.final_code[:1500] + ("..." if len(result.final_code) > 1500 else "")
    print_section("Generated IaC", code_preview)

    # ── Step 6: Show validation results ───────────────────────────────────
    report = result.final_report
    status = "PASS" if result.success else "FAIL"
    print_section(
        f"Validation Results [{status}]",
        f"  Syntax valid:       {report.syntax_valid}\n"
        f"  Schema valid:       {report.schema_valid}\n"
        f"  Security score:     {report.security_score:.2f} "
        f"({'≥0.50 PASS' if report.security_score >= 0.5 else '<0.50 FAIL'})\n"
        f"  Best-practice:      {report.best_practice_score:.2f} "
        f"({'≥0.60 PASS' if report.best_practice_score >= 0.6 else '<0.60 FAIL'})\n"
        f"  Dry-run server:     {report.dry_run_server_valid} "
        f"(None = kind not running)\n"
        f"  Rounds used:        {result.total_rounds_used}\n"
        f"  Total time:         {result.total_duration_s:.1f}s",
    )

    if report.errors:
        print_section(
            f"Validation Errors ({len(report.errors)})",
            "\n".join(
                f"  [{e.layer.value.upper()}] {e.rule_id}: {e.message[:80]}"
                for e in report.errors[:10]
            ),
        )

    # ── Step 7: Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  Result: {'✓ PASSED' if result.success else '✗ FAILED'} "
          f"in {result.total_rounds_used} correction round(s)")
    print(f"  Tip: Add --rounds 3 to enable self-correction.")
    print(f"       Add --model gpt4o for commercial model (set OPENAI_API_KEY).")
    print("=" * 72)

    # Optionally dump full result as JSON
    if args.verbose:
        out = Path("results/example_01_result.json")
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\n  Full result saved to: {out}")


if __name__ == "__main__":
    main()
