"""
Example 02 — Self-Correction Loop (step by step)

Watch self-correction in action for a task that fails on the first attempt:
  Round 0: Generate (one-shot)  → validation failure
  Round 1: Error → RAG requery → corrected generation → re-validate
  Round 2: (if needed) repeat
  ...
  Round N: PASS (or exhausted)

This script uses a deliberately hard task (L4: NetworkPolicy + PSS-restricted
Deployment) to reliably trigger at least one correction round.

Run:
  python examples/02_self_correction_loop.py
  python examples/02_self_correction_loop.py --rounds 5 --verbose
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from infraagent.agent import InfraAgent, RoundRecord
from infraagent.generator import ModelID
from infraagent.validators import ValidationReport


# L4 task — high probability of failing one-shot
TASK_INTENT = (
    "Create a Kubernetes Deployment 'internal-api' in namespace 'backend' with "
    "a Pod Security Standards restricted profile (runAsNonRoot, readOnlyRootFilesystem, "
    "drop ALL capabilities, no privilege escalation, seccompProfile RuntimeDefault). "
    "Add a NetworkPolicy: (1) default deny all ingress and egress, "
    "(2) allow ingress from pods labelled app=gateway on port 8080, "
    "(3) allow egress to kube-dns (port 53 UDP/TCP). "
    "Use networking.k8s.io/v1 for NetworkPolicy and apps/v1 for Deployment."
)


def fmt_report(report: ValidationReport, round_num: int) -> str:
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"  ┌── Round {round_num} [{status}] ─────────────────────────────────────",
        f"  │  Syntax:       {_tf(report.syntax_valid)}",
        f"  │  Schema:       {_tf(report.schema_valid)}",
        f"  │  Security:     {report.security_score:.2f}  "
        f"({'PASS' if report.security_score >= 0.5 else 'FAIL'})",
        f"  │  Best-prac:    {report.best_practice_score:.2f}  "
        f"({'PASS' if report.best_practice_score >= 0.6 else 'FAIL'})",
        f"  │  Errors:       {len(report.errors)}",
    ]
    if report.errors:
        lines.append(f"  │")
        for e in report.errors[:5]:
            msg = e.message[:65]
            lines.append(f"  │  [{e.layer.value:8s}] {e.rule_id}: {msg}")
        if len(report.errors) > 5:
            lines.append(f"  │  ... and {len(report.errors) - 5} more errors")
    lines.append(f"  └{'─' * 55}")
    return "\n".join(lines)


def _tf(v: bool) -> str:
    return "✓ PASS" if v else "✗ FAIL"


def print_rag_query(query: str, round_num: int) -> None:
    if not query:
        return
    preview = query[:200].replace("\n", " ") + ("..." if len(query) > 200 else "")
    print(f"\n  RAG query (round {round_num}):")
    print(f"  {textwrap.fill(preview, width=68, subsequent_indent='  ')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-correction loop example")
    parser.add_argument("--model",   choices=["deepseek","codellama","mistral","phi3","gpt4o","claude"],
                        default="deepseek")
    parser.add_argument("--rounds",  type=int, default=3)
    parser.add_argument("--no-rag",  action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--stub",    action="store_true",
                        help="Stub mode: no Ollama required (shows loop mechanics only)")
    args = parser.parse_args()

    model_map = {
        "deepseek": ModelID.DEEPSEEK_CODER, "codellama": ModelID.CODELLAMA,
        "mistral":  ModelID.MISTRAL,        "phi3":      ModelID.PHI3,
        "gpt4o":    ModelID.GPT4O,          "claude":    ModelID.CLAUDE,
    }

    print("=" * 72)
    print("  InfraAgent — Self-Correction Loop")
    print(f"  Model: {args.model}  |  max_rounds: {args.rounds}  |  RAG: {not args.no_rag}")
    print("=" * 72)
    print(f"\n  Task (L4 difficulty):")
    print(textwrap.fill(f"  {TASK_INTENT}", width=70, subsequent_indent="  "))

    agent = InfraAgent(
        model=model_map[args.model],
        max_rounds=args.rounds,
        use_rag=not args.no_rag,
        use_stub=args.stub,
        verbose=False,   # we handle our own per-round printing
    )

    # Monkey-patch to intercept RAG queries per round
    rag_queries: list[str] = []
    _orig_build = agent.rag.build_context_string
    def _tracked_build(query="", **kw):
        rag_queries.append(query)
        return _orig_build(query=query, **kw)
    agent.rag.build_context_string = _tracked_build

    print(f"\n  {'─' * 68}")
    print(f"  Running pipeline...")
    print(f"  {'─' * 68}\n")

    result = agent.run(TASK_INTENT, task_id="k8s-l4-example-02")

    # Print per-round breakdown
    for i, record in enumerate(result.rounds):
        print(fmt_report(record.report, i))

        if not record.report.passed and i < len(result.rounds) - 1:
            # Show error messages fed into next round
            errors = record.report.errors[:3]
            if errors:
                print(f"\n  ↳ Error feedback to correction prompt:")
                for e in errors:
                    print(f"      [{e.layer.value}] {e.rule_id}: {e.message[:70]}")

            # Show RAG query reformulation
            if not args.no_rag and i + 1 < len(rag_queries):
                print_rag_query(rag_queries[i + 1], i + 1)

        print()

    # Summary
    print("=" * 72)
    if result.success:
        print(f"  ✓ PASSED after {result.total_rounds_used} correction round(s)  "
              f"[{result.total_duration_s:.1f}s total]")
        if result.total_rounds_used == 0:
            print("  The model got it right on the first try (one-shot).")
        else:
            print(f"  Self-correction recovered the task in round {result.total_rounds_used}.")
    else:
        print(f"  ✗ FAILED after {result.total_rounds_used} correction round(s)")
        remaining = len(result.final_report.errors)
        print(f"  {remaining} unresolved error(s) remain.")
        if not args.no_rag:
            print("  Tip: These are likely security misconfigs (8% SC recovery rate).")
            print("       RAG prevention (--rounds 0 + RAG) often outperforms SC repair.")
    print("=" * 72)


if __name__ == "__main__":
    main()
