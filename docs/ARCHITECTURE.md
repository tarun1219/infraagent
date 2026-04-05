# System Architecture

## Design Philosophy

InfraAgent is an **experimental platform**, not a novel algorithm. It composes existing
components to enable rigorous measurement of the prevention-over-repair asymmetry:

> RAG (prevention) is more effective per intervention than self-correction (repair)
> for security correctness in LLM-generated IaC.

## Pipeline

```
User Prompt (natural language)
        │
        ▼
┌───────────────────┐
│   Task Planner    │  keyword heuristics + resource dependency graphs
└────────┬──────────┘
         │ sub-tasks
         ▼
┌───────────────────┐
│    RAG Module     │  ChromaDB + all-MiniLM-L6-v2 embeddings
│  (top-5 docs)     │  Queries: task prompt + (in SC) error messages
└────────┬──────────┘
         │ context
         ▼
┌───────────────────┐
│  LLM Generator   │  Ollama (local) or API (GPT-4o / Claude)
│  (generate IaC)  │  4-bit quantized open-source models
└────────┬──────────┘
         │ generated code
         ▼
┌────────────────────────────────────────┐
│         Multi-Layer Validator          │
│  L1: Syntax    (yamllint/hadolint)     │
│  L2: Schema    (kubeconform/tf plan)   │
│  L2.5: Dry-run (kubectl dry-run=server)│
│  L3: Security  (Checkov + Trivy)       │
│  L4: Best Prac (OPA/Conftest)          │
└────────┬───────────────────────────────┘
         │
    ┌────┴─────┐
    │  Pass?   │
    └──┬───┬───┘
     Yes   No
      │     │
      ▼     ▼
   Return  Self-Correction Loop (up to N rounds)
   Output   │
            │  error report + fresh RAG query
            ▼
         LLM corrects → re-validate → repeat
```

## Validation Layer Contribution (Ablation)

| Layers Active | Functional Correctness | Δ from Full |
|---------------|------------------------|-------------|
| Full (L1+L2+L3+L4) | 72.0% | — |
| No Security (L1+L2+L4) | 68.3% | −3.7 pp |
| No Best Practices (L1+L2+L3) | 70.1% | −1.9 pp |
| **No Schema (L1+L3+L4)** | **54.2%** | **−17.8 pp** |
| Syntax Only (L1) | 38.1% | −33.9 pp |

Schema (L2) is the most critical layer — removing it causes the largest single drop.

## Why Prevention > Repair

Security error messages (e.g., "CKV_K8S_30: containers must not run as root") tell the
model *what is wrong* but not *what the correct configuration should be*. That knowledge
lives in documentation — which RAG retrieves. Self-correction without RAG can only work
from error signal; RAG provides the solution.

This explains the core asymmetry measured in the paper.

## Key Files

| File | Purpose |
|------|---------|
| `infraagent/agent.py` | Main pipeline orchestration |
| `infraagent/generator.py` | LLM inference (Ollama + API routing) |
| `infraagent/validators.py` | Multi-layer validation coordinator |
| `iachench/benchmark.py` | Task loader + evaluation harness |
| `iachench/metrics.py` | Canonical metric definitions |
| `scripts/simulate_results.py` | Simulation of results for figure generation |
| `scripts/generate_figures.py` | All 14 paper figures |
