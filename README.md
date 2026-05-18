# IaCBench + InfraAgent

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/Paper-Preprint-brightgreen.svg)](#citation)

**Paper:** *"IaCBench: How Reliably Do Large Language Models Generate Secure Infrastructure-as-Code? A Multi-Language Benchmark and Empirical Study of Frontier Models"*

---

## Overview

**IaCBench** is the first multi-language benchmark for evaluating LLM-generated Infrastructure-as-Code (IaC). It covers 150 tasks across Kubernetes YAML, Terraform HCL, and Dockerfile at five difficulty levels, with a four-layer validation pipeline (syntax → schema → security → functional correctness).

**InfraAgent** is the evaluation harness that runs any LLM against IaCBench. It supports:
- Documentation-augmented retrieval (RAG)
- Iterative self-correction with structured validation feedback
- Multi-provider API support: OpenAI, Anthropic, Groq, Ollama (local)

---

## Key Results

| Model | Baseline Functional | Baseline Security | RAG Functional | RAG Security |
|-------|--------------------|--------------------|----------------|--------------|
| GPT-4o | 97.3% | 84.1% | 97.3% | 84.9% |
| GPT-4o-mini | 86.0% | 84.0% | 87.3% | 84.6% |
| Claude Haiku 4.5 | 98.0% | 94.3% | 99.3% | 96.2% |
| **Claude Sonnet 4.5** | **98.7%** | **96.5%** | **100.0%** | **96.5%** |

*150 tasks, up to 2 self-correction rounds per condition.*

### Key Findings

1. **Frontier models achieve 86–99% functional correctness** — far exceeding ~35% reported for open-source models on prior benchmarks.
2. **Security is the primary differentiator**: Claude models score 94–97% vs ~84% for GPT models, a gap that persists across all conditions.
3. **RAG is a ceiling-breaker**: pushes Claude Sonnet 4.5 to perfect **100%** across all 150 tasks.
4. **Kubernetes is the hardest language**: GPT-4o-mini achieves only 70% on Kubernetes vs 100% on Dockerfile.
5. **Terraform IAM security** is the hardest security domain (~65% for GPT models due to strict Checkov IAM checks).

---

## Repository Structure

```
infraagent/
├── iachench/
│   └── tasks/
│       ├── k8s_tasks.json       # 50 Kubernetes tasks (L1–L5)
│       ├── tf_tasks.json        # 50 Terraform tasks (L1–L5)
│       └── df_tasks.json        # 50 Dockerfile tasks (L1–L5)
├── rag_corpus/
│   ├── kubernetes/              # K8s API docs, security guides
│   ├── terraform/               # AWS/GCP/Azure provider docs
│   └── dockerfile/              # Docker best practices
├── analysis/
│   ├── run_groq_evaluation.py   # Multi-provider eval (Groq/Anthropic/OpenRouter)
│   ├── run_llm_evaluation.py    # OpenAI GPT-4o evaluation
│   ├── run_mini_evaluation.py   # OpenAI GPT-4o-mini evaluation
│   ├── run_ollama_evaluation.py # Local model evaluation via Ollama
│   ├── generate_figures.py      # Reproduce all paper figures
│   └── results/                 # Evaluation result JSONs
├── figures/                     # Paper figures (PDF, reproducible via generate_figures.py)
└── docs/
    ├── INSTALL.md
    └── QUICKSTART.md
```

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/tarun1219/infraagent.git
cd infraagent
pip install -r requirements.txt
```

Required validation tools:
```bash
# macOS
brew install checkov kubeconform hadolint

# Python
pip install yamllint
```

### 2. Run evaluation

**OpenAI models:**
```bash
export OPENAI_API_KEY=your-key-here
python3 analysis/run_llm_evaluation.py        # GPT-4o
python3 analysis/run_mini_evaluation.py       # GPT-4o-mini
```

**Anthropic models:**
```bash
export ANTHROPIC_API_KEY=your-key-here
export PROVIDER=anthropic
python3 analysis/run_groq_evaluation.py
```

**Groq (free tier):**
```bash
export GROQ_API_KEY=your-key-here
export PROVIDER=groq
python3 analysis/run_groq_evaluation.py
```

**Local models via Ollama:**
```bash
ollama pull deepseek-coder:6.7b-instruct-q4_K_M
python3 analysis/run_ollama_evaluation.py
```

### 3. Reproduce paper figures

```bash
python3 analysis/generate_figures.py
# Outputs 8 PDF figures to figures/
```

---

## Evaluation Output

Each script produces a JSON results file in `analysis/results/`:

```json
{
  "model_stats": {
    "claude-sonnet-4-5": {
      "baseline": { "pct_functional": 0.987, "mean_security": 0.965 },
      "rag":      { "pct_functional": 1.000, "mean_security": 0.965 }
    }
  },
  "model_results": { ... }
}
```

---

## Adding a New Model

To evaluate any OpenAI-compatible API, extend the `PROVIDERS` dict in `run_groq_evaluation.py`:

```python
PROVIDERS = {
    "your_provider": {
        "url": "https://api.yourprovider.com/v1/chat/completions",
        "key_env": "YOUR_API_KEY",
        "models": ["model-name-here"],
    },
    ...
}
```

Then run:
```bash
export PROVIDER=your_provider
export YOUR_API_KEY=your-key-here
python3 analysis/run_groq_evaluation.py
```

---

## Benchmark Design

### Task Distribution

| Language | L1 | L2 | L3 | L4 | L5 | Total |
|----------|----|----|----|----|----|-------|
| Kubernetes | 10 | 10 | 10 | 10 | 10 | 50 |
| Terraform  | 10 | 10 | 10 | 10 | 10 | 50 |
| Dockerfile | 10 | 10 | 10 | 10 | 10 | 50 |

### Difficulty Levels

| Level | Label | Description |
|-------|-------|-------------|
| L1 | Basic | Single resource, no security constraints |
| L2 | Intermediate | 2+ cooperating resources |
| L3 | Advanced | Stateful resources, scaling, volumes |
| L4 | Security-Focused | L2/L3 + explicit security requirements |
| L5 | Expert | Multi-component systems with cross-cutting concerns |

### Validation Pipeline

| Layer | Tool | Criterion |
|-------|------|-----------|
| 1 – Syntax | yamllint / hcl2 / dockerfile-parse | No parse errors |
| 2 – Schema | kubeconform / terraform validate / hadolint | API-conformant |
| 3 – Security | Checkov / OPA/Rego / Hadolint | Security score ≥ 0.5 |
| 4 – Functional | Structural diffing + semantic checks | Meets task specification |

A task is **functionally correct** only if it passes all four layers.

---

## System Requirements

- Python 3.10+
- macOS / Linux
- For local model evaluation: [Ollama](https://ollama.com) + 8 GB+ RAM
- Validation tools: `checkov`, `kubeconform`, `hadolint`, `yamllint`

---

## Citation

If you use IaCBench or InfraAgent in your research, please cite:

```bibtex
@inproceedings{iacbench2025,
  title     = {IaCBench: How Reliably Do Large Language Models Generate Secure
               Infrastructure-as-Code? A Multi-Language Benchmark and
               Empirical Study of Frontier Models},
  author    = {Anonymous},
  booktitle = {Proceedings of the ACM/IEEE International Conference on
               Software Engineering},
  year      = {2025}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
