# InfraAgent: Prevention Over Repair in LLM-Generated IaC

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/tarun1219/infraagent/workflows/Tests/badge.svg)](https://github.com/tarun1219/infraagent/actions)
[![Paper](https://img.shields.io/badge/Paper-ICSE%202026-brightgreen.svg)](#citation)

**Paper:** *"Prevention Over Repair: Quantifying the RAG-vs-Self-Correction Asymmetry in LLM-Generated Infrastructure-as-Code"*

## Key Finding

RAG-based prevention (**+19.9 pp** security in one step) outperforms iterative self-correction repair (**+18.4 pp** over 3 rounds). Self-correction recovery rate is only **7.9%** for initially-failing tasks, dropping to **8%** for security misconfigurations — the dominant failure class (46% of errors).

## What's Included

| Component | Description |
|-----------|-------------|
| **IaCBench** | 300 curated IaC tasks (Kubernetes, Terraform, Dockerfile × L1–L5 difficulty) |
| **InfraAgent** | Reference framework: RAG + multi-layer validation + self-correction |
| **Scripts** | Reproduce all 14 paper figures and statistical tables |
| **Results** | Simulation results for 8 models × 5 conditions |

## Results Summary

| Condition | Syntax | Schema | Security | Functional |
|-----------|--------|--------|----------|------------|
| One-Shot | 60.3% | 51.5% | 16.8% | 35.8% |
| One-Shot + RAG | 71.2% | 64.7% | 36.7% | 45.9% |
| SC (3r) | 86.0% | 77.9% | 53.3% | 60.4% |
| **SC + RAG (3r)** | **91.1%** | **88.2%** | **74.3%** | **72.0%** |
| SC + RAG (5r) | 94.0% | 91.4% | 78.6% | 74.5% |

*Primary model: DeepSeek-Coder-V2-16B. Commercial ceilings: GPT-4o 93.9%, Claude-3.5-Sonnet 91.2%.*

## Quick Start

```bash
git clone https://github.com/tarun1219/infraagent.git
cd infraagent
pip install -r requirements.txt

# Run 10 sample tasks
python scripts/run_experiments.py --condition sc+rag+3r --model deepseek --num_tasks 10

# Reproduce all paper figures
python scripts/plot_figures.py --output paper_figures/
```

## System Requirements

- Python 3.10+
- 16 GB RAM (for local LLM inference via Ollama)
- Docker + kind (optional, for Kubernetes server-side dry-run validation)
- LocalStack (optional, for Terraform deployment testing)

## Documentation

- 📖 [Installation](docs/INSTALL.md)
- 🚀 [5-Minute Quickstart](docs/QUICKSTART.md)
- 📊 [Reproduce Paper Results](docs/PAPER.md)
- 🏗️ [Architecture](docs/ARCHITECTURE.md)
- ❓ [FAQ](docs/FAQ.md)

## Prevention Over Repair: The Core Asymmetry

```
Error Class          | % of Failures | SC Recovery | Why
---------------------|---------------|-------------|------------------------------
Syntax Errors        |     23%       |     72%     | Precise line/token feedback
Schema Violations    |     31%       |     55%     | Named field + API version
Security Misconfigs  |     46%       |      8%     | No "what to add" signal
Cross-Resource       |   (subset)    |      2%     | Global state, no local fix
```

This table explains why RAG (which provides documentation of *what to add*) prevents
security failures more effectively than SC (which only signals *what is wrong*).

## Citation

```bibtex
@inproceedings{infraagent2026,
  title     = {Prevention Over Repair: Quantifying the RAG-vs-Self-Correction
               Asymmetry in LLM-Generated Infrastructure-as-Code},
  author    = {Anonymous},
  booktitle = {Proceedings of ICSE 2026},
  year      = {2026}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
