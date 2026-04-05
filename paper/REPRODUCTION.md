# Paper Reproduction Guide

This document explains how to reproduce all results, figures, and tables from the paper:

> **Prevention Over Repair: Quantifying the RAG-vs-Self-Correction Asymmetry in LLM-Generated Infrastructure-as-Code**

## Quick Reproduction (Figures Only)

Reproduces all 14 paper figures from pre-computed results. No model inference required.

```bash
git clone https://github.com/tarun1219/infraagent
cd infraagent
pip install -r requirements.txt
bash scripts/reproduce_paper.sh
```

Figures will be written to `results/figures/fig{01..14}.{pdf,png}`.

## Full Reproduction (Live Model Inference)

Requires Ollama with all 6 local models downloaded (~120GB total disk space).

### 1. Install Ollama and pull models

```bash
# Install Ollama: https://ollama.ai/download
ollama pull deepseek-coder-v2:16b-lite-instruct-q4_K_M   # ~9.1GB
ollama pull codellama:13b-instruct-q4_K_M                 # ~7.4GB
ollama pull mistral:7b-instruct-q4_K_M                    # ~4.1GB
ollama pull phi3:3.8b-instruct-q4_K_M                     # ~2.3GB
ollama pull llama3.1:70b-instruct-q4_K_M                  # ~40GB
ollama pull qwen2.5-coder:32b-instruct-q4_K_M             # ~18GB
```

### 2. (Optional) Set up Kubernetes validation

```bash
bash scripts/setup_kind_cluster.sh
```

### 3. (Optional) Set up Terraform LocalStack validation

```bash
bash scripts/setup_localstack.sh
```

### 4. Run experiments

```bash
# Primary model, primary condition, 5 runs (Table 3, main results)
python scripts/run_experiments.py --model deepseek --condition sc+rag+3r --runs 5

# All conditions, all models (full paper results — ~48 hours on 8x A100)
python scripts/run_experiments.py --condition all --model all --runs 5
```

### 5. Generate figures and statistics

```bash
bash scripts/reproduce_paper.sh --live
```

## Compile the Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Mapping: Paper Elements → Code

| Paper Element | Source |
|--------------|--------|
| Table 1 (task distribution) | `iachench/benchmark.py:get_task_distribution()` |
| Table 2 (validation layers) | `iachench/validators/` |
| Table 3 (main results) | `results/experiment_results.json["main_results"]` |
| Table 4 (SC by error class) | `results/experiment_results.json["correction_success_rate"]` |
| Figure 1 (overall accuracy) | `scripts/generate_figures.py:fig_overall_accuracy()` |
| Figure 2 (SC rounds) | `scripts/generate_figures.py:fig_round_improvement()` |
| Figure 3 (security delta) | `scripts/generate_figures.py:fig_security_delta()` |
| Figure 4 (by difficulty) | `scripts/generate_figures.py:fig_by_difficulty()` |
| Figure 11 (CI bars) | `scripts/generate_figures.py:fig_confidence_intervals()` |
| Figure 12 (LocalStack) | `scripts/generate_figures.py:fig_terraform_deploy()` |
| Figure 13 (layer ablation) | `scripts/generate_figures.py:fig_layer_ablation()` |
| Figure 14 (RAG quality) | `scripts/generate_figures.py:fig_rag_quality()` |

## Compute Environment

All local model experiments were run on:
- GPU: NVIDIA A100 80GB
- RAM: 256GB
- Storage: 2TB NVMe
- CUDA: 12.2, Driver: 535.86.10
- Ollama: 0.1.27

GPT-4o and Claude 3.5 Sonnet used official APIs with temperature=0.

## Known Limitations

- Live inference requires ~48 GPU-hours for all 40 (model × condition) combinations × 5 runs.
- kubectl dry-run requires a running Kubernetes cluster (kind setup script provided).
- LocalStack deployment validation requires Docker and ~4GB RAM for LocalStack container.
- Results may vary slightly due to model quantization differences across hardware.

## Questions

Open an issue at https://github.com/tarun1219/infraagent/issues if you encounter problems reproducing results.
