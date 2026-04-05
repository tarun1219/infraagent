# Reproducing Paper Results

All commands run from the repo root.

## One-Time Setup

```bash
pip install -r requirements.txt

# Optional: enable Layer 2.5 K8s dry-run validation
bash scripts/setup_kind_cluster.sh

# Optional: enable Terraform LocalStack deployment testing
bash scripts/setup_localstack.sh
```

## Full Reproduction (~12 hours on M2 Pro, DeepSeek only)

```bash
bash scripts/reproduce_paper.sh
```

## Individual Conditions

```bash
# DeepSeek, 5 runs each condition
python scripts/run_experiments.py --condition one-shot     --model deepseek --runs 5
python scripts/run_experiments.py --condition one-shot+rag --model deepseek --runs 5
python scripts/run_experiments.py --condition sc+3r        --model deepseek --runs 5
python scripts/run_experiments.py --condition sc+rag+3r    --model deepseek --runs 5
python scripts/run_experiments.py --condition sc+rag+5r    --model deepseek --runs 5
```

## Regenerate All Figures

```bash
python scripts/plot_figures.py --output paper_figures/
# Produces fig1–fig14 in paper_figures/ as PDF + PNG
```

## Reproduce Statistics (Table: significance tests)

```bash
python scripts/compute_statistics.py \
  --results results/experiment_results.json \
  --output results/STATISTICS.md
```

## Figure Reference

| Figure | Key Finding | Data Key |
|--------|-------------|----------|
| Fig 1  | Main results bar chart | `main_results` |
| Fig 2  | Self-correction round curves | `round_curves` |
| Fig 3  | Security by language (RAG impact) | `security_rag_impact_deepseek` |
| Fig 4  | Difficulty breakdown | `per_difficulty_deepseek` |
| Fig 5  | Model comparison + commercial ceiling | `model_comparison` |
| Fig 6  | Failure taxonomy (8 categories) | `failure_taxonomy` |
| Fig 7  | SC recovery rate per model | `correction_success_rate` |
| Fig 8  | Language heatmap | `per_language_deepseek` |
| Fig 9  | K8s: kubeconform vs dry-run=server | `k8s_dry_run_server` |
| Fig 10 | Failure mode distribution per model | `failure_mode_analysis` |
| Fig 11 | Confidence intervals (5 runs) | `multi_run_statistics` |
| Fig 12 | Terraform validate vs LocalStack deploy | `terraform_deploy_localstack` |
| Fig 13 | Validation layer ablation | `layer_ablation` |
| Fig 14 | RAG retrieval quality (P@5, MRR) | `rag_retrieval_quality` |

## Expected Key Numbers

| Metric | Value | Location |
|--------|-------|----------|
| One-shot functional correctness | 35.8% | Table 3 |
| SC+RAG(3r) functional correctness | 72.0% | Table 3 |
| RAG security gain (one step) | +19.9 pp | Section 7.2 |
| SC security gain (3 rounds, no RAG) | +18.4 pp | Section 7.3 |
| SC recovery rate (DeepSeek) | 7.9% | Section 7.3 |
| Schema ablation impact | -17.8 pp | Table: ablation |
| GPT-4o ceiling | 93.9% | Section 7.9 |
| LocalStack deployment gap | 8.3 pp | Section 7.11 |
| RAG Precision@5 (average) | 0.72 | Table: rag_quality |
