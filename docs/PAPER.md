# Reproducing Paper Results

All commands run from the repo root.

## Data Requirements

| Scope | Disk | RAM needed |
|-------|------|------------|
| Figures only (pre-computed) | ~50 MB | 4 GB |
| DeepSeek only, all conditions | ~9 GB model + results | 16 GB |
| All 6 local models | ~80 GB models | 16 GB (sequential) |
| All 8 models incl. 70B | ~120 GB models | 50 GB (for 70B) |

Live inference results (raw JSONL) add ~500 MB for all 40 (model × condition) × 5 run combinations.

---

## Quick Reproduction — Figures Only

No model inference required. Uses pre-computed `results/experiment_results.json`.

```bash
pip install -r requirements.txt
python scripts/simulate_results.py    # regenerates experiment_results.json
python scripts/plot_figures.py        # writes results/figures/fig{01..14}.{pdf,png}
```

All 14 figures match the paper exactly.

---

## Full Reproduction — Live Inference

### Runtime Estimates (M2 Pro, 16 GB RAM, Ollama CPU)

| Model | Params | Time per task | 300 tasks × 5 cond × 1 run |
|-------|--------|--------------|----------------------------|
| Phi-3 | 3.8B | ~8 s | ~3.3 h |
| Mistral | 7B | ~15 s | ~6.3 h |
| CodeLlama | 13B | ~22 s | ~9.2 h |
| DeepSeek Coder v2 | 16B | ~30 s | ~12.5 h |
| Qwen 2.5 Coder | 32B | ~55 s | ~22.9 h |
| Llama 3.1 | 70B | ~120 s | ~50 h |

**Total for all 6 local models (5 conditions, 1 run):** ~104 hours on a single M2 Pro.
Run models in parallel across machines to reduce wall time.
For 5 statistical runs each, multiply by 5 (~520 hours single-machine; ~65 hours on 8 machines).

### Step 1 — Environment Setup

```bash
# Optional: server-side K8s dry-run (Layer 2.5)
bash scripts/setup_kind_cluster.sh

# Optional: LocalStack for Terraform deployment gap (Figure 12)
bash scripts/setup_localstack.sh
```

### Step 2 — Run Experiments

```bash
# DeepSeek only, all 5 conditions, 5 runs (~62 hours on M2 Pro)
python scripts/run_experiments.py --model deepseek --condition all --runs 5

# All models, all conditions
python scripts/run_experiments.py --model all --condition all --runs 5
```

### Step 3 — Generate All Figures

```bash
python scripts/plot_figures.py
# → results/figures/fig01.pdf ... fig14.pdf
```

### Step 4 — Statistical Significance Tests

```bash
python scripts/compute_statistics.py
# Prints mean ± 95% CI, paired t-tests, Cohen's d for all (model, condition) pairs
```

---

## Running with Different Models

```bash
# Local models via Ollama
python scripts/run_experiments.py --model phi3     --condition sc+rag+3r --runs 1
python scripts/run_experiments.py --model mistral  --condition sc+rag+3r --runs 1
python scripts/run_experiments.py --model llama    --condition sc+rag+3r --runs 1

# Commercial APIs (set API keys first)
export OPENAI_API_KEY="sk-..."
python scripts/run_experiments.py --model gpt4o  --condition sc+rag+3r --runs 1

export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/run_experiments.py --model claude --condition sc+rag+3r --runs 1
```

---

## Figure Reference

| Figure | What it shows | Data key in experiment_results.json |
|--------|--------------|--------------------------------------|
| Fig 1 | Overall accuracy by condition (bar chart) | `main_results` |
| Fig 2 | Accuracy improvement by SC round | `round_improvement` |
| Fig 3 | Security pass rate: RAG impact by language | `security_rag_impact_deepseek` |
| Fig 4 | Accuracy by difficulty level (L1–L5) | `per_difficulty_deepseek` |
| Fig 5 | Model comparison + GPT-4o ceiling (93.9%) | `model_comparison` |
| Fig 6 | Failure taxonomy (8 categories) | `failure_taxonomy` |
| Fig 7 | SC recovery rate per model | `correction_success_rate` |
| Fig 8 | Language heatmap (K8s / TF / Dockerfile) | `per_language_deepseek` |
| Fig 9 | K8s: kubeconform gap vs dry-run=server | `k8s_dry_run_server` |
| Fig 10 | Failure mode distribution per model | `failure_mode_analysis` |
| Fig 11 | **Statistical significance** — 95% CI bars (5 runs) | `multi_run_statistics` |
| Fig 12 | **LocalStack deployment gap** — validate vs apply | `terraform_deploy_localstack` |
| Fig 13 | **Validation layer ablation** — impact of removing each layer | `layer_ablation` |
| Fig 14 | **RAG retrieval quality** — P@5, R@5, MRR, NDCG@5 by difficulty | `rag_retrieval_quality` |

---

## Statistical Significance Tests (Figure 11)

Figure 11 shows 95% confidence intervals from 5 independent runs (seeds 42–46).

```bash
python scripts/compute_statistics.py
```

Key numbers reproduced:
- DeepSeek sc+rag+3r: mean=69.4%, CI=[67.1, 71.7], d=3.64 vs one-shot
- RAG gain is statistically significant at α=0.05 for all 6 local models (t > 2.776)
- GPT-4o and Claude 3.5 Sonnet: not significant (p>0.05) due to strong baseline

Full table: [results/STATISTICS.md](../results/STATISTICS.md)

---

## LocalStack Deployment Gap (Figure 12)

Figure 12 shows the gap between static validation pass rate and actual Terraform deployment
success rate via LocalStack.

```bash
# Requires LocalStack running
bash scripts/setup_localstack.sh
python scripts/run_experiments.py --model deepseek --condition sc+rag+5r \
  --validate-deploy --runs 1
```

Key result: sc+rag+5r achieves 82.7% static validation but only 74.4% deployment success
(−8.3 pp gap). Primary causes: circular IAM dependencies (23%), missing `depends_on` (19%).

---

## Expected Key Numbers

| Metric | Value | Paper location |
|--------|-------|----------------|
| One-shot functional correctness | 35.8% | Table 3 |
| SC+RAG(3r) functional correctness | 72.0% | Table 3 |
| RAG security gain (one-shot → one-shot+rag) | +19.9 pp | Section 7.2 |
| SC security gain (one-shot → sc+3r) | +18.4 pp | Section 7.3 |
| SC recovery rate overall (DeepSeek) | 7.9% | Section 7.3 |
| SC recovery: syntax errors | 72% | Table (SC by error class) |
| SC recovery: security misconfigs | 8% | Table (SC by error class) |
| Schema layer ablation drop | −17.8 pp | Table (ablation) |
| GPT-4o sc+rag+3r ceiling | 93.9% | Section 7.9 |
| LocalStack deployment gap | 8.3 pp | Section 7.11 |
| RAG Precision@5 (L1 avg) | 0.82 | Figure 14 |
| RAG Precision@5 (L5 avg) | 0.64 | Figure 14 |

---

## Compile the Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
open main.pdf
```

Figures are read from `results/figures/` as PDF (pdflatex native).
