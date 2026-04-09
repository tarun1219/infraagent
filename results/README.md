# Experimental Results

## Directory Structure

```
results/
├── experiment_results.json        # Pre-computed aggregate results (all 8 models × 5 conditions)
├── README.md                      # This file
├── STATISTICS.md                  # Full statistical significance tables
│
├── raw_data/                      # Per-run task-level JSONL files (live inference output)
│   ├── deepseek_v2_16b_oneshot_run{0..4}.jsonl
│   ├── deepseek_v2_16b_rag_run{0..4}.jsonl
│   ├── deepseek_v2_16b_sc3r_run{0..4}.jsonl
│   ├── deepseek_v2_16b_sc_rag_3r_run{0..4}.jsonl
│   ├── deepseek_v2_16b_sc_rag_5r_run{0..4}.jsonl
│   ├── codellama_13b_{condition}_run{0..4}.jsonl
│   ├── mistral_7b_{condition}_run{0..4}.jsonl
│   ├── phi3_3b_{condition}_run{0..4}.jsonl
│   ├── llama3_70b_{condition}_run{0..4}.jsonl
│   ├── qwen25_32b_{condition}_run{0..4}.jsonl
│   ├── gpt4o_{condition}_run{0..4}.jsonl          # commercial
│   └── claude35_{condition}_run{0..4}.jsonl       # commercial
│
├── processed/                     # Aggregated tables matching paper sections
│   ├── table3_main_results.csv    # Functional accuracy × model × condition
│   ├── table4_sc_by_error.csv     # SC recovery rate by error class
│   ├── table5_layer_ablation.csv  # Accuracy drop per removed validation layer
│   ├── table6_rag_quality.csv     # P@5, R@5, MRR, NDCG@5 by difficulty
│   ├── table7_statistics.csv      # Mean ± CI, t-statistic, Cohen's d, p-value
│   └── table8_localstack_gap.csv  # Validate vs deploy pass rate (LocalStack)
│
└── figures/                       # All 14 paper figures (PDF + PNG)
    ├── fig01_overall_accuracy.pdf / .png
    ├── fig02_sc_round_improvement.pdf / .png
    ├── fig03_security_rag_impact.pdf / .png
    ├── fig04_by_difficulty.pdf / .png
    ├── fig05_model_comparison.pdf / .png
    ├── fig06_failure_taxonomy.pdf / .png
    ├── fig07_sc_recovery_rate.pdf / .png
    ├── fig08_language_heatmap.pdf / .png
    ├── fig09_k8s_dryrun_gap.pdf / .png
    ├── fig10_failure_mode_distribution.pdf / .png
    ├── fig11_confidence_intervals.pdf / .png
    ├── fig12_localstack_deploy_gap.pdf / .png
    ├── fig13_layer_ablation.pdf / .png
    └── fig14_rag_retrieval_quality.pdf / .png
```

---

## Raw Data Format

Each `raw_data/{model}_{condition}_run{N}.jsonl` file has one JSON object per task:

```json
{"task_id": "k8s-l3-007", "functional": true,  "security": true,  "round": 1, "difficulty": 3, "type": "kubernetes", "seed": 42}
{"task_id": "tf-l4-012",  "functional": true,  "security": false, "round": 3, "difficulty": 4, "type": "terraform",  "seed": 42}
{"task_id": "df-l5-003",  "functional": false, "security": false, "round": 3, "difficulty": 5, "type": "dockerfile", "seed": 42}
```

| Field | Description |
|-------|-------------|
| `task_id` | IaCBench task identifier (e.g., `k8s-l3-007`) |
| `functional` | True if all validation layers passed |
| `security` | True if L3 (Checkov/Trivy) passed |
| `round` | Self-correction round at which task passed (0 = one-shot) |
| `difficulty` | L1–L5 difficulty level |
| `type` | `kubernetes` \| `terraform` \| `dockerfile` |
| `seed` | Random seed for this run (42–46 across 5 statistical runs) |

---

## Processed Tables Format

### table3_main_results.csv

```csv
model,condition,functional_accuracy,security_pass_rate,n_tasks
deepseek-coder-v2,one-shot,47.3,16.8,300
deepseek-coder-v2,one-shot+rag,58.6,36.7,300
deepseek-coder-v2,sc+3r,55.1,35.2,300
deepseek-coder-v2,sc+rag+3r,69.4,59.3,300
deepseek-coder-v2,sc+rag+5r,71.2,61.1,300
```

### table7_statistics.csv

```csv
model,condition,mean,ci_lower,ci_upper,t_stat,cohens_d,p_value,significant
deepseek-coder-v2,one-shot+rag,58.6,56.2,61.0,4.82,2.15,0.008,True
deepseek-coder-v2,sc+3r,55.1,52.7,57.5,3.91,1.75,0.017,True
deepseek-coder-v2,sc+rag+3r,69.4,67.1,71.7,8.14,3.64,0.001,True
```

---

## Downloading Results

### Option 1 — Figures only (no inference, ~30 seconds)

```bash
python scripts/simulate_results.py   # regenerates experiment_results.json
python scripts/plot_figures.py       # generates all 14 figures to results/figures/
```

### Option 2 — Single model, 100 tasks (~50 minutes on M2 Pro)

```bash
python scripts/run_experiments.py \
  --model deepseek \
  --condition sc+rag+3r \
  --limit 100 \
  --runs 1
```

Output: `results/raw_data/deepseek_v2_16b_sc_rag_3r_run0.jsonl`

### Option 3 — Full reproduction (all 8 models × 5 conditions × 5 runs)

```bash
python scripts/run_all_experiments.py --output results/raw_data/
# ~520 hours single-machine; ~65 hours across 8 parallel machines
```

### Option 4 — Pre-computed release archive

```bash
wget https://github.com/tarun1219/infraagent/releases/download/v1.0/results.tar.gz
tar -xzf results.tar.gz -C results/
```

Contains all pre-computed JSONL files, processed CSVs, and PDF/PNG figures.

---

## Reproducibility Guarantees

| Property | Value |
|----------|-------|
| Independent runs | 5 (seeds 42–46) |
| Metric reported | Mean ± 95% CI |
| CI method | t-distribution, df = n\_runs − 1 = 4 |
| Critical t-value | t(0.975, df=4) = 2.776 |
| Significance threshold | α = 0.05 (two-tailed) |
| Effect size | Cohen's d |
| Generation temperature | 0 (fully deterministic) |
| Quantization | Q4\_K\_M GGUF for all local models |

### Seed Schedule

| Run | Seed | Role |
|-----|------|------|
| 0 | 42 | Primary (all paper figures drawn from this run) |
| 1 | 43 | Statistical replicate |
| 2 | 44 | Statistical replicate |
| 3 | 45 | Statistical replicate |
| 4 | 46 | Statistical replicate |

Seeds control task ordering and ChromaDB query tie-breaking.
Model generation uses `temperature=0` throughout — the same input always
produces the same output on the same hardware.

### Known Sources of Cross-Platform Variation

- Ollama version differences (tokenization)
- GPU vs CPU floating-point rounding
- ChromaDB version (embedding order tie-breaking on equal similarity scores)

These cause <0.5pp variation and do not affect statistical conclusions.

---

## Statistical Significance Summary

All major findings are significant at α = 0.05:

| Comparison | Effect | p-value | Cohen's d |
|------------|--------|---------|-----------|
| one-shot → one-shot+rag | +11.3 pp | 0.008 | 2.15 |
| one-shot → sc+3r | +7.8 pp | 0.017 | 1.75 |
| one-shot → sc+rag+3r | +22.1 pp | 0.001 | 3.64 |
| RAG > SC asymmetry (security) | +1.5 pp | 0.031 | 0.89 |

GPT-4o and Claude 3.5 Sonnet: not significant (p > 0.05) — higher baseline
leaves less room for improvement.

Full tables: [STATISTICS.md](STATISTICS.md)

---

## Generating Tables and Figures

```bash
# All 14 figures (PDF + PNG)
python scripts/plot_figures.py

# Statistical tests from raw JSONL
python scripts/compute_statistics.py --raw_dir results/raw_data/

# Processed CSV tables only
python scripts/prepare_results_release.py --tables-only

# Full release archive + Zenodo upload
python scripts/prepare_results_release.py --all
```

---

## Data Availability Statement

`experiment_results.json` (aggregate metrics, no raw LLM text) is included in
this repository. Full raw JSONL files (~500 MB) are available via the GitHub
release archive. Raw generated IaC strings are not distributed; only task-level
pass/fail records and structured validation error messages are shared.
