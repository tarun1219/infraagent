# 5-Minute Quickstart

## Prerequisites

```bash
# Clone and install
git clone https://github.com/tarun1219/infraagent.git
cd infraagent
pip install -r requirements.txt
pip install -e .

# Pull the primary model (first time only, ~9 GB)
ollama pull deepseek-coder-v2:16b-lite-instruct-q4_K_M
```

Full installation details: [docs/INSTALL.md](INSTALL.md)

---

## Step 1 — Reproduce All Figures (No Inference Required)

The fastest way to see results. Regenerates all 14 paper figures from pre-computed data.

```bash
python scripts/simulate_results.py   # writes results/experiment_results.json
python scripts/plot_figures.py       # writes results/figures/fig{01..14}.{pdf,png}
```

Open `results/figures/fig01.png` to see the main results bar chart.

---

## Step 2 — Evaluate a Single Task

```bash
python scripts/evaluate_single_task.py \
  --task examples/example_task_kubernetes.json \
  --model deepseek \
  --condition sc+rag+3r
```

Expected output:
```
Task loaded successfully:
{
  "id": "k8s-003",
  "type": "kubernetes",
  "difficulty": 3,
  "description": "Create a Kubernetes Deployment ... HorizontalPodAutoscaler ...",
  ...
}
Would evaluate with: model=deepseek-coder-v2:16b-lite-instruct-q4_K_M, rag=True, rounds=3
```

To run live inference, uncomment the `InfraAgent` block in `scripts/evaluate_single_task.py`.

---

## Step 3 — Run 10 Sample Tasks

```bash
python scripts/run_experiments.py \
  --condition sc+rag+3r \
  --model deepseek \
  --runs 1 \
  --limit 10
```

This takes ~5 minutes on Ollama CPU (~30 s/task). Output goes to
`results/raw/deepseek_sc_rag_3r_run0.jsonl`.

### Interpreting the Output

Each line in the JSONL file is one task result:

```json
{"task_id": "k8s-003", "functional": true,  "security": true,  "round": 1, "difficulty": 3, "type": "kubernetes"}
{"task_id": "tf-012",  "functional": true,  "security": false, "round": 3, "difficulty": 4, "type": "terraform"}
{"task_id": "df-007",  "functional": false, "security": false, "round": 3, "difficulty": 5, "type": "dockerfile"}
```

| Field | Description |
|-------|-------------|
| `functional` | True if the generated IaC passed all validation layers |
| `security` | True if security-specific checks (L3) passed |
| `round` | Which self-correction round produced a passing result (0 = one-shot) |
| `difficulty` | L1–L5 difficulty level of the task |

### Quick Summary

```python
import json
from pathlib import Path

results = [
    json.loads(line)
    for line in Path("results/raw/deepseek_sc_rag_3r_run0.jsonl").read_text().splitlines()
    if line.strip()
]
n = len(results)
print(f"Tasks:      {n}")
print(f"Functional: {sum(r['functional'] for r in results)/n*100:.1f}%")
print(f"Security:   {sum(r['security']   for r in results)/n*100:.1f}%")
```

---

## Single Model vs All 8 Models

### Single model (quickstart)

```bash
python scripts/run_experiments.py --model deepseek --condition sc+rag+3r --runs 1
```

### All 8 models (full paper replication)

```bash
# All conditions, 5 statistical runs each (~48 hours total)
python scripts/run_experiments.py --model all --condition all --runs 5
```

Run models in parallel across machines to reduce wall time:

```bash
# Machine 1 (needs 16 GB RAM)
python scripts/run_experiments.py --model deepseek  --condition all --runs 5
python scripts/run_experiments.py --model codellama --condition all --runs 5

# Machine 2
python scripts/run_experiments.py --model mistral --condition all --runs 5
python scripts/run_experiments.py --model phi3    --condition all --runs 5

# Machine 3 (needs 50+ GB RAM for 70B)
python scripts/run_experiments.py --model llama --condition all --runs 5
python scripts/run_experiments.py --model qwen  --condition all --runs 5

# API models (fast, billed per token)
python scripts/run_experiments.py --model gpt4o  --condition all --runs 5
python scripts/run_experiments.py --model claude --condition all --runs 5
```

After all runs finish:

```bash
python scripts/plot_figures.py
python scripts/compute_statistics.py
```

---

## Available Conditions

| Flag | Description |
|------|-------------|
| `one-shot` | Single LLM call, no RAG, no correction |
| `one-shot+rag` | Single LLM call with RAG context injected |
| `sc+3r` | Self-correction up to 3 rounds, no RAG |
| `sc+rag+3r` | Self-correction up to 3 rounds + RAG (primary condition) |
| `sc+rag+5r` | Self-correction up to 5 rounds + RAG |

---

## Next Steps

- Full paper reproduction: [docs/PAPER.md](PAPER.md)
- Benchmark task format: [iachench/tasks/README.md](../iachench/tasks/README.md)
- Architecture details: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- Common issues: [docs/FAQ.md](FAQ.md)
