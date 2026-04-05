# 5-Minute Quickstart

## Prerequisites

```bash
pip install -r requirements.txt
ollama pull deepseek-coder-v2:16b-lite-instruct-q4_K_M  # first time only, ~9 GB
```

## Run a Single Task

```bash
python scripts/evaluate_single_task.py \
  --language kubernetes \
  --difficulty 1 \
  --task_id 0 \
  --model deepseek \
  --condition sc+rag+3r
```

Expected output:
```json
{
  "task_id": "kubernetes_l1_00",
  "condition": "sc+rag+3r",
  "syntax_valid": true,
  "schema_valid": true,
  "security_score": 0.82,
  "functional_correct": true,
  "rounds_used": 2
}
```

## Run 10 Sample Tasks

```bash
python scripts/run_experiments.py \
  --condition sc+rag+3r \
  --model deepseek \
  --num_tasks 10 \
  --output results/quickstart.jsonl
```

## View Results

```python
import jsonlines, statistics
with jsonlines.open("results/quickstart.jsonl") as f:
    rows = list(f)
fc = [r.get("functional_correct", False) for r in rows]
print(f"Functional Correctness: {statistics.mean(fc)*100:.1f}%")
```

## Reproduce Paper Figures

```bash
python scripts/plot_figures.py --output paper_figures/
# Generates fig1–fig14 as PDF + PNG
```

## Next Steps

- Full paper reproduction: [docs/PAPER.md](PAPER.md)
- Benchmark task format: [iachench/tasks/README.md](../iachench/tasks/README.md)
- Architecture details: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- Common issues: [docs/FAQ.md](FAQ.md)
