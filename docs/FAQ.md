# Frequently Asked Questions

**Q: Why is one-shot correctness only 35.8%?**

LLMs default to insecure patterns (root containers, deprecated APIs, no resource limits)
without explicit documentation context. Security score is only 16.8% one-shot because
security-relevant configurations are well-documented but not in the model's default
generation behavior.

---

**Q: Why is self-correction recovery only 7.9%?**

Tasks that fail completely after round 1 tend to have security misconfigurations where
the error message doesn't encode what the *correct* configuration should be. SC recovery
drops to 8% specifically for security failures (46% of all failures). See Table in
Section 7.3 for the full breakdown by error class.

---

**Q: Can I use GPT-4o or Claude instead of local models?**

Yes. Set environment variables and pass the model flag:
```bash
export OPENAI_API_KEY="sk-..."
python scripts/run_experiments.py --model gpt4o --condition one-shot --num_tasks 10
# ~$0.05 for 10 tasks with GPT-4o
```

---

**Q: Do I need a GPU?**

No. All open-source models run via Ollama CPU inference with 4-bit quantization (Q4_K_M).
Inference is slower (~30s/task on M2 Pro) but produces identical outputs. 16 GB RAM is
sufficient for models up to 16B parameters.

---

**Q: The kind cluster setup fails.**

Install kind first: `brew install kind` (macOS) or from
https://kind.sigs.k8s.io/docs/user/quick-start/

The pipeline degrades gracefully — Layer 2.5 dry-run validation is skipped when no
cluster is reachable (returns `dry_run_valid: null` instead of failing).

---

**Q: How do I add a new IaC language (e.g., Ansible)?**

1. Add tasks in `iachench/tasks/ansible_l[1-5].json`
2. Add a validator in `iachench/validators/ansible_validator.py`
3. Add the language to `LANGUAGES` in `scripts/simulate_results.py`
4. Update `infraagent/agent.py` to route Ansible tasks to the new validator

---

**Q: What's the difference between `experiment_results.json` and `results/raw/*.jsonl`?**

`experiment_results.json` is the **simulated** results used to generate all paper figures
(produced by `scripts/simulate_results.py`). The `results/raw/*.jsonl` files are intended
for **real** experimental runs produced by `scripts/run_experiments.py` — they're not
included in the repo due to size, but the format is documented in `results/README.md`.

---

**Q: How was IaCBench validated?**

- **Inter-annotator agreement**: 92% (46/50 tasks, 2 authors independently categorized)
- **Difficulty calibration**: 3 external infrastructure engineers estimated "time to write
  by hand": L1=5min, L2=15min, L3=30min, L4=60min, L5=120min
- **Spearman correlation** between median time and one-shot failure rate: ρ=0.94
