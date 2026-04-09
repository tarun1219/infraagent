# Frequently Asked Questions

---

**Q: Can I use GPT-4o or Claude instead of local models?**

Yes. Set the appropriate API key and pass `--model gpt4o` or `--model claude`:

```bash
export OPENAI_API_KEY="sk-..."
python scripts/run_experiments.py --model gpt4o --condition sc+rag+3r --runs 1

export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/run_experiments.py --model claude --condition sc+rag+3r --runs 1
```

**Cost estimates** for 300 tasks × 1 run:
- GPT-4o: ~$5 (sc+rag+3r with context, avg ~5K tokens/task)
- Claude 3.5 Sonnet: ~$6

Both commercial models are supported as drop-in replacements. The paper reports their results
as a performance ceiling (GPT-4o sc+rag+3r: 93.9% functional correctness).

---

**Q: How much GPU or CPU is needed?**

No GPU is required. All open-source models run via Ollama with 4-bit quantization (Q4_K_M):

| Model | RAM needed | M2 Pro speed |
|-------|-----------|-------------|
| Phi-3 3.8B | 4 GB | ~8 s/task |
| Mistral 7B | 6 GB | ~15 s/task |
| CodeLlama 13B | 10 GB | ~22 s/task |
| DeepSeek Coder v2 16B | 12 GB | ~30 s/task |
| Qwen 2.5 Coder 32B | 24 GB | ~55 s/task |
| Llama 3.1 70B | 48 GB | ~120 s/task |

A 16 GB MacBook M2 Pro handles all models up to 16B. For 70B you need 48+ GB RAM.

If you have a GPU, Ollama uses it automatically — inference is 5–10× faster.

---

**Q: How long does a full paper reproduction take?**

For all 8 models × 5 conditions × 300 tasks × 5 runs on a single M2 Pro (Ollama CPU):

| Scope | Estimated time |
|-------|---------------|
| DeepSeek only, 1 run, all conditions | ~12 hours |
| DeepSeek only, 5 runs, all conditions | ~62 hours |
| All 6 local models, 1 run, all conditions | ~104 hours |
| All 8 models, 5 runs, all conditions | ~520 hours |

**Practical approach:** Run models in parallel across machines (see [PAPER.md](PAPER.md)).
With 8 machines in parallel you can reproduce the full paper in ~65 hours.

Figures and statistics can be regenerated instantly from pre-computed results:
```bash
python scripts/simulate_results.py && python scripts/plot_figures.py
```

---

**Q: Can I add Ansible, Helm, or CloudFormation?**

Yes. The architecture is designed to be extensible. See [CONTRIBUTING.md](../CONTRIBUTING.md)
for the full guide. The short version:

1. Add tasks: `iachench/tasks/ansible/l[1-5]/*.json`
2. Add a validator: `iachench/validators/ansible_validator.py`
   - Implement `validate(content: str) -> dict` returning `{syntax_valid, schema_valid, security_valid, errors}`
3. Register the validator in `iachench/validators/__init__.py`
4. Add RAG corpus docs: `rag_corpus/ansible/*.md`
5. Add the language to `LANGUAGES` in `scripts/simulate_results.py`

The Task Planner uses keyword heuristics — add Ansible keywords (e.g., `hosts:`, `tasks:`,
`playbook`) to the routing table in `infraagent/agent.py`.

---

**Q: What if my RAG retrieval quality is poor?**

The default embedding model (`all-MiniLM-L6-v2`) is fast and lightweight. If you observe
low P@5 scores, upgrade to a stronger model:

```bash
# In infraagent/agent.py, change the embedding model:
# SentenceTransformer("all-MiniLM-L6-v2")   →   SentenceTransformer("BAAI/bge-large-en-v1.5")
```

From the paper's ablation (Figure 14): `bge-large-en-v1.5` improves P@5 by **+2.3 pp**
over `all-MiniLM-L6-v2`, at the cost of ~6× larger model (~1.3 GB vs ~90 MB) and ~3×
slower encoding.

You can also expand the RAG corpus by adding more documentation to `rag_corpus/`.
The corpus is re-indexed automatically when files change.

---

**Q: Why is my `kubectl dry-run=server` gap larger than the paper reports?**

The paper measures a gap of 6.2–12.4 pp between `kubeconform` (static schema) and
`kubectl dry-run=server` (live cluster) validation, depending on difficulty level.

Your gap may be larger if:

1. **Kubernetes version mismatch** — the paper used K8s 1.29. Newer clusters have additional
   admission webhooks and stricter defaults. Pin the kind cluster version:
   ```bash
   # In scripts/setup_kind_cluster.sh, K8S_VERSION is set to v1.29.2
   bash scripts/setup_kind_cluster.sh
   ```

2. **CRDs installed in your cluster** — Custom Resource Definitions (e.g., from cert-manager,
   Istio, Prometheus Operator) add validation that kubeconform doesn't know about.
   The paper's kind cluster is vanilla with no CRDs installed.

3. **Namespace-scoped admission controllers** — PodSecurity admission, OPA Gatekeeper,
   or Kyverno policies installed in your cluster will reject manifests that kubeconform accepts.
   The paper cluster has no admission controllers beyond the default PSA in `restricted` mode.

To reproduce the paper's exact dry-run environment:
```bash
kind delete cluster --name infraagent-eval 2>/dev/null || true
bash scripts/setup_kind_cluster.sh   # creates vanilla K8s 1.29 cluster
```

Issues or unexplained gaps? Open an issue at https://github.com/tarun1219/infraagent/issues
with your `kubectl version` and `kind version` output.

---

**Q: Why is one-shot correctness only 35.8%?**

LLMs default to insecure and outdated patterns without documentation context:
- Root containers (`runAsUser: 0`) instead of non-root
- Deprecated API versions (`extensions/v1beta1`, `autoscaling/v2beta2`)
- No resource limits, no health probes
- Wildcard IAM actions (`Action: "*"`)

Security pass rate is even lower (16.8% one-shot) because security-correct configuration
requires knowledge that is well-documented but not well-represented in training data
distributions for code generation.

---

**Q: Why is self-correction recovery rate only 7.9%?**

The 7.9% figure is the overall SC recovery rate — the fraction of initially-failing tasks
that pass after up to 3 rounds of self-correction. It breaks down by error class:

| Error class | Recovery | Reason |
|-------------|---------|--------|
| Syntax | 72% | Exact line/token in error message |
| Schema | 55% | Named field + API version in error |
| Security | 8% | Error names violation, not fix |
| Cross-resource | 2% | Fix requires global state awareness |

Security misconfigs (46% of all failures) have near-zero SC recovery because the error
message says *what is wrong* but not *what the correct configuration should be*. That
knowledge lives in documentation — which RAG retrieves. This is the root cause of the
prevention-over-repair asymmetry.

---

**Q: What is the difference between `experiment_results.json` and `results/raw/*.jsonl`?**

`experiment_results.json` is produced by `scripts/simulate_results.py` — it generates
all the aggregate metrics (means, CIs, ablation deltas) used to draw the paper figures.
It is committed to the repo so figures can be reproduced without running any models.

`results/raw/*.jsonl` files are per-run task-level outputs from `scripts/run_experiments.py`
(live inference). They are not committed due to size, but their format is documented in
[results/README.md](../results/README.md).

---

**Q: How was IaCBench validated?**

- **Inter-annotator agreement:** 92% (46/50 sampled tasks, 2 authors independently labeled type + difficulty)
- **Difficulty calibration:** 3 external infrastructure engineers estimated "time to write by hand": L1 ≈ 5 min, L2 ≈ 15 min, L3 ≈ 30 min, L4 ≈ 60 min, L5 ≈ 120 min
- **Spearman correlation** between median estimated time and one-shot failure rate: ρ = 0.94

---

**Q: How do I cite this work?**

```bibtex
@misc{infraagent2026,
  title  = {Prevention Over Repair: Quantifying the RAG-vs-Self-Correction
            Asymmetry in LLM-Generated Infrastructure-as-Code},
  author = {Anonymous},
  year   = {2026}
}
```
