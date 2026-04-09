# System Architecture

## Design Philosophy

InfraAgent is an **experimental platform**, not a novel algorithm. It composes existing,
well-understood components to enable rigorous measurement of a specific empirical question:

> Is RAG-based prevention more effective than self-correction-based repair for security
> correctness in LLM-generated Infrastructure-as-Code?

The architecture is deliberately minimal — every component has a single job, and the
pipeline is designed so each component can be swapped or ablated independently.

---

## Full Pipeline (ASCII)

```
 Natural Language Task Description
           │
           ▼
 ┌─────────────────────┐
 │     Task Planner    │
 │                     │
 │  • keyword heuristics → resource type (K8s/TF/Dockerfile)
 │  • difficulty classifier (L1–L5)
 │  • dependency graph  → ordered sub-task list
 └──────────┬──────────┘
            │ typed sub-tasks
            ▼
 ┌─────────────────────┐
 │     RAG Module      │  (only when use_rag=True)
 │                     │
 │  • query: task desc + error messages (in SC rounds)
 │  • ChromaDB vector store
 │  • all-MiniLM-L6-v2 embeddings (384-dim)
 │  • top-5 chunk retrieval (cosine similarity)
 │  • corpus: K8s PSP docs, TF IAM guides, Dockerfile best practices
 └──────────┬──────────┘
            │ retrieved context (≤2 048 tokens)
            ▼
 ┌─────────────────────┐
 │    Code Generator   │
 │                     │
 │  • Ollama (local): deepseek-coder-v2, codellama, mistral, phi3 ...
 │  • API: GPT-4o, Claude 3.5 Sonnet
 │  • 4-bit quantization (Q4_K_M GGUF) for local models
 │  • structured prompt: system + task + RAG context + (correction feedback)
 │  • temperature=0 for reproducibility
 └──────────┬──────────┘
            │ generated IaC (YAML / HCL / Dockerfile)
            ▼
 ┌────────────────────────────────────────────────┐
 │             Multi-Layer Validator              │
 │                                                │
 │  L1  Syntax       yamllint / hadolint / hclfmt │
 │  L2  Schema       kubeconform / terraform plan  │
 │  L2.5 Dry-run     kubectl dry-run=server        │  ← optional (kind cluster)
 │  L3  Security     Checkov + Trivy               │
 │  L4  Best Prac    OPA/Conftest custom policies  │
 └──────────┬─────────────────┬───────────────────┘
            │                 │
          PASS              FAIL
            │                 │
            ▼                 ▼
        Return          ┌─────────────────┐
        Output          │ Self-Correction  │  (only if rounds > 0)
                        │                 │
                        │ • structured error report
                        │ • fresh RAG query (error + task)
                        │ • feed back to Code Generator
                        │ • repeat up to N rounds
                        └─────────────────┘
```

---

## Component Details

### Task Planner

The Task Planner maps a natural language description to a typed, ordered execution plan.

**Keyword heuristics:**
- `Deployment`, `Service`, `ConfigMap`, `Ingress`, `HPA`, `NetworkPolicy` → Kubernetes
- `resource "aws_`, `provider`, `terraform` → Terraform
- `FROM`, `RUN`, `COPY`, `ENTRYPOINT` → Dockerfile

**Dependency graph:**
Cross-resource tasks (e.g., Deployment + HPA + Service) are parsed into a DAG.
Sub-tasks are executed in topological order so later resources can reference earlier ones.

**Difficulty classifier:**
Assigns L1–L5 based on:
- Number of resource types referenced
- Presence of cross-resource dependencies
- Security context requirements
- Deprecated API awareness needed

---

### RAG Module

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~90 MB)

**Vector store:** ChromaDB (local, persistent at `.chroma/`)

**Corpus (included in `rag_corpus/`):**

| File | Content |
|------|---------|
| `kubernetes/pod_security_standards.md` | PSP → PSA migration, securityContext fields |
| `kubernetes/api_versions.md` | Deprecated API versions by K8s release |
| `terraform/iam_best_practices.md` | Least-privilege IAM, wildcard action risks |
| `dockerfile/security.md` | Non-root users, distroless, COPY vs ADD |

**Retrieval:**
- Query = task description + (in SC rounds) concatenated error messages
- Top-5 chunks retrieved by cosine similarity
- Injected into the generation prompt as a `## Relevant Documentation` section
- Max 2,048 tokens of retrieved context to stay within model context windows

**Embedding model ablation (Figure 14):**
Switching from `all-MiniLM-L6-v2` to `bge-large-en-v1.5` improves P@5 by +2.3 pp.

---

### Code Generator

**Local inference (Ollama):**
- All local models use Q4_K_M GGUF quantization (4-bit, ~4× memory reduction)
- `temperature=0`, `top_p=1` for reproducible outputs
- `num_ctx=8192` (8K context window for all models)
- Ollama REST API at `http://localhost:11434`

**Commercial APIs:**
- GPT-4o: `openai` SDK, `temperature=0`
- Claude 3.5 Sonnet: `anthropic` SDK, `temperature=0`

**Prompt structure:**

```
[SYSTEM]
You are an expert infrastructure engineer. Generate valid, production-ready
Infrastructure-as-Code. Output only the IaC, no explanations.

[USER]
## Task
{task_description}

## Requirements
{requirements_list}

## Relevant Documentation          ← injected by RAG (when use_rag=True)
{retrieved_chunks}

## Previous Errors                 ← injected in SC rounds 2+
{structured_error_report}

Generate the IaC now:
```

---

### Multi-Layer Validator

Each layer returns a structured result that feeds the self-correction prompt.

| Layer | Tool | What it catches | SC-recoverable? |
|-------|------|----------------|-----------------|
| L1 Syntax | yamllint / hadolint / hclfmt | Malformed YAML, unclosed brackets | Yes (72%) |
| L2 Schema | kubeconform / `terraform validate` | Wrong API versions, missing required fields | Yes (55%) |
| L2.5 Dry-run | `kubectl dry-run=server` | CRD mismatches, admission webhooks | Partial |
| L3 Security | Checkov + Trivy | Root containers, wildcard IAM, exposed secrets | No (8%) |
| L4 Best Practices | OPA/Conftest | Missing resource limits, no health probes | Rarely |

**Why L3 is hard for SC:**
Security error messages (e.g., `CKV_K8S_30: containers must not run as root`) name
the violation but do not provide the correct fix. RAG retrieves that fix from
documentation — self-correction alone cannot.

---

### Algorithm 1 — Full Evaluation Loop

```
Input:  task T, model M, use_rag R, max_rounds N
Output: result {functional, security, round, errors}

1. plan ← TaskPlanner(T)
2. for sub_task in plan.ordered_tasks:
3.   context ← RAGModule.retrieve(sub_task) if R else ""
4.   code    ← Generator.generate(sub_task, context)
5.   result  ← Validator.validate(code, sub_task.type)
6.   round   ← 0
7.   while not result.passed and round < N:
8.     feedback ← format_errors(result.errors)
9.     context  ← RAGModule.retrieve(sub_task, query=feedback) if R else ""
10.    code     ← Generator.correct(code, feedback, context)
11.    result   ← Validator.validate(code, sub_task.type)
12.    round    ← round + 1
13. return {functional: result.functional, security: result.security,
14.          round: round, errors: result.errors}
```

**Example walk-through (task k8s-003, sc+rag+3r):**

- Round 0: Model generates HPA with `autoscaling/v2beta2` (deprecated), missing `securityContext`
  - L2 fails: `autoscaling/v2beta2` removed in K8s 1.25
  - L3 fails: CKV_K8S_30 (root user), CKV_K8S_22 (writable root fs)
- Round 1: RAG retrieves `api_versions.md` (autoscaling/v2) + `pod_security_standards.md`
  - Model fixes API version; adds `runAsNonRoot: true`
  - L2 passes; L3 still fails: capabilities not dropped (CKV_K8S_36)
- Round 2: RAG retrieves specific `capabilities.drop: ["ALL"]` documentation
  - Model adds full `securityContext` block
  - All layers pass → `functional: true, security: true, round: 2`

---

## Key Files

| File | Purpose |
|------|---------|
| `infraagent/agent.py` | Main pipeline orchestration (Algorithm 1) |
| `infraagent/generator.py` | LLM inference — Ollama + API routing |
| `infraagent/validators.py` | Multi-layer validation coordinator |
| `iachench/benchmark.py` | Task loader + evaluation harness |
| `iachench/metrics.py` | Canonical `compute_metric()` — single source of truth |
| `iachench/validators/kubernetes_validator.py` | K8s: yamllint → kubeconform → dry-run → Checkov |
| `iachench/validators/terraform_validator.py` | TF: hclfmt → validate → Checkov → LocalStack |
| `iachench/validators/dockerfile_validator.py` | DF: hadolint → Trivy → OPA |
| `scripts/simulate_results.py` | Deterministic simulation for figure generation |
| `scripts/generate_figures.py` | All 14 paper figures (PDF + PNG) |
