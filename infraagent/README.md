# InfraAgent Module Structure

## Directory Layout

```
infraagent/
├── __init__.py          # Package exports and version
├── agent.py             # InfraAgent orchestrator (Algorithm 1)
├── generator.py         # LLM inference + prompt construction
├── validators.py        # Multi-layer validation pipeline
└── prompts/
    ├── system_prompt.txt      # System role for all LLM calls
    ├── generation_prompt.txt  # One-shot / RAG-augmented generation
    └── correction_prompt.txt  # Self-correction with error feedback
```

---

## Core Modules

### `infraagent/agent.py`

**`InfraAgent`** — top-level orchestrator implementing Algorithm 1 from the paper.

```python
class InfraAgent:
    def __init__(
        model: ModelID,
        max_rounds: int = 3,
        use_rag: bool = True,
        use_stub: bool = False,
        verbose: bool = False,
    ): ...

    def run(intent: str, task_id: str | None = None) -> AgentResult: ...
```

Pipeline executed by `run()`:

1. **Task Planner** (inline) — detects IaC language and decomposes multi-resource intents
2. **RAG retrieval** — queries ChromaDB for top-5 documentation chunks matching the intent
3. **LLM generation** — builds structured prompt (intent + RAG context) and calls the model
4. **Multi-layer validation** — runs L1→L4 validators, collects `ValidationReport`
5. **Self-correction loop** — on failure, reformulates RAG query from error messages and repeats up to `max_rounds` times

Supporting dataclasses:

| Class | Purpose |
|-------|---------|
| `RoundRecord` | Code + report snapshot for one SC round |
| `AgentResult` | Final code, report, all round records, timing, success flag |

---

### `infraagent/generator.py`

**`LLMCodeGenerator`** — wraps Ollama (local) and OpenAI/Anthropic (API) inference with structured prompting.

```python
class LLMCodeGenerator:
    def __init__(model: ModelID, temperature: float = 0, num_ctx: int = 8192): ...

    def generate(intent: str, rag_context: str = "", language: str = "") -> GenerationResult: ...
    def self_correct(intent: str, previous_code: str, errors: list[dict],
                     rag_context: str = "") -> GenerationResult: ...
```

**`ModelID`** enum — supported backends:

| Alias | Model | Backend |
|-------|-------|---------|
| `DEEPSEEK_CODER` | `deepseek-coder-v2:16b-lite-instruct-q4_K_M` | Ollama |
| `CODELLAMA` | `codellama:13b-instruct-q4_K_M` | Ollama |
| `MISTRAL` | `mistral:7b-instruct-q4_K_M` | Ollama |
| `PHI3` | `phi3:3.8b-instruct-q4_K_M` | Ollama |
| `GPT4O` | `gpt-4o` | OpenAI API |
| `CLAUDE` | `claude-3-5-sonnet-20241022` | Anthropic API |

Key design decisions:
- `temperature=0` for reproducibility across seeds
- `num_ctx=8192` accommodates task intent + RAG context + generated YAML
- Stub mode (`use_stub=True`) returns syntactically valid but semantically empty IaC — used in CI without Ollama

---

### `infraagent/validators.py`

**`MultiLayerValidator`** — runs four sequential validation layers and aggregates results into a `ValidationReport`.

```python
class MultiLayerValidator:
    def __init__(k8s_version: str = "1.29.0"): ...

    def validate(code: str, language: str) -> ValidationReport: ...
    def errors_to_feedback(report: ValidationReport) -> list[dict]: ...
```

**Validation layers:**

| Layer | ID | Tool(s) | Pass condition |
|-------|----|---------|----------------|
| L1 Syntax | `SYNTAX` | yamllint, HCL parse, hadolint | No parse errors |
| L2 Schema | `SCHEMA` | kubeconform (K8s), `terraform validate` (TF), hadolint (Dockerfile) | Schema-valid for target k8s version |
| L2.5 Dry-run | `DRY_RUN` | `kubectl apply --dry-run=server` | Server accepts manifest (requires kind) |
| L3 Security | `SECURITY` | Checkov, Trivy | `security_score ≥ 0.50` |
| L4 Best practices | `BEST_PRACTICE` | OPA/Conftest | `best_practice_score ≥ 0.60` |

L2.5 is optional — returns `dry_run_server_valid = None` when no kind cluster is available rather than failing.

Supporting types:

| Type | Fields |
|------|--------|
| `ValidationError` | `layer`, `rule_id`, `message`, `severity`, `line` |
| `ValidationReport` | `passed`, `syntax_valid`, `schema_valid`, `security_score`, `best_practice_score`, `dry_run_server_valid`, `errors` |
| `ValidationLayer` | Enum: `SYNTAX`, `SCHEMA`, `DRY_RUN`, `SECURITY`, `BEST_PRACTICE` |
| `Severity` | Enum: `ERROR`, `WARNING`, `INFO` |

---

### `infraagent/prompts/`

Three plain-text prompt templates loaded at runtime:

| File | Used by | Role |
|------|---------|------|
| `system_prompt.txt` | All calls | Sets model persona as an IaC security expert |
| `generation_prompt.txt` | `generator.generate()` | Slots in `{intent}`, `{language}`, `{rag_context}` |
| `correction_prompt.txt` | `generator.self_correct()` | Slots in `{intent}`, `{previous_code}`, `{errors}`, `{rag_context}` |

---

## RAG Module

The RAG pipeline lives in the `infraagent` package and is instantiated by `InfraAgent`. It uses:

- **Vector store**: ChromaDB (local persistent at `.chroma/`)
- **Embedding model**: `all-MiniLM-L6-v2` (384-dim, 23M params) via `sentence-transformers`
- **Corpus**: Markdown documentation chunks in `rag_corpus/` (Kubernetes, Terraform, Dockerfile)
- **Retrieval**: top-5 chunks, filtered by `language_filter` (collection per language)
- **Query reformulation**: on SC round > 0, the query is built from error rule IDs + messages rather than the raw intent

Key method used by `InfraAgent`:

```python
rag.build_context_string(query: str, language_filter: str = "") -> str
rag.retrieve(query: str, language_filter: str = "") -> list[dict]
```

---

## Task Planner

Keyword-based intent analysis built into `InfraAgent.run()`:

- **Language detection**: scans intent for Terraform keywords (`resource`, `aws_`, `provider`), Dockerfile keywords (`FROM`, `RUN`, `COPY`), defaulting to Kubernetes
- **Multi-resource decomposition**: splits on conjunctions (`and`, `,`, `+`) and resource-type keywords to identify subtasks
- **Dependency ordering**: enforces `Deployment → HPA`, `Service → Ingress`, `ConfigMap/Secret → Pod`
- **Difficulty estimation**: L1 (single resource, no security) through L5 (multi-resource, NetworkPolicy, cross-references)

---

## Metrics (`iachench/metrics.py`)

Standalone functions used by evaluation scripts and notebooks:

```python
compute_functional_correctness(validation_result: dict) -> bool
aggregate_metrics(results: list[dict]) -> dict  # mean functional, security, BP
compute_recovery_rate(corrected_results, failing_results=None) -> float | None
compute_metric(results: list[dict], by_difficulty: bool = False) -> dict
compute_pass_at_k(results_per_task: list[list[dict]], k: int) -> float
```

Statistical helpers used in `scripts/prepare_results_release.py`:
- Paired t-test (df = n−1, T_crit = 2.776 for n=5 seeds)
- Cohen's d effect size
- 95% CI half-width via t-distribution

---

## Example Usage

```python
from infraagent import InfraAgent
from infraagent.generator import ModelID

# Full pipeline: RAG-augmented generation + 3-round self-correction
agent = InfraAgent(
    model=ModelID.DEEPSEEK_CODER,
    max_rounds=3,
    use_rag=True,
)

result = agent.run(
    "Create a Kubernetes Deployment with PSS restrictions",
    task_id="k8s-l3-demo",
)

print(f"Passed:           {result.success}")
print(f"Rounds used:      {result.total_rounds_used}")
print(f"Security score:   {result.final_report.security_score:.2f}")
print(f"Best-prac score:  {result.final_report.best_practice_score:.2f}")
print(f"Duration:         {result.total_duration_s:.1f}s")
```

Access per-round details:

```python
for i, record in enumerate(result.rounds):
    print(f"Round {i}: passed={record.report.passed}  "
          f"errors={len(record.report.errors)}")
```

Export to JSON (all fields serializable):

```python
import json
print(json.dumps(result.to_dict(), indent=2))
```

One-shot generation without self-correction:

```python
agent = InfraAgent(model=ModelID.GPT4O, max_rounds=0, use_rag=True)
result = agent.run("Create an S3 bucket with KMS encryption")
```

Stub mode — no Ollama required (for CI / testing):

```python
agent = InfraAgent(model=ModelID.DEEPSEEK_CODER, use_stub=True)
result = agent.run("...")  # returns syntactically valid stub output
```

---

## Adding a New Model

1. Add an entry to `ModelID` in `generator.py`:
   ```python
   MY_MODEL = "my-model:7b-q4_K_M"
   ```
2. If it's an Ollama model, no further changes are needed — the generator calls `http://localhost:11434` by default.
3. For a new API provider, add a branch in `LLMCodeGenerator._call_llm()`.

## Adding a New Validation Rule

1. Add the rule check inside the appropriate layer method in `validators.py` (e.g., `_run_layer3_security()`).
2. Emit a `ValidationError` with a unique `rule_id` (prefix: `SEC_`, `BP_`, `SCH_`, `SYN_`).
3. Add a matching test in `tests/test_validator_layers.py`.
