# IaCBench: 300 Curated IaC Tasks

IaCBench is the evaluation benchmark for the InfraAgent paper. It contains 300 manually
curated Infrastructure-as-Code generation tasks across three languages and five difficulty levels.

---

## Structure

```
iachench/
├── benchmark.py              # All 300 tasks as Python dataclasses (source of truth)
├── metrics.py                # Canonical scoring: compute_metric(), compute_pass_at_k()
├── validators/
│   ├── kubernetes_validator.py
│   ├── terraform_validator.py
│   └── dockerfile_validator.py
└── tasks/
    ├── kubernetes/
    │   ├── L1_single_resource/    (20 tasks: k8s-l1-001 … k8s-l1-020)
    │   ├── L2_api_awareness/      (20 tasks: k8s-l2-001 … k8s-l2-020)
    │   ├── L3_security_context/   (20 tasks: k8s-l3-001 … k8s-l3-020)
    │   ├── L4_policy_compliance/  (20 tasks: k8s-l4-001 … k8s-l4-020)
    │   └── L5_system_design/      (20 tasks: k8s-l5-001 … k8s-l5-020)
    ├── terraform/
    │   ├── L1_single_resource/    (20 tasks: tf-l1-001 … tf-l1-020)
    │   ├── L2_networking/         (20 tasks: tf-l2-001 … tf-l2-020)
    │   ├── L3_modules/            (20 tasks: tf-l3-001 … tf-l3-020)
    │   ├── L4_encryption/         (20 tasks: tf-l4-001 … tf-l4-020)
    │   └── L5_multi_account/      (20 tasks: tf-l5-001 … tf-l5-020)
    └── dockerfile/
        ├── L1_single_stage/       (20 tasks: df-l1-001 … df-l1-020)
        ├── L2_multi_stage/        (20 tasks: df-l2-001 … df-l2-020)
        ├── L3_non_root/           (20 tasks: df-l3-001 … df-l3-020)
        ├── L4_distroless/         (20 tasks: df-l4-001 … df-l4-020)
        └── L5_hardened/           (20 tasks: df-l5-001 … df-l5-020)
```

---

## Task Format

Each task is a `BenchmarkTask` dataclass defined in `benchmark.py`.
The equivalent JSON representation is:

```json
{
  "task_id": "k8s-l4-001",
  "language": "kubernetes",
  "difficulty": 4,
  "category": "PodSecurityAdmission",
  "prompt": "Create a Pod Security Standards-restricted Deployment with a NetworkPolicy that allows ingress only from pods labelled app=gateway on port 8080.",
  "expected_resources": ["Deployment", "NetworkPolicy"],
  "validation": {
    "syntax_valid": true,
    "schema_valid": true,
    "min_security_score": 0.5,
    "min_bp_score": 0.6,
    "required_resources": ["Deployment", "NetworkPolicy"],
    "required_api_versions": {
      "Deployment": "apps/v1",
      "NetworkPolicy": "networking.k8s.io/v1"
    },
    "required_patterns": [
      "runAsNonRoot: true",
      "readinessProbe",
      "policyTypes"
    ],
    "forbidden_patterns": [
      "extensions/v1beta1",
      "runAsUser: 0"
    ]
  },
  "common_failures": [
    "insecure_defaults",
    "label_mismatch",
    "deprecated_api"
  ]
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | str | Unique identifier: `{lang}-l{level}-{NNN}` |
| `language` | str | `"kubernetes"` \| `"terraform"` \| `"dockerfile"` |
| `difficulty` | int | 1–5 (see Difficulty Levels below) |
| `category` | str | Human-readable resource category |
| `prompt` | str | Natural language instruction given to the agent |
| `expected_resources` | list[str] | Resource types the solution must contain |
| `validation.required_patterns` | list[str] | Regex patterns that must appear in output |
| `validation.forbidden_patterns` | list[str] | Patterns that must NOT appear (e.g., deprecated APIs) |
| `validation.min_security_score` | float | Minimum Checkov/Trivy score (default 0.5) |
| `common_failures` | list[str] | Observed failure modes from one-shot evaluation |

---

## Difficulty Levels

| Level | Name | Validation Layers | Description | Approx. time to write |
|-------|------|------------------|-------------|----------------------|
| L1 | Single Resource | L1 syntax | One resource type, basic configuration | ~5 min |
| L2 | API Awareness | L1 + L2 schema | Correct API version selection, multi-field schemas | ~15 min |
| L3 | Security Context | L1 + L2 + L3 security | Non-root, capability dropping, cross-resource refs | ~30 min |
| L4 | Policy Compliance | L1 + L2 + L3 + L4 best practices | Admission controllers, encryption, RBAC | ~60 min |
| L5 | System Design | All layers + L2.5 dry-run | Multi-namespace, stateful, complex dependencies | ~120 min |

Difficulty calibration: external infrastructure engineers provided time estimates for
a random sample of 50 tasks. **Spearman ρ = 0.94** between median expert time and
one-shot failure rate. See `metadata.json` for raw calibration data.

---

## Dataset Statistics

```
Total tasks:      300
├── kubernetes:   100  (k8s-l1-001 … k8s-l5-020)
├── terraform:    100  (tf-l1-001  … tf-l5-020)
└── dockerfile:   100  (df-l1-001  … df-l5-020)

Per difficulty:
├── L1:  60 tasks  (20 per language)
├── L2:  60 tasks  (20 per language)
├── L3:  60 tasks  (20 per language)
├── L4:  60 tasks  (20 per language)
└── L5:  60 tasks  (20 per language)
```

### Validation Quality

| Metric | Value | Method |
|--------|-------|--------|
| Inter-annotator agreement | 92% | 2 authors, 50 sampled tasks (4 L3/L4 boundary disagreements) |
| Difficulty calibration ρ | 0.94 | Spearman, expert time vs. one-shot failure rate |
| Prompt uniqueness | 100% | No duplicate prompts |
| Coverage: L3+ Checkov checks | ≥50% of L3+ tasks | At least one CKV_ rule per security task |

### Task Sources

Tasks were constructed from:
- **Official documentation**: Kubernetes docs, Terraform AWS provider docs, Docker best practices
- **Certification scenarios**: CKA (Certified Kubernetes Administrator), CKS (Certified Kubernetes Security Specialist) exam domains
- **Production incident reports**: Common misconfigurations from public postmortems
- **Security benchmarks**: CIS Kubernetes, CIS Docker, AWS Security Best Practices

---

## Accessing Tasks Programmatically

```python
from iachench.benchmark import (
    ALL_TASKS,
    get_tasks_by_language,
    get_tasks_by_difficulty,
    get_task_by_id,
    summary,
)

# All 300 tasks
print(len(ALL_TASKS))   # 300

# Filter by language
k8s_tasks = get_tasks_by_language("kubernetes")   # 100 tasks

# Filter by difficulty
l3_tasks = get_tasks_by_difficulty(3)              # 60 tasks

# Single task by ID
task = get_task_by_id("k8s-l4-001")
print(task.prompt)
print(task.validation.required_patterns)

# Summary statistics
s = summary()
# {"total": 300, "by_language": {...}, "by_difficulty": {...}}
```

---

## Validation

Run the dataset validation script to verify completeness, structure, and calibration:

```bash
# Check everything
python scripts/validate_iacbench.py

# Individual checks
python scripts/validate_iacbench.py --check-completeness
python scripts/validate_iacbench.py --check-structure
python scripts/validate_iacbench.py --check-difficulty-spread
```

Expected output on a valid dataset:
```
[PASS] Completeness: 300 tasks (100 per language, 60 per level, 20 per language×level)
[PASS] Structure: all 300 tasks have required fields
[PASS] Difficulty spread: Spearman ρ = 0.94 ≥ 0.90
```

---

## metadata.json

`iachench/tasks/metadata.json` (generated by `validate_iacbench.py --generate-metadata`) contains:

```json
{
  "total_tasks": 300,
  "inter_annotator_agreement": {
    "n_sampled": 50,
    "n_agreed": 46,
    "agreement_rate": 0.92,
    "disagreements": "4 tasks at L3/L4 boundary (security threshold interpretation)"
  },
  "difficulty_calibration": {
    "expert_time_minutes": {"1": 5, "2": 15, "3": 30, "4": 60, "5": 120},
    "one_shot_failure_rate": {"1": 0.18, "2": 0.32, "3": 0.52, "4": 0.71, "5": 0.85},
    "spearman_rho": 0.94
  },
  "task_sources": ["kubernetes_docs", "cka_cks_domains", "incident_reports", "cis_benchmarks"],
  "generated_at": "2026-04-09"
}
```
