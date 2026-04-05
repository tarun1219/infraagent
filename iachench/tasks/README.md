# IaCBench Task Format

## File Layout

| File | Language | Difficulty | Tasks |
|------|----------|------------|-------|
| `kubernetes_l1.json` | Kubernetes | L1 (Single resource) | 20 |
| `kubernetes_l2.json` | Kubernetes | L2 (Multi-resource) | 20 |
| `kubernetes_l3.json` | Kubernetes | L3 (Complex configs) | 20 |
| `kubernetes_l4.json` | Kubernetes | L4 (Security-constrained) | 20 |
| `kubernetes_l5.json` | Kubernetes | L5 (Multi-component system) | 20 |
| `terraform_l[1-5].json` | Terraform HCL | L1–L5 | 20 each |
| `dockerfile_l[1-5].json` | Dockerfile | L1–L5 | 20 each |

**Total: 300 tasks** (20 per language–difficulty cell, 3 languages × 5 levels)

## Task Object Schema

```json
{
  "id": "kubernetes_l1_00",
  "language": "kubernetes",
  "difficulty": 1,
  "prompt": "Create a Kubernetes ConfigMap named 'app-config' with keys: DATABASE_URL=postgres://localhost, LOG_LEVEL=info",
  "expected_resources": ["ConfigMap"],
  "validation_criteria": {
    "required_api_versions": ["v1"],
    "required_patterns": ["metadata.name: app-config"],
    "forbidden_patterns": [],
    "security_threshold": 0.5,
    "best_practice_threshold": 0.6
  },
  "expected_output": "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: app-config\ndata:\n  DATABASE_URL: postgres://localhost\n  LOG_LEVEL: info\n",
  "failure_modes": [
    "missing ConfigMap kind",
    "wrong metadata structure",
    "missing data section"
  ]
}
```

## Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique ID: `{language}_l{difficulty}_{index:02d}` |
| `language` | enum | `"kubernetes"`, `"terraform"`, `"dockerfile"` |
| `difficulty` | int 1–5 | Task difficulty level |
| `prompt` | string | Natural language infrastructure description |
| `expected_resources` | list | Resource types that must appear in output |
| `validation_criteria` | object | Threshold-based pass/fail rules |
| `expected_output` | string | Reference correct configuration (documentation only) |
| `failure_modes` | list | Common mistakes catalogued for this task |

## Difficulty Levels

| Level | Name | Description | Median manual time |
|-------|------|-------------|-------------------|
| L1 | Single-resource | One resource type, no security constraints | 5 min |
| L2 | Multi-resource | 2–3 interdependent resources | 15 min |
| L3 | Complex | Advanced configuration options, multi-version APIs | 30 min |
| L4 | Security-constrained | Explicit security requirements (PSS, IAM, non-root) | 60 min |
| L5 | Multi-component system | Full service stack with cross-resource dependencies | 120 min |

## Benchmark Validation

- **Inter-annotator agreement**: 92% (46/50 tasks, 2 authors independently categorized)
- **Difficulty calibration**: 3 external engineers estimated manual writing time
- **Spearman ρ = 0.94** between median time and one-shot failure rate
