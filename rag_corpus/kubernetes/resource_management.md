# Kubernetes Resource Management

## Overview
Kubernetes resource management ensures fair sharing of cluster capacity. The core primitives are `requests` (scheduling hint), `limits` (enforcement cap), `ResourceQuota` (namespace-level budget), and `LimitRange` (per-object defaults and bounds).

## Requests vs Limits

| Field | Purpose | Enforced By |
|-------|---------|-------------|
| `requests.cpu` | Scheduling — guaranteed CPU | Scheduler |
| `requests.memory` | Scheduling — guaranteed memory | Scheduler |
| `limits.cpu` | CPU throttling ceiling | cgroups (CFS) |
| `limits.memory` | OOM kill threshold | cgroups (OOM killer) |

**QoS Classes:**
- `Guaranteed` — requests == limits for all containers (highest priority, last to be evicted)
- `Burstable` — requests < limits (middle priority)
- `BestEffort` — no requests or limits (lowest priority, first evicted)

## Container Resource Spec

```yaml
resources:
  requests:
    cpu: "250m"       # 250 millicores = 0.25 vCPU
    memory: "256Mi"
  limits:
    cpu: "1000m"      # 1 vCPU
    memory: "1Gi"
```

## LimitRange (Namespace Defaults)

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: default-limits
  namespace: production
spec:
  limits:
    - type: Container
      default:              # Applied if container omits limits
        cpu: "500m"
        memory: "256Mi"
      defaultRequest:       # Applied if container omits requests
        cpu: "100m"
        memory: "128Mi"
      max:                  # Hard ceiling — API rejects anything above
        cpu: "4"
        memory: "8Gi"
      min:                  # Hard floor
        cpu: "50m"
        memory: "64Mi"
    - type: Pod
      max:
        cpu: "8"
        memory: "16Gi"
    - type: PersistentVolumeClaim
      max:
        storage: "50Gi"
```

## ResourceQuota (Namespace Budget)

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: production-quota
  namespace: production
spec:
  hard:
    requests.cpu: "10"
    requests.memory: "20Gi"
    limits.cpu: "40"
    limits.memory: "80Gi"
    pods: "50"
    services: "10"
    persistentvolumeclaims: "20"
    requests.storage: "500Gi"
    count/deployments.apps: "20"
```

## Sizing Guidelines

| Workload Type | CPU Request | Memory Request |
|---------------|-------------|----------------|
| Lightweight API | 50–100m | 64–128Mi |
| Standard web app | 100–250m | 128–512Mi |
| JVM application | 500m–1 | 512Mi–2Gi |
| ML inference | 1–4 | 2–8Gi |

## Common Mistakes
- Setting CPU limit = CPU request (Guaranteed QoS) — fine for critical apps, but wastes capacity for bursty workloads
- Setting memory limit much higher than request — memory is not compressible; OOM kills happen at the limit
- No LimitRange in a namespace — containers without resources have BestEffort QoS and are evicted first
- CPU limits causing throttling — monitor `container_cpu_cfs_throttled_seconds_total` in Prometheus
