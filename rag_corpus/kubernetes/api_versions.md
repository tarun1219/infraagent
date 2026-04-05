# Kubernetes API Version Reference

## Current (Non-Deprecated) API Versions

| Resource | Current API | Deprecated (DO NOT USE) |
|----------|-------------|------------------------|
| Deployment | `apps/v1` | `extensions/v1beta1` |
| ReplicaSet | `apps/v1` | `extensions/v1beta1` |
| DaemonSet | `apps/v1` | `extensions/v1beta1` |
| StatefulSet | `apps/v1` | — |
| Ingress | `networking.k8s.io/v1` | `extensions/v1beta1`, `networking.k8s.io/v1beta1` |
| NetworkPolicy | `networking.k8s.io/v1` | — |
| HorizontalPodAutoscaler | `autoscaling/v2` | `autoscaling/v2beta2`, `autoscaling/v1` |
| CronJob | `batch/v1` | `batch/v1beta1` |
| PodDisruptionBudget | `policy/v1` | `policy/v1beta1` |
| ClusterRole/RoleBinding | `rbac.authorization.k8s.io/v1` | — |
| ConfigMap / Secret | `v1` | — |
| Service / ServiceAccount | `v1` | — |

## Removed in Kubernetes 1.25+

The following were removed and will cause API server rejection:
- `extensions/v1beta1` for Ingress, Deployment, DaemonSet, ReplicaSet
- `networking.k8s.io/v1beta1` for Ingress
- `autoscaling/v2beta2` for HPA
- `policy/v1beta1` for PodDisruptionBudget, PodSecurityPolicy

## HPA v2 Example (correct)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```
