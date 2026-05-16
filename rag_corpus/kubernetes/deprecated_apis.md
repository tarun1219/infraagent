# Kubernetes Deprecated and Removed API Versions

## Overview
Kubernetes removes deprecated API versions after several release cycles. Using a removed API will cause `kubectl apply` to fail with a 404 error. Always migrate before upgrading the cluster.

## Removed in Kubernetes 1.16

| Resource | Removed API | Replacement |
|----------|-------------|-------------|
| Deployment | `apps/v1beta1`, `apps/v1beta2`, `extensions/v1beta1` | `apps/v1` |
| DaemonSet | `apps/v1beta2`, `extensions/v1beta1` | `apps/v1` |
| StatefulSet | `apps/v1beta1`, `apps/v1beta2` | `apps/v1` |
| ReplicaSet | `apps/v1beta1`, `apps/v1beta2`, `extensions/v1beta1` | `apps/v1` |
| NetworkPolicy | `extensions/v1beta1` | `networking.k8s.io/v1` |
| PodSecurityPolicy | `extensions/v1beta1` | `policy/v1beta1` (also removed in 1.25) |

## Removed in Kubernetes 1.22

| Resource | Removed API | Replacement |
|----------|-------------|-------------|
| Ingress | `extensions/v1beta1`, `networking.k8s.io/v1beta1` | `networking.k8s.io/v1` |
| IngressClass | `networking.k8s.io/v1beta1` | `networking.k8s.io/v1` |
| CertificateSigningRequest | `certificates.k8s.io/v1beta1` | `certificates.k8s.io/v1` |
| ClusterRole / ClusterRoleBinding / Role / RoleBinding | `rbac.authorization.k8s.io/v1beta1` | `rbac.authorization.k8s.io/v1` |
| ValidatingWebhookConfiguration / MutatingWebhookConfiguration | `admissionregistration.k8s.io/v1beta1` | `admissionregistration.k8s.io/v1` |
| CustomResourceDefinition | `apiextensions.k8s.io/v1beta1` | `apiextensions.k8s.io/v1` |

## Removed in Kubernetes 1.25

| Resource | Removed API | Replacement |
|----------|-------------|-------------|
| PodSecurityPolicy | `policy/v1beta1` | Pod Security Admission (built-in) or Kyverno/OPA Gatekeeper |
| PodDisruptionBudget | `policy/v1beta1` | `policy/v1` |
| RuntimeClass | `node.k8s.io/v1beta1` | `node.k8s.io/v1` |
| HorizontalPodAutoscaler | `autoscaling/v2beta1`, `autoscaling/v2beta2` | `autoscaling/v2` |

## Removed in Kubernetes 1.26

| Resource | Removed API | Replacement |
|----------|-------------|-------------|
| FlowSchema / PriorityLevelConfiguration | `flowcontrol.apiserver.k8s.io/v1beta1` | `flowcontrol.apiserver.k8s.io/v1beta3` |

## Removed in Kubernetes 1.27

| Resource | Removed API | Replacement |
|----------|-------------|-------------|
| CSIStorageCapacity | `storage.k8s.io/v1beta1` | `storage.k8s.io/v1` |

## Ingress Migration Example

```yaml
# OLD (removed in 1.22)
apiVersion: networking.k8s.io/v1beta1
kind: Ingress
spec:
  rules:
    - http:
        paths:
          - path: /
            backend:
              serviceName: my-app   # Old format
              servicePort: 80

# NEW (use this)
apiVersion: networking.k8s.io/v1
kind: Ingress
spec:
  ingressClassName: nginx            # Required field in v1
  rules:
    - http:
        paths:
          - path: /
            pathType: Prefix         # Required field in v1
            backend:
              service:              # Nested service object
                name: my-app
                port:
                  number: 80
```

## Detection Commands

```bash
# Find deprecated API usage in a cluster
kubectl get --raw /metrics | grep apiserver_requested_deprecated_apis

# Use pluto to scan manifests and Helm charts
pluto detect-files -d ./manifests/
pluto detect-helm -o wide

# kubent — Kubernetes Non-compliant Terminology (deprecated API scanner)
kubent
```

## HPA Migration (v2beta2 → v2)

```yaml
# OLD
apiVersion: autoscaling/v2beta2

# NEW
apiVersion: autoscaling/v2
# The spec structure is identical; only the apiVersion changes
```

## Best Practices
- Run `pluto` or `kubent` in CI pipelines before every cluster upgrade
- Check Helm chart API versions: `helm template my-chart | pluto detect -`
- Subscribe to Kubernetes release notes for deprecation announcements
- Upgrade charts and manifests at least one minor version ahead of the planned cluster upgrade
