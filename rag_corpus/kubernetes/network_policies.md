# Kubernetes NetworkPolicy Patterns

## Overview
NetworkPolicy is a namespaced resource that controls which pods can communicate with each other and with external endpoints. Without any NetworkPolicy, all pods can communicate freely (allow-all default).

## Default Deny All (Start Here)

```yaml
# Apply to every namespace to establish a zero-trust baseline
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}          # Matches ALL pods in the namespace
  policyTypes:
    - Ingress
    - Egress
```

## Allow Ingress from Specific Namespace

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-from-frontend
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: frontend-ns
          podSelector:
            matchLabels:
              app: frontend
      ports:
        - protocol: TCP
          port: 8080
```

## Allow Egress to Database

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-egress-to-db
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
    - Egress
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
    # Allow DNS resolution
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53
```

## Allow Monitoring Ingress (Prometheus Scraping)

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-prometheus-scrape
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: monitoring
          podSelector:
            matchLabels:
              app: prometheus
      ports:
        - port: 9090
          protocol: TCP
```

## Common Mistakes
- Using `namespaceSelector` OR `podSelector` as separate `from` entries creates an OR condition — use them under the same `from` entry item for AND
- Forgetting to allow DNS egress — pods will fail DNS lookups when egress is restricted
- Not labeling namespaces — `namespaceSelector` requires labels; use `kubernetes.io/metadata.name` (auto-set in K8s 1.21+)
- NetworkPolicy requires a CNI that enforces them (Calico, Cilium, Weave) — not all CNIs do

## Testing Policies

```bash
# Test connectivity between pods
kubectl exec -n production deploy/backend -- curl -s http://postgres:5432 --max-time 3

# Use netshoot for debugging
kubectl run tmp-shell --rm -it --image=nicolaka/netshoot -- /bin/bash
```
