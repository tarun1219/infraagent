# Kubernetes Service Patterns

## Overview
Services provide stable network identities for dynamic sets of Pods. The selector maps to pod labels; traffic is load-balanced across matching pods.

## Service Types

| Type | Use Case |
|------|----------|
| ClusterIP | Internal cluster communication (default) |
| NodePort | Direct node access, testing, on-prem L4 LB |
| LoadBalancer | Cloud provider external L4 load balancer |
| ExternalName | CNAME alias to external DNS name |

## ClusterIP (Internal Service)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app
  namespace: production
  labels:
    app: my-app
spec:
  type: ClusterIP
  selector:
    app: my-app          # Must match pod template labels
  ports:
    - name: http
      port: 80           # Port clients use to reach the Service
      targetPort: 8080   # Port the container listens on
      protocol: TCP
```

## LoadBalancer (External Service)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app-lb
  namespace: production
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-internal: "true"
spec:
  type: LoadBalancer
  selector:
    app: my-app
  ports:
    - name: https
      port: 443
      targetPort: 8443
  loadBalancerSourceRanges:
    - "10.0.0.0/8"       # Restrict to internal CIDR only
```

## Headless Service (for StatefulSets)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-db-headless
  namespace: production
spec:
  clusterIP: None        # Headless — returns pod IPs directly via DNS
  selector:
    app: my-db
  ports:
    - port: 5432
      targetPort: 5432
```

## Multi-Port Service

```yaml
spec:
  ports:
    - name: http
      port: 80
      targetPort: 8080
    - name: metrics
      port: 9090
      targetPort: 9090
```
Named ports are required when exposing multiple ports on the same Service.

## Selector Best Practices
- The selector must reference at least one label key
- Selectors use AND logic — all key/value pairs must match
- Never use mutable labels (like `version`) in the Service selector if pods are updated in place
- Use `app` + optionally `component` for fine-grained selection

## Common Mistakes
- Missing `selector` on a ClusterIP Service — traffic goes nowhere (unless intentional headless)
- `port` vs `targetPort` confusion — `port` is the Service port, `targetPort` is the container port
- Using `NodePort` in production — exposes high-numbered ports on every node, prefer LoadBalancer or Ingress
