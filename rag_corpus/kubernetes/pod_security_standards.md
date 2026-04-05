# Kubernetes Pod Security Standards

## Restricted Profile (required for L4+ tasks)

The restricted profile requires containers to run with minimal privileges:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 3000
  fsGroup: 2000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
  seccompProfile:
    type: RuntimeDefault
```

## Resource Limits (always required)

```yaml
resources:
  limits:
    cpu: "500m"
    memory: "128Mi"
  requests:
    cpu: "250m"
    memory: "64Mi"
```

## Common Checkov Failures

| Check ID | Description | Fix |
|----------|-------------|-----|
| CKV_K8S_30 | Containers must not run as root | Add `runAsNonRoot: true` |
| CKV_K8S_11 | CPU limits must be set | Add `resources.limits.cpu` |
| CKV_K8S_22 | Read-only root filesystem | Add `readOnlyRootFilesystem: true` |
| CKV_K8S_28 | No privilege escalation | Add `allowPrivilegeEscalation: false` |
| CKV_K8S_36 | Drop ALL capabilities | Add `capabilities.drop: ["ALL"]` |

## Example: Secure Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secure-app
  namespace: production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: secure-app
  template:
    metadata:
      labels:
        app: secure-app
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
      containers:
      - name: app
        image: nginx:1.25.3
        ports:
        - containerPort: 8080
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: ["ALL"]
        resources:
          limits:
            cpu: "500m"
            memory: "128Mi"
          requests:
            cpu: "250m"
            memory: "64Mi"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
```
