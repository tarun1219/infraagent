# Kubernetes Deployment Best Practices

## Overview
A Deployment manages a ReplicaSet and provides declarative updates for Pods. Use Deployments for stateless workloads.

## Minimal Production-Ready Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  namespace: production
  labels:
    app: my-app
    version: "1.2.3"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0        # Zero-downtime rollout
  template:
    metadata:
      labels:
        app: my-app
        version: "1.2.3"
    spec:
      serviceAccountName: my-app-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
      containers:
        - name: my-app
          image: my-registry/my-app:1.2.3
          ports:
            - containerPort: 8080
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
          livenessProbe:
            httpGet:
              path: /healthz
              port: 8080
            initialDelaySeconds: 15
            periodSeconds: 20
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
          volumeMounts:
            - mountPath: /tmp
              name: tmp-dir
      volumes:
        - name: tmp-dir
          emptyDir: {}
```

## Selector Pattern Rules
- `spec.selector.matchLabels` must be a subset of `spec.template.metadata.labels`
- The selector is immutable after creation — plan your label scheme carefully
- Never use version labels in the selector (only in template labels) to avoid selector conflicts on updates

## Rolling Update Strategy
- `maxUnavailable: 0` ensures zero downtime (new pods start before old ones stop)
- `maxSurge: 1` allows one extra pod during rollout (requires extra capacity)
- For faster rollouts: increase maxSurge, for lower resource use: increase maxUnavailable

## Common Mistakes
- Using `image: myapp:latest` — always pin to a digest or semver tag
- Missing `namespace` field — resources land in `default`, which is shared and unprotected
- Setting `replicas: 1` — single replica creates a single point of failure
- Forgetting `selector.matchLabels` — without it the Deployment cannot find its pods
- Setting requests > limits — this is invalid and will be rejected by the API server
