# Kubernetes StatefulSet Patterns

## Overview
StatefulSets are used for stateful workloads that require stable network identities, stable persistent storage, and ordered deployment/scaling (e.g., databases, message queues, distributed caches).

## Key Differences from Deployments
- Pods get stable DNS names: `<pod-name>.<headless-svc>.<namespace>.svc.cluster.local`
- Pods are numbered sequentially: `my-db-0`, `my-db-1`, `my-db-2`
- Ordered deployment (0→1→2) and reverse-order termination (2→1→0) by default
- Each pod gets its own PersistentVolumeClaim via `volumeClaimTemplates`

## Production StatefulSet (PostgreSQL example)

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: data
  labels:
    app: postgres
spec:
  serviceName: postgres-headless    # Must reference a headless Service
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      serviceAccountName: postgres-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 999
        fsGroup: 999
      containers:
        - name: postgres
          image: postgres:15.3-alpine
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-secret
                  key: password
            - name: PGDATA
              value: /var/lib/postgresql/data/pgdata
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: false   # Postgres writes to data dir
            capabilities:
              drop: ["ALL"]
          livenessProbe:
            exec:
              command: ["pg_isready", "-U", "postgres"]
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            exec:
              command: ["pg_isready", "-U", "postgres"]
            initialDelaySeconds: 5
            periodSeconds: 5
          volumeMounts:
            - name: postgres-data
              mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
    - metadata:
        name: postgres-data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: gp3
        resources:
          requests:
            storage: 100Gi
```

## Required Headless Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres-headless
  namespace: data
spec:
  clusterIP: None
  selector:
    app: postgres
  ports:
    - port: 5432
```

## Update Strategy

```yaml
spec:
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: 0    # Set >0 to do canary: only pods with index >= partition update
```

## Common Mistakes
- Using `Deployment` for stateful workloads — pods share no storage identity
- Missing `serviceName` field — required and must reference an existing headless Service
- Using `ReadWriteMany` when storage class only supports `ReadWriteOnce`
- Forgetting `fsGroup` — without it, mounted volumes may not be writable by the container user
