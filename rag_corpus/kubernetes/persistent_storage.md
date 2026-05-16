# Kubernetes Persistent Storage

## Overview
Kubernetes storage uses PersistentVolume (PV) — the actual storage — and PersistentVolumeClaim (PVC) — a request for storage. StorageClass enables dynamic provisioning.

## PersistentVolumeClaim

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-app-data
  namespace: production
spec:
  accessModes:
    - ReadWriteOnce       # Single node read/write
  storageClassName: gp3   # References a StorageClass
  resources:
    requests:
      storage: 20Gi
```

## Access Modes

| Mode | Abbreviation | Description |
|------|-------------|-------------|
| `ReadWriteOnce` | RWO | One node can mount read/write |
| `ReadOnlyMany` | ROX | Many nodes can mount read-only |
| `ReadWriteMany` | RWX | Many nodes can mount read/write |
| `ReadWriteOncePod` | RWOP | One pod (not just node) read/write (K8s 1.22+) |

## StorageClass (AWS EBS gp3)

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  encrypted: "true"
  kmsKeyId: arn:aws:kms:us-east-1:123456789012:key/mrk-abc123
volumeBindingMode: WaitForFirstConsumer  # Provisions in the AZ where the pod lands
reclaimPolicy: Retain                    # Don't delete volume when PVC is deleted
allowVolumeExpansion: true
```

## Pod Using a PVC

```yaml
spec:
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: my-app-data
  containers:
    - name: my-app
      volumeMounts:
        - name: data
          mountPath: /data
```

## StatefulSet volumeClaimTemplates

```yaml
volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: gp3
      resources:
        requests:
          storage: 100Gi
```
Each pod in the StatefulSet gets its own PVC: `data-my-db-0`, `data-my-db-1`, etc.

## Reclaim Policies

| Policy | Behavior when PVC is deleted |
|--------|------------------------------|
| `Retain` | PV kept, data preserved, must be manually reclaimed |
| `Delete` | PV and underlying storage deleted automatically |
| `Recycle` | (Deprecated) Basic scrub and reuse |

Use `Retain` for production databases. Use `Delete` for ephemeral test environments.

## Volume Expansion

```bash
# Edit PVC to increase storage (StorageClass must have allowVolumeExpansion: true)
kubectl patch pvc my-app-data -n production -p '{"spec":{"resources":{"requests":{"storage":"50Gi"}}}}'
```

## Common Mistakes
- Using `reclaimPolicy: Delete` for production databases — a `kubectl delete pvc` permanently destroys data
- Using `ReadWriteMany` with EBS — EBS only supports `ReadWriteOnce`; use EFS for shared storage
- Forgetting `WaitForFirstConsumer` — without it, the PV is provisioned in a random AZ and may not be accessible to the pod
- Not encrypting storage — always set `encrypted: "true"` in StorageClass parameters
