# Kubernetes RBAC Patterns

## Overview
Role-Based Access Control (RBAC) controls who (subjects: users, groups, service accounts) can perform which actions (verbs) on which resources. Always follow least-privilege.

## Role vs ClusterRole
- `Role` — namespaced; grants permissions within one namespace
- `ClusterRole` — cluster-wide; grants permissions across all namespaces or on non-namespaced resources (nodes, PVs)
- `RoleBinding` — binds a Role or ClusterRole to subjects within a namespace
- `ClusterRoleBinding` — binds a ClusterRole to subjects across the entire cluster

## Application Service Account (Least Privilege)

```yaml
# 1. Create a dedicated ServiceAccount
apiVersion: v1
kind: ServiceAccount
metadata:
  name: my-app-sa
  namespace: production
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/my-app-role  # IRSA

---
# 2. Create a Role with only what the app needs
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: my-app-role
  namespace: production
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["my-app-secret"]   # Restrict to named secret only
    verbs: ["get"]

---
# 3. Bind the Role to the ServiceAccount
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: my-app-rolebinding
  namespace: production
subjects:
  - kind: ServiceAccount
    name: my-app-sa
    namespace: production
roleRef:
  kind: Role
  apiGroup: rbac.authorization.k8s.io
  name: my-app-role
```

## Read-Only ClusterRole (for monitoring tools)

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: read-only-cluster
rules:
  - apiGroups: [""]
    resources: ["pods", "services", "endpoints", "nodes"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["apps"]
    resources: ["deployments", "replicasets", "statefulsets"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["metrics.k8s.io"]
    resources: ["pods", "nodes"]
    verbs: ["get", "list"]
```

## Dangerous Patterns to Avoid

```yaml
# NEVER do this — grants full cluster admin to a service account
rules:
  - apiGroups: ["*"]
    resources: ["*"]
    verbs: ["*"]

# NEVER bind to system:masters group
subjects:
  - kind: Group
    name: system:masters    # This bypasses all RBAC checks
```

## Audit RBAC

```bash
# Check who can do what
kubectl auth can-i create pods --as=system:serviceaccount:production:my-app-sa -n production

# List all RoleBindings in a namespace
kubectl get rolebindings -n production -o wide

# List ClusterRoleBindings for a service account
kubectl get clusterrolebindings -o json | jq '.items[] | select(.subjects[]?.name=="my-app-sa")'
```

## Best Practices
- Each application gets its own ServiceAccount — never use `default`
- Disable token automounting if the app does not call the Kubernetes API: `automountServiceAccountToken: false`
- Use `resourceNames` to restrict access to specific named resources
- Audit with `kubectl auth can-i --list` and tools like `rakkess` or `rbac-tool`
