#!/usr/bin/env bash
# Sets up a local Kubernetes cluster using kind for server-side validation.
# Required for kubectl dry-run=server validation (L2.5 validation layer).
#
# Usage:
#   bash scripts/setup_kind_cluster.sh
#   bash scripts/setup_kind_cluster.sh --cluster-name infraagent-eval
set -euo pipefail

CLUSTER_NAME="${1:-infraagent-eval}"
KIND_VERSION="v0.22.0"
K8S_VERSION="v1.29.2"

echo "==> Setting up kind cluster: ${CLUSTER_NAME}"

# ── Check dependencies ──────────────────────────────────────────────────────
for cmd in docker kubectl; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd is required but not installed."
        exit 1
    fi
done

# ── Install kind if not present ─────────────────────────────────────────────
if ! command -v kind &>/dev/null; then
    echo "==> Installing kind ${KIND_VERSION}..."
    OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
    ARCH="$(uname -m)"
    [ "$ARCH" = "x86_64" ] && ARCH="amd64"
    [ "$ARCH" = "aarch64" ] && ARCH="arm64"
    curl -Lo /usr/local/bin/kind \
        "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-${OS}-${ARCH}"
    chmod +x /usr/local/bin/kind
    echo "==> kind installed at /usr/local/bin/kind"
fi

# ── Delete existing cluster if present ──────────────────────────────────────
if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    echo "==> Deleting existing cluster '${CLUSTER_NAME}'..."
    kind delete cluster --name "${CLUSTER_NAME}"
fi

# ── Create cluster ──────────────────────────────────────────────────────────
echo "==> Creating cluster '${CLUSTER_NAME}' with Kubernetes ${K8S_VERSION}..."
kind create cluster \
    --name "${CLUSTER_NAME}" \
    --image "kindest/node:${K8S_VERSION}" \
    --wait 120s

# ── Verify ──────────────────────────────────────────────────────────────────
echo "==> Verifying cluster..."
kubectl cluster-info --context "kind-${CLUSTER_NAME}"
kubectl get nodes

echo ""
echo "==> kind cluster '${CLUSTER_NAME}' is ready."
echo "    kubectl dry-run=server validation is now available."
echo ""
echo "    To use with InfraAgent:"
echo "      export KUBECONFIG=\$(kind get kubeconfig-path --name=${CLUSTER_NAME})"
echo "    Or add to iachench config:"
echo "      validators: {k8s: {dry_run_server: true, context: kind-${CLUSTER_NAME}}}"
