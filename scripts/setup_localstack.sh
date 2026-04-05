#!/usr/bin/env bash
# Sets up LocalStack for zero-cost Terraform deployment validation.
# LocalStack mocks the AWS API locally, enabling real `terraform apply` runs.
#
# Usage:
#   bash scripts/setup_localstack.sh
#   bash scripts/setup_localstack.sh --port 4566
set -euo pipefail

LOCALSTACK_PORT="${1:-4566}"
LOCALSTACK_VERSION="3.2"

echo "==> Setting up LocalStack ${LOCALSTACK_VERSION} on port ${LOCALSTACK_PORT}"

# ── Check dependencies ──────────────────────────────────────────────────────
for cmd in docker terraform; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd is required but not installed."
        exit 1
    fi
done

# ── Install tflocal if not present ──────────────────────────────────────────
if ! command -v tflocal &>/dev/null; then
    echo "==> Installing tflocal (Terraform LocalStack wrapper)..."
    pip install terraform-local
fi

# ── Stop existing LocalStack container ──────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q "^localstack$"; then
    echo "==> Stopping existing LocalStack container..."
    docker stop localstack 2>/dev/null || true
    docker rm   localstack 2>/dev/null || true
fi

# ── Start LocalStack ─────────────────────────────────────────────────────────
echo "==> Starting LocalStack ${LOCALSTACK_VERSION}..."
docker run -d \
    --name localstack \
    -p "${LOCALSTACK_PORT}:4566" \
    -p "4510-4559:4510-4559" \
    -e LOCALSTACK_SERVICES="s3,iam,ec2,lambda,dynamodb,cloudformation" \
    -e DEBUG=0 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    "localstack/localstack:${LOCALSTACK_VERSION}"

# ── Wait for LocalStack to be ready ─────────────────────────────────────────
echo "==> Waiting for LocalStack to be ready..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${LOCALSTACK_PORT}/_localstack/health" &>/dev/null; then
        echo "==> LocalStack is ready (attempt ${i}/30)"
        break
    fi
    sleep 2
done

# ── Configure AWS CLI for LocalStack ────────────────────────────────────────
echo "==> Configuring AWS CLI for LocalStack..."
export AWS_DEFAULT_REGION=us-east-1
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test

# Verify
curl -s "http://localhost:${LOCALSTACK_PORT}/_localstack/health" | python3 -m json.tool

echo ""
echo "==> LocalStack is ready for Terraform deployment validation."
echo ""
echo "    To use with InfraAgent validators:"
echo "      export LOCALSTACK_ENDPOINT=http://localhost:${LOCALSTACK_PORT}"
echo "      export AWS_DEFAULT_REGION=us-east-1"
echo "      export AWS_ACCESS_KEY_ID=test"
echo "      export AWS_SECRET_ACCESS_KEY=test"
echo ""
echo "    Run Terraform against LocalStack:"
echo "      tflocal init && tflocal plan && tflocal apply -auto-approve"
