# Dockerfile Build Secrets Management

## Overview
Secrets (API keys, npm tokens, SSH keys, database passwords) must never be baked into image layers. Even if removed in a later `RUN` step, the secret is visible in earlier layers via `docker history`. Use BuildKit secret mounts or build arguments carefully.

## The Problem: Secrets in Layers

```dockerfile
# DANGEROUS — token visible in image history even if later deleted
ARG NPM_TOKEN
RUN echo "//registry.npmjs.org/:_authToken=${NPM_TOKEN}" > ~/.npmrc && \
    npm install && \
    rm ~/.npmrc
```

```bash
docker history my-image --no-trunc | grep NPM_TOKEN    # Secret is exposed
```

## BuildKit Secret Mount (Correct Approach)

```dockerfile
# syntax=docker/dockerfile:1

FROM node:20-alpine AS builder
WORKDIR /app

COPY package.json package-lock.json ./

# Secret is mounted at runtime but NEVER written to any layer
RUN --mount=type=secret,id=npm_token \
    NPM_TOKEN=$(cat /run/secrets/npm_token) \
    npm set "//registry.npmjs.org/:_authToken=${NPM_TOKEN}" && \
    npm ci && \
    npm config delete "//registry.npmjs.org/:_authToken"
```

```bash
# Pass secret at build time — not stored in image
docker build --secret id=npm_token,src=~/.npmrc .

# Or from environment variable
echo "$NPM_TOKEN" | docker build --secret id=npm_token,src=/dev/stdin .
```

## SSH Key for Private Git Repositories

```dockerfile
# syntax=docker/dockerfile:1

FROM golang:1.22-alpine AS builder

RUN apk add --no-cache git openssh-client

RUN mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

# SSH key is mounted but not copied into the image
RUN --mount=type=ssh \
    git clone git@github.com:my-org/private-lib.git /tmp/private-lib

WORKDIR /app
COPY go.mod go.sum ./
RUN --mount=type=ssh \
    --mount=type=cache,target=/go/pkg/mod \
    go mod download

COPY . .
RUN go build -o server ./cmd/server
```

```bash
# Enable SSH agent forwarding to the build
docker build --ssh default=$SSH_AUTH_SOCK .
```

## ARG vs ENV (Understanding the Difference)

```dockerfile
# ARG — only available during build, NOT in the final image environment
ARG BUILD_VERSION
RUN echo "Building version ${BUILD_VERSION}"

# ENV — persisted in the image and available to all containers at runtime
ENV APP_PORT=8080
```

Safe use of ARG:
```dockerfile
ARG BUILD_DATE
ARG GIT_COMMIT
LABEL build-date="${BUILD_DATE}" git-commit="${GIT_COMMIT}"
# Labels are metadata, not secrets — safe to put non-sensitive build info here
```

Unsafe use of ARG (secret persists in image metadata):
```dockerfile
ARG DATABASE_PASSWORD          # Visible in: docker inspect, docker history
ENV DATABASE_PASSWORD=${ARG}   # Even worse — in runtime environment
```

## Runtime Secrets (Not Build Secrets)

For secrets needed at runtime (database passwords, API keys):

```dockerfile
# Reference environment variable — inject at runtime via Kubernetes Secret
ENV DATABASE_URL=""    # Empty default; inject via K8s Secret

CMD ["node", "src/index.js"]
```

```yaml
# Kubernetes — inject from Secret
env:
  - name: DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: app-secrets
        key: database_url
```

Or fetch from Secrets Manager at startup:

```python
import boto3, os
secret = boto3.client('secretsmanager').get_secret_value(
    SecretId=os.environ['SECRET_ARN']
)
```

## .dockerignore for Secret File Protection

```dockerignore
# Prevent accidental COPY of secret files
.env
.env.*
*.pem
*.key
*.p12
*.pfx
id_rsa
id_ed25519
~/.aws/credentials
secrets/
```

## Common Mistakes
- `COPY .env .` — copies secrets into the image layer permanently
- `ARG API_KEY` then `RUN curl -H "Authorization: ${API_KEY}"` — the full command (including key) is stored in image history
- Using `ENV` for secrets — environment variables are visible via `docker inspect` and in the container's `/proc` filesystem
- Using `ADD` to fetch from a URL that requires credentials — the URL (including any embedded credentials) is stored in the layer
- Not using BuildKit (`DOCKER_BUILDKIT=1`) — legacy builder has no secret mount support
