# Dockerfile Layer Optimization

## Overview
Each `RUN`, `COPY`, and `ADD` instruction creates a new image layer. Optimizing layer order and content reduces build times (via caching), final image size, and the number of files available to an attacker.

## Layer Caching Rules
1. Instructions are cached until a preceding layer changes
2. `COPY` and `ADD` are cache-invalidated when file content changes
3. Order instructions from least-frequently-changed to most-frequently-changed

## Optimal Layer Order

```dockerfile
FROM node:20-alpine

WORKDIR /app

# 1. System dependencies (rarely change)
RUN apk add --no-cache dumb-init

# 2. Dependency manifests (change when packages are updated)
COPY package.json package-lock.json ./

# 3. Install dependencies (cached until manifests change)
RUN npm ci --omit=dev

# 4. Application source (changes most frequently)
COPY src/ ./src/
COPY config/ ./config/

RUN addgroup -S app && adduser -S app -G app
RUN chown -R app:app /app

USER app

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget -qO- http://localhost:3000/health || exit 1

EXPOSE 3000
CMD ["dumb-init", "node", "src/index.js"]
```

## Combining RUN Commands to Reduce Layers

```dockerfile
# BAD: Creates 4 layers, intermediate files persist in earlier layers
RUN apt-get update
RUN apt-get install -y curl wget
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get clean

# GOOD: Single layer, no leftover apt cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl \
      wget \
    && rm -rf /var/lib/apt/lists/*
```

## BuildKit Cache Mounts (Persistent Cache Across Builds)

```dockerfile
# syntax=docker/dockerfile:1

# Go modules cache
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download

# apt cache (no need to rm -rf after)
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends curl

# pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

## .dockerignore (Critical for Cache and Security)

```dockerignore
# Version control
.git
.gitignore

# Development files
node_modules
npm-debug.log
.npm

# Test artifacts
coverage/
.nyc_output
*.test.js

# Build artifacts (built inside Docker, not from host)
dist/
build/

# Environment files (NEVER copy secrets)
.env
.env.*
*.pem
*.key
*.crt
secrets/

# CI/CD
.github/
.gitlab-ci.yml
Jenkinsfile

# Documentation
*.md
docs/

# IDE
.vscode/
.idea/
```

## Image Size Optimization Techniques

```dockerfile
# Use specific package versions to prevent cache busting
RUN apk add --no-cache curl=8.5.0-r0

# Remove unnecessary files after install
RUN pip install --no-cache-dir -r requirements.txt

# Use --no-install-recommends on apt to avoid pulling optional packages
RUN apt-get install -y --no-install-recommends python3

# Strip debug symbols from binaries
RUN go build -ldflags="-w -s" -o server .
```

## Analyzing Image Layers

```bash
# Show layer sizes
docker history my-image:1.0.0

# Detailed dive into layers
dive my-image:1.0.0

# Check final image size
docker images my-image:1.0.0
```

## Common Mistakes
- `COPY . .` as the first instruction after `FROM` — invalidates the entire cache on every source change
- Separate `RUN apt-get update` and `RUN apt-get install` — the update may be cached and stale
- Not having a `.dockerignore` — `COPY . .` copies `node_modules`, `.git`, `.env` into the image
- Downloading and extracting archives without cleaning up the archive: `ADD https://example.com/file.tar.gz /tmp/` adds the archive and extract separately
