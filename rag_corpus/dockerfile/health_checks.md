# Dockerfile HEALTHCHECK Patterns

## Overview
The `HEALTHCHECK` instruction tells Docker how to test whether the container is still working. Without it, Docker considers a container healthy as soon as it starts, even if the application has crashed or is deadlocked. Kubernetes uses its own probes, but HEALTHCHECK is still valuable for Docker Compose and standalone Docker deployments.

## HEALTHCHECK Syntax

```dockerfile
HEALTHCHECK [OPTIONS] CMD <command>

# Options:
# --interval=DURATION    Time between checks (default: 30s)
# --timeout=DURATION     Timeout for each check (default: 30s)
# --start-period=DURATION Grace period during startup (default: 0s)
# --retries=N            Failures before marking unhealthy (default: 3)
```

## HTTP API Health Check

```dockerfile
FROM node:20-alpine

WORKDIR /app
COPY --chown=node:node . .
RUN npm ci --omit=dev

USER node

HEALTHCHECK --interval=30s \
            --timeout=5s \
            --start-period=10s \
            --retries=3 \
  CMD wget -qO- http://localhost:3000/health || exit 1

EXPOSE 3000
CMD ["node", "src/index.js"]
```

Use `wget` (available in Alpine) or `curl` instead of bash HTTP clients. Prefer `wget` in Alpine since `curl` must be explicitly installed.

## Go/Distroless (No wget/curl)

```dockerfile
FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=builder /app/server /server
COPY --from=builder /app/healthcheck /healthcheck    # Compiled Go health binary

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD ["/healthcheck"]

USER nonroot
EXPOSE 8080
ENTRYPOINT ["/server"]
```

Since distroless has no shell or wget, compile a small health check binary:

```go
// cmd/healthcheck/main.go
package main

import (
    "net/http"
    "os"
)

func main() {
    resp, err := http.Get("http://localhost:8080/healthz")
    if err != nil || resp.StatusCode != http.StatusOK {
        os.Exit(1)
    }
}
```

## PostgreSQL Database

```dockerfile
FROM postgres:15-alpine

HEALTHCHECK --interval=10s \
            --timeout=5s \
            --start-period=30s \
            --retries=5 \
  CMD pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" || exit 1
```

## Redis

```dockerfile
FROM redis:7-alpine

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD redis-cli ping || exit 1
```

## Nginx

```dockerfile
FROM nginx:1.25-alpine

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://localhost/nginx-health || exit 1

COPY nginx.conf /etc/nginx/nginx.conf
```

## Python Flask/FastAPI

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN groupadd -r app && useradd -r -g app app
USER app

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Docker Compose Health Check (Dependency Ordering)

```yaml
services:
  app:
    image: my-app:1.0.0
    depends_on:
      db:
        condition: service_healthy    # Wait until DB is healthy

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: secret
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
```

## Common Mistakes
- `HEALTHCHECK NONE` — explicitly disables health checking; do not use this unless you have a very specific reason
- Health check that tests external dependencies (database, external APIs) — if the DB is down, all containers are marked unhealthy and Docker tries to restart them, worsening the outage
- `start-period` too short for slow-starting apps (JVM, Python with heavy imports) — the app is marked unhealthy before it finishes initializing
- Using `curl` without installing it in Alpine images — `wget` is pre-installed in Alpine
- Returning non-zero exit code for non-critical issues — only fail health checks on conditions that warrant a container restart
