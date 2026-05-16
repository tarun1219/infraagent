# Dockerfile Non-Root User Patterns

## Overview
Containers run as root (UID 0) by default. If the container is compromised, the attacker has root-level access within the container and potentially on the host. Always switch to a non-root user before the final CMD/ENTRYPOINT.

## Alpine-Based Image

```dockerfile
FROM alpine:3.19

# Create a system user and group (no home dir, no login shell)
RUN addgroup -S appgroup && \
    adduser -S -G appgroup -u 1001 appuser

WORKDIR /app

COPY --chown=appuser:appgroup . .

RUN chmod +x /app/server

# Switch to non-root user
USER appuser

EXPOSE 8080
CMD ["/app/server"]
```

## Debian/Ubuntu-Based Image

```dockerfile
FROM debian:12-slim

# Create system group and user
RUN groupadd --gid 1001 --system appgroup && \
    useradd \
      --uid 1001 \
      --gid appgroup \
      --system \
      --no-create-home \
      --shell /sbin/nologin \
      appuser

WORKDIR /app

COPY --chown=appuser:appgroup dist/ ./dist/
COPY --chown=appuser:appgroup config/ ./config/

USER 1001    # Prefer UID over name for Kubernetes runAsUser compatibility

EXPOSE 3000
CMD ["node", "dist/index.js"]
```

## Multi-Stage with Non-Root

```dockerfile
FROM golang:1.22-alpine AS builder
WORKDIR /build
COPY . .
RUN CGO_ENABLED=0 go build -o server ./cmd/server

FROM alpine:3.19
RUN addgroup -S app && adduser -S -G app app

WORKDIR /app
COPY --from=builder --chown=app:app /build/server .

USER app
EXPOSE 8080
ENTRYPOINT ["./server"]
```

## UID/GID Selection Guidelines

| UID Range | Type |
|-----------|------|
| 0 | root — never use in containers |
| 1–999 | System users (created by OS packages) |
| 1000 | First regular user (ubuntu, debian) |
| 1001–9999 | Good range for app users |
| 65532 | distroless `nonroot` user |

## Distroless (Built-in Non-Root)

```dockerfile
FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=builder /app/server /server

# USER nonroot is already set by the :nonroot image tag
EXPOSE 8080
ENTRYPOINT ["/server"]
```

## Write Permissions for Necessary Directories

```dockerfile
RUN adduser -S appuser && \
    mkdir -p /app/tmp /app/logs && \
    chown -R appuser:appuser /app/tmp /app/logs

USER appuser

# Now mount readOnlyRootFilesystem: true in Kubernetes and provide
# /app/tmp and /app/logs as emptyDir volumes
```

## Kubernetes Alignment

When using `runAsNonRoot: true` in Kubernetes, the container USER must be numeric or the kubelet cannot verify it is non-root. Prefer:

```dockerfile
USER 1001    # Numeric UID — Kubernetes can verify this is not 0
```

Over:
```dockerfile
USER appuser    # String name — Kubernetes must resolve to verify non-root
```

## Common Mistakes
- `USER root` anywhere in the final stage — explicitly grants root
- No `USER` instruction — defaults to root (UID 0)
- Creating the user but forgetting `chown` on files — app runs as non-root but cannot read its own files
- Using `USER 0` — equivalent to root
- Giving the app user `sudo` access — defeats the purpose of running as non-root
- Using `--privileged` in `docker run` or Kubernetes — grants all capabilities to root
