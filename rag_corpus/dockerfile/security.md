# Dockerfile Security Best Practices

## Non-Root User (required for L4+ tasks)

```dockerfile
FROM python:3.11-slim

# Install dependencies as root BEFORE switching user
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Copy application files
COPY --chown=appuser:appuser . .

# Switch to non-root for runtime
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8080/health || exit 1
CMD ["python", "app.py"]
```

## Common Checkov/Trivy Failures

| Check ID | Description | Fix |
|----------|-------------|-----|
| CKV_DOCKER_8 | Ensure containers do not run as root | Add `USER nonroot` directive |
| CKV_DOCKER_7 | Avoid using latest image tag | Use `nginx:1.25.3` not `nginx:latest` |
| CKV_DOCKER_28 | Add HEALTHCHECK | Add `HEALTHCHECK` instruction |
| CKV_DOCKER_2 | Use COPY not ADD | Replace `ADD` with `COPY` |
| DS002 | root user | Switch to non-root with USER directive |

## Multi-Stage Build (reduces attack surface)

```dockerfile
# Build stage
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY . .
RUN go build -o server .

# Runtime stage — minimal image
FROM scratch
COPY --from=builder /app/server /server
USER 65534:65534  # nobody
EXPOSE 8080
ENTRYPOINT ["/server"]
```
