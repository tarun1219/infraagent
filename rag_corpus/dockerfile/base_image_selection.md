# Dockerfile Base Image Selection

## Overview
The base image determines the attack surface, image size, and available tooling. Prefer minimal, well-maintained images. Always pin to a specific version or SHA256 digest.

## Image Hierarchy (Smallest to Largest Attack Surface)

| Image | Size | Shell | Package Manager | Use Case |
|-------|------|-------|-----------------|----------|
| `scratch` | 0 MB | No | No | Statically compiled binaries (Go, Rust) |
| `distroless/static` | ~2 MB | No | No | Statically compiled binaries |
| `distroless/base` | ~20 MB | No | No | Dynamically linked binaries |
| `alpine:3.x` | ~7 MB | sh | apk | General purpose, small footprint |
| `debian:12-slim` | ~75 MB | bash | apt | When glibc or system libs are needed |
| `ubuntu:22.04` | ~77 MB | bash | apt | Full system compatibility |

## Distroless (Recommended for Production)

```dockerfile
# Go static binary
FROM gcr.io/distroless/static-debian12:nonroot

# Python application
FROM gcr.io/distroless/python3-debian12:nonroot

# Java application
FROM gcr.io/distroless/java21-debian12:nonroot

# Node.js application
FROM gcr.io/distroless/nodejs20-debian12:nonroot
```

Distroless images:
- Contain no shell, no package manager, no coreutils
- `nonroot` variant runs as UID 65532 by default
- Significantly reduce CVE surface area
- Pass CIS benchmark checks out of the box

## Alpine Images

```dockerfile
FROM alpine:3.19

# Install only what you need, clean up in the same layer
RUN apk add --no-cache \
    ca-certificates \
    curl \
    && rm -rf /var/cache/apk/*
```

Alpine uses musl libc instead of glibc. Some software requires glibc compatibility shims.

## Official Language Runtime Images

```dockerfile
# Prefer -slim or -alpine variants for smaller footprint
FROM python:3.12-slim        # Good
FROM python:3.12             # Too large (includes build tools)
FROM python:3.12-alpine      # Smallest, but musl libc may cause issues

FROM node:20-alpine          # Good
FROM node:20-slim            # Good alternative (uses debian, glibc)

FROM eclipse-temurin:21-jre-jammy    # JRE only (not JDK) for runtime
```

## Pinning to SHA256 Digest (Maximum Reproducibility)

```dockerfile
# Pin by digest — immune to tag mutation attacks
FROM python:3.12-slim@sha256:4b7432c062f6d669f5babc2bed9b6a66e1de93f2ef2d67e8e2d4b2b9b44c9f3a

# Or pin by immutable tag (for well-maintained images)
FROM alpine:3.19.1    # Patch version pinning is safer than just 3.19
```

## Renovate / Dependabot for Automated Updates

Add a `renovate.json` to auto-update base images:
```json
{
  "extends": ["config:base"],
  "dockerfile": {
    "enabled": true
  }
}
```

## Image Scanning Before Shipping

```bash
# Trivy scan (fast, open source)
trivy image --exit-code 1 --severity HIGH,CRITICAL my-image:1.0.0

# Grype scan
grype my-image:1.0.0 --fail-on high

# Docker Scout
docker scout cves my-image:1.0.0
```

## Common Mistakes
- `FROM ubuntu:latest` — `latest` is mutable; the image can change silently between builds
- `FROM python:3` — implicit `latest`; pin to at least major.minor (e.g., `python:3.12-slim`)
- Using `FROM` with an untagged image — treated as `:latest` by Docker
- Installing development tools (`gcc`, `git`, `curl`) in the final stage — use multi-stage builds and leave these in the builder stage only
- Using outdated base images — scan regularly and update; many CVEs come from base OS packages
