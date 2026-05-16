# Dockerfile Multi-Stage Build Patterns

## Overview
Multi-stage builds separate the build environment from the runtime environment. The final image contains only the compiled artifacts and runtime dependencies — not build tools, source code, or intermediate files. This dramatically reduces image size and attack surface.

## Go Application (Builder + Distroless Runtime)

```dockerfile
# Stage 1: Build
FROM golang:1.22-alpine AS builder

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source and build
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-w -s" \
    -o /app/server \
    ./cmd/server

# Stage 2: Runtime (distroless — no shell, no package manager)
FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=builder /app/server /server

EXPOSE 8080

USER nonroot:nonroot

ENTRYPOINT ["/server"]
```

## Node.js Application (Builder + Slim Runtime)

```dockerfile
# Stage 1: Install dependencies
FROM node:20.12-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# Stage 2: Build (TypeScript compile)
FROM node:20.12-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 3: Runtime
FROM node:20.12-alpine AS runtime
WORKDIR /app

RUN addgroup -S appgroup && adduser -S appuser -G appgroup

COPY --from=deps /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY package.json ./

RUN chown -R appuser:appgroup /app

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://localhost:3000/health || exit 1

EXPOSE 3000
CMD ["node", "dist/index.js"]
```

## Java Application (Maven Build + JRE Runtime)

```dockerfile
# Stage 1: Build with Maven
FROM maven:3.9-eclipse-temurin-21 AS builder
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:go-offline -B    # Cache dependencies
COPY src ./src
RUN mvn package -DskipTests -B

# Stage 2: Extract layers for optimal caching (Spring Boot)
FROM eclipse-temurin:21-jre-jammy AS runtime
WORKDIR /app

RUN groupadd -r appgroup && useradd -r -g appgroup appuser

COPY --from=builder /app/target/*.jar app.jar

RUN chown appuser:appgroup app.jar

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/actuator/health || exit 1

EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]
```

## Build Arguments for Version Pinning

```dockerfile
ARG GO_VERSION=1.22
ARG ALPINE_VERSION=3.19

FROM golang:${GO_VERSION}-alpine${ALPINE_VERSION} AS builder
```

## Layer Caching Strategy
1. Copy dependency manifests (`go.mod`, `package.json`, `pom.xml`) first
2. Run dependency install as a separate `RUN` step
3. Copy source code last (changes most frequently, invalidates fewer cached layers)
4. Use `--mount=type=cache` for package manager caches (BuildKit)

```dockerfile
# BuildKit cache mount (keeps package cache across builds)
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download
```

## Common Mistakes
- `COPY . .` before `COPY go.mod go.sum ./` — every source change invalidates the dependency install layer
- Not using `AS` named stages — unnamed stages cannot be referenced with `--from`
- Leaving build tools in final stage — gcc, git, make are not needed at runtime and increase attack surface
- Using `CMD` in intermediate stages — only the final stage's CMD matters
