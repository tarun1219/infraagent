# Kubernetes Health Probe Patterns

## Overview
Kubernetes supports three types of container probes. Each probe uses one of three mechanisms: HTTP GET, TCP Socket, or Exec command.

## Probe Types

| Probe | Purpose | Failure Action |
|-------|---------|----------------|
| `livenessProbe` | Is the container alive? | Container is killed and restarted |
| `readinessProbe` | Is the container ready to serve traffic? | Removed from Service endpoints |
| `startupProbe` | Has the container finished starting? | Delays liveness/readiness checks |

## HTTP Probes (most common)

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
    httpHeaders:
      - name: Custom-Header
        value: health-check
  initialDelaySeconds: 15    # Wait before first probe
  periodSeconds: 20          # How often to probe
  timeoutSeconds: 5          # Probe timeout
  failureThreshold: 3        # Restarts after 3 consecutive failures
  successThreshold: 1        # Only 1 success needed to pass

readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
  failureThreshold: 3
  successThreshold: 1
```

## Startup Probe (for slow-starting apps)

```yaml
startupProbe:
  httpGet:
    path: /healthz
    port: 8080
  failureThreshold: 30       # Allow 30 * 10s = 300s for startup
  periodSeconds: 10
# livenessProbe and readinessProbe only start after startupProbe succeeds
```

## TCP Socket Probe (for non-HTTP services)

```yaml
livenessProbe:
  tcpSocket:
    port: 5432             # Check that the port accepts connections
  initialDelaySeconds: 30
  periodSeconds: 10
```

## Exec Probe (for custom checks)

```yaml
livenessProbe:
  exec:
    command:
      - /bin/sh
      - -c
      - "pg_isready -U postgres -h 127.0.0.1"
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
```

## Separate Liveness and Readiness Endpoints

```
GET /healthz  → liveness: returns 200 if process is alive (not deadlocked)
GET /ready    → readiness: returns 200 if app can handle requests
              (checks DB connectivity, cache warmup, etc.)
```

A container should be removed from load balancer rotation (readiness fails) before it's restarted (liveness fails). Keep liveness checks lightweight — heavy checks can cause cascading restarts.

## Probe Timing for JVM Apps

```yaml
startupProbe:
  httpGet:
    path: /healthz
    port: 8080
  failureThreshold: 20    # 20 * 15s = 300s max startup time
  periodSeconds: 15

livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
  initialDelaySeconds: 0  # startupProbe handles initial delay
  periodSeconds: 30
  timeoutSeconds: 10
  failureThreshold: 3
```

## Common Mistakes
- Using the same endpoint for liveness and readiness — if the app is slow (not ready), it gets killed and restarted in a loop
- `initialDelaySeconds` too short for slow-starting apps — use `startupProbe` instead
- Liveness probe checks external dependencies (DB, other services) — if those fail, all pods restart simultaneously, making the outage worse
- No probes at all — Kubernetes routes traffic to containers immediately on startup, even if the app hasn't finished initializing
