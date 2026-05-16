# Kubernetes HorizontalPodAutoscaler (HPA)

## Overview
HPA automatically scales the number of pod replicas based on observed metrics. v2 (GA in 1.25) supports CPU, memory, and custom/external metrics with fine-grained behavior configuration.

## HPA v2 with CPU and Memory

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-app-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60     # Scale when average CPU > 60%
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 75
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60     # Wait 60s before scaling up again
      policies:
        - type: Pods
          value: 4
          periodSeconds: 60              # Add at most 4 pods per 60s
        - type: Percent
          value: 100
          periodSeconds: 60             # Or double the replica count
      selectPolicy: Max                 # Use whichever policy adds more pods
    scaleDown:
      stabilizationWindowSeconds: 300   # Wait 5min before scaling down
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60             # Remove at most 2 pods per 60s
```

## HPA with Custom Metrics (Prometheus Adapter)

```yaml
metrics:
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"           # 100 req/s per pod
  - type: External
    external:
      metric:
        name: sqs_queue_depth
        selector:
          matchLabels:
            queue: my-app-queue
      target:
        type: AverageValue
        averageValue: "50"
```

## Prerequisites
- Metrics Server must be installed (`kubectl top pods` works)
- Containers must have `resources.requests.cpu` set for CPU-based HPA
- For custom metrics: Prometheus Adapter or KEDA must be installed

## Behavior Configuration Tips
- `stabilizationWindowSeconds` prevents flapping — use 300s+ for scale-down
- Always set `minReplicas >= 2` to maintain HA during scale-down events
- `maxReplicas` should account for cluster node capacity
- The HPA does NOT override a Deployment's `replicas` field permanently — they co-exist

## Common Mistakes
- Using HPA with `replicas` set in the Deployment manifest managed by Argo CD/Flux — the GitOps tool will reset replicas on every sync; use server-side apply or ignore the `replicas` field
- Setting `minReplicas: 1` — HPA can scale to 1 during low traffic, creating a single point of failure
- Missing resource requests — CPU-based HPA cannot calculate utilization without requests
