# Kubernetes Jobs and CronJobs

## Overview
- **Job**: Runs pods to completion. Useful for batch processing, migrations, and one-off tasks.
- **CronJob**: Schedules Jobs on a cron schedule. Replaces cron daemons in containerized environments.

## Job (Batch Processing)

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration
  namespace: production
spec:
  completions: 1           # Number of successful completions needed
  parallelism: 1           # Pods running in parallel
  backoffLimit: 3          # Retry up to 3 times before marking as failed
  activeDeadlineSeconds: 600  # Kill job if not done in 10 minutes
  ttlSecondsAfterFinished: 3600  # Clean up 1 hour after completion
  template:
    metadata:
      labels:
        app: db-migration
        job-type: migration
    spec:
      restartPolicy: OnFailure    # Never | OnFailure (not Always)
      serviceAccountName: migration-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
      containers:
        - name: migration
          image: my-registry/db-migrator:1.0.5
          command: ["./migrate", "--up"]
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: url
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
```

## CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: nightly-report
  namespace: production
spec:
  schedule: "0 2 * * *"              # 2am UTC every day
  timeZone: "America/New_York"        # Requires K8s 1.27+ or time zone feature gate
  concurrencyPolicy: Forbid           # Don't start new job if previous still running
  successfulJobsHistoryLimit: 3       # Keep last 3 successful job records
  failedJobsHistoryLimit: 5           # Keep last 5 failed job records
  startingDeadlineSeconds: 300        # Skip if not started within 5 min of schedule
  suspend: false
  jobTemplate:
    spec:
      backoffLimit: 2
      activeDeadlineSeconds: 3600
      template:
        spec:
          restartPolicy: OnFailure
          serviceAccountName: report-sa
          containers:
            - name: reporter
              image: my-registry/reporter:2.1.0
              resources:
                requests:
                  cpu: "200m"
                  memory: "256Mi"
                limits:
                  cpu: "1"
                  memory: "1Gi"
```

## Concurrency Policies

| Policy | Behavior |
|--------|----------|
| `Allow` | Multiple jobs can run concurrently (default) |
| `Forbid` | Skip new job if previous is still running |
| `Replace` | Cancel running job and start new one |

## Indexed Jobs (for parallel work queues)

```yaml
spec:
  completions: 10
  parallelism: 5
  completionMode: Indexed    # Each pod gets JOB_COMPLETION_INDEX env var (0-9)
```

## Common Mistakes
- `restartPolicy: Always` — not allowed for Jobs; use `OnFailure` or `Never`
- Missing `activeDeadlineSeconds` — a stuck job runs forever, accumulating costs
- `concurrencyPolicy: Allow` for jobs that modify shared state — can cause data corruption
- No `ttlSecondsAfterFinished` — completed Job resources accumulate in etcd over time
- Scheduling during business hours — CronJob time is cluster time (often UTC); verify timezone
