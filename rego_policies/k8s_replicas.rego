package main

# Deployments should have at least 2 replicas for high availability.
# A single replica creates a single point of failure; node maintenance or
# pod eviction will cause downtime. Minimum 2 replicas is the baseline for
# any production workload.

deny[msg] {
    input.kind == "Deployment"
    replicas := input.spec.replicas
    replicas < 2
    msg := sprintf("Deployment '%v' has only %v replica(s) — set replicas >= 2 for high availability", [input.metadata.name, replicas])
}

deny[msg] {
    input.kind == "Deployment"
    not input.spec.replicas
    msg := sprintf("Deployment '%v' does not set spec.replicas — explicitly set replicas >= 2", [input.metadata.name])
}
