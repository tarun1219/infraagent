package main

# Deployments and StatefulSets must have livenessProbe on all containers.
# A missing liveness probe means Kubernetes cannot detect and restart deadlocked
# containers, leading to silent failures.

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.livenessProbe
    msg := sprintf("Container '%v' in Deployment '%v' is missing livenessProbe", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "StatefulSet"
    container := input.spec.template.spec.containers[_]
    not container.livenessProbe
    msg := sprintf("Container '%v' in StatefulSet '%v' is missing livenessProbe", [container.name, input.metadata.name])
}
