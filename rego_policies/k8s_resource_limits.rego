package main

# All containers must set resources.limits.cpu and resources.limits.memory.
# Without limits, a noisy-neighbour container can exhaust node resources,
# causing evictions and outages for other workloads on the same node.

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.resources.limits.cpu
    msg := sprintf("Container '%v' in Deployment '%v' is missing resources.limits.cpu", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.resources.limits.memory
    msg := sprintf("Container '%v' in Deployment '%v' is missing resources.limits.memory", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "StatefulSet"
    container := input.spec.template.spec.containers[_]
    not container.resources.limits.cpu
    msg := sprintf("Container '%v' in StatefulSet '%v' is missing resources.limits.cpu", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "StatefulSet"
    container := input.spec.template.spec.containers[_]
    not container.resources.limits.memory
    msg := sprintf("Container '%v' in StatefulSet '%v' is missing resources.limits.memory", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "DaemonSet"
    container := input.spec.template.spec.containers[_]
    not container.resources.limits.memory
    msg := sprintf("Container '%v' in DaemonSet '%v' is missing resources.limits.memory", [container.name, input.metadata.name])
}
