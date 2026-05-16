package main

# Containers must set securityContext.runAsNonRoot: true.
# Running containers as root greatly expands the blast radius of a container
# escape. Setting runAsNonRoot prevents processes from running as UID 0.

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.runAsNonRoot
    msg := sprintf("Container '%v' in Deployment '%v' must set securityContext.runAsNonRoot: true", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    container.securityContext.runAsNonRoot == false
    msg := sprintf("Container '%v' in Deployment '%v' has runAsNonRoot: false — must be true", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "StatefulSet"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.runAsNonRoot
    msg := sprintf("Container '%v' in StatefulSet '%v' must set securityContext.runAsNonRoot: true", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "DaemonSet"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.runAsNonRoot
    msg := sprintf("Container '%v' in DaemonSet '%v' must set securityContext.runAsNonRoot: true", [container.name, input.metadata.name])
}
