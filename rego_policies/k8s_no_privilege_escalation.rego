package main

# Containers must set allowPrivilegeEscalation: false.
# Without this flag, a process inside the container could use setuid binaries
# or sudo to gain additional privileges beyond what was originally granted.
# This is required for CIS Kubernetes Benchmark compliance.

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.allowPrivilegeEscalation == false
    msg := sprintf("Container '%v' in Deployment '%v' must set securityContext.allowPrivilegeEscalation: false", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "StatefulSet"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.allowPrivilegeEscalation == false
    msg := sprintf("Container '%v' in StatefulSet '%v' must set securityContext.allowPrivilegeEscalation: false", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "DaemonSet"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.allowPrivilegeEscalation == false
    msg := sprintf("Container '%v' in DaemonSet '%v' must set securityContext.allowPrivilegeEscalation: false", [container.name, input.metadata.name])
}
