package main

# Containers must drop ALL Linux capabilities.
# Capabilities like NET_RAW, SYS_ADMIN, and CHOWN grant elevated privileges
# that are rarely needed by application containers. Dropping ALL capabilities
# and adding back only what is explicitly required follows least-privilege.

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.capabilities.drop
    msg := sprintf("Container '%v' in Deployment '%v' must drop all capabilities via securityContext.capabilities.drop", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    drop := container.securityContext.capabilities.drop
    not contains_all(drop)
    msg := sprintf("Container '%v' in Deployment '%v' must include 'ALL' in capabilities.drop", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "StatefulSet"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.capabilities.drop
    msg := sprintf("Container '%v' in StatefulSet '%v' must drop all capabilities via securityContext.capabilities.drop", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "DaemonSet"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.capabilities.drop
    msg := sprintf("Container '%v' in DaemonSet '%v' must drop all capabilities via securityContext.capabilities.drop", [container.name, input.metadata.name])
}

contains_all(drop) {
    drop[_] == "ALL"
}
