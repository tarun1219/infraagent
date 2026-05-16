package main

# Containers should set readOnlyRootFilesystem: true in their securityContext.
# A read-only root filesystem prevents attackers from writing malicious files
# to the container filesystem after a compromise. Writable volumes should be
# mounted explicitly at specific paths only.

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.readOnlyRootFilesystem
    msg := sprintf("Container '%v' in Deployment '%v' should set securityContext.readOnlyRootFilesystem: true", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    container.securityContext.readOnlyRootFilesystem == false
    msg := sprintf("Container '%v' in Deployment '%v' has readOnlyRootFilesystem: false — enable it and use explicit volume mounts", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "StatefulSet"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.readOnlyRootFilesystem
    msg := sprintf("Container '%v' in StatefulSet '%v' should set securityContext.readOnlyRootFilesystem: true", [container.name, input.metadata.name])
}
