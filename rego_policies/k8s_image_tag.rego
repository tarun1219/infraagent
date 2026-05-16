package main

# Container images must not use the ':latest' tag.
# The 'latest' tag is mutable, making deployments non-reproducible and
# preventing effective rollback. Pin images to a specific digest or semver tag.

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    endswith(container.image, ":latest")
    msg := sprintf("Container '%v' in Deployment '%v' uses ':latest' image tag — pin to a specific version", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not contains(container.image, ":")
    msg := sprintf("Container '%v' in Deployment '%v' has no image tag — pin to a specific version", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "StatefulSet"
    container := input.spec.template.spec.containers[_]
    endswith(container.image, ":latest")
    msg := sprintf("Container '%v' in StatefulSet '%v' uses ':latest' image tag — pin to a specific version", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "DaemonSet"
    container := input.spec.template.spec.containers[_]
    endswith(container.image, ":latest")
    msg := sprintf("Container '%v' in DaemonSet '%v' uses ':latest' image tag — pin to a specific version", [container.name, input.metadata.name])
}
