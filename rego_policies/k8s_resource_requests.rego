package main

# All containers must set resources.requests for CPU and memory.
# Requests are used by the scheduler to place pods on nodes with adequate
# capacity. Without requests, pods may be scheduled onto overloaded nodes.

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.resources.requests.cpu
    msg := sprintf("Container '%v' in Deployment '%v' is missing resources.requests.cpu", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.resources.requests.memory
    msg := sprintf("Container '%v' in Deployment '%v' is missing resources.requests.memory", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "StatefulSet"
    container := input.spec.template.spec.containers[_]
    not container.resources.requests.cpu
    msg := sprintf("Container '%v' in StatefulSet '%v' is missing resources.requests.cpu", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "StatefulSet"
    container := input.spec.template.spec.containers[_]
    not container.resources.requests.memory
    msg := sprintf("Container '%v' in StatefulSet '%v' is missing resources.requests.memory", [container.name, input.metadata.name])
}
