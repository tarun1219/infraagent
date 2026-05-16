package main

# Resources must explicitly specify a namespace.
# Relying on the 'default' namespace makes it easy to accidentally expose or
# pollute resources. Every workload should be isolated in a named namespace.

deny[msg] {
    input.kind == "Deployment"
    not input.metadata.namespace
    msg := sprintf("Deployment '%v' does not specify a namespace — avoid using the default namespace", [input.metadata.name])
}

deny[msg] {
    input.kind == "Deployment"
    input.metadata.namespace == "default"
    msg := sprintf("Deployment '%v' uses the 'default' namespace — deploy to a dedicated namespace", [input.metadata.name])
}

deny[msg] {
    input.kind == "Service"
    not input.metadata.namespace
    msg := sprintf("Service '%v' does not specify a namespace", [input.metadata.name])
}

deny[msg] {
    input.kind == "Service"
    input.metadata.namespace == "default"
    msg := sprintf("Service '%v' uses the 'default' namespace — deploy to a dedicated namespace", [input.metadata.name])
}

deny[msg] {
    input.kind == "StatefulSet"
    not input.metadata.namespace
    msg := sprintf("StatefulSet '%v' does not specify a namespace", [input.metadata.name])
}
