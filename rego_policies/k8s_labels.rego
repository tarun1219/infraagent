package main

# Resources must have an 'app' label in metadata.labels.
# The 'app' label is a standard convention used by Services, HPAs, and tooling
# (Helm, Argo CD) to identify and select workloads. Missing labels break
# service discovery and observability.

deny[msg] {
    input.kind == "Deployment"
    not input.metadata.labels.app
    msg := sprintf("Deployment '%v' is missing required label 'app' in metadata.labels", [input.metadata.name])
}

deny[msg] {
    input.kind == "StatefulSet"
    not input.metadata.labels.app
    msg := sprintf("StatefulSet '%v' is missing required label 'app' in metadata.labels", [input.metadata.name])
}

deny[msg] {
    input.kind == "Service"
    not input.metadata.labels.app
    msg := sprintf("Service '%v' is missing required label 'app' in metadata.labels", [input.metadata.name])
}

deny[msg] {
    input.kind == "DaemonSet"
    not input.metadata.labels.app
    msg := sprintf("DaemonSet '%v' is missing required label 'app' in metadata.labels", [input.metadata.name])
}
