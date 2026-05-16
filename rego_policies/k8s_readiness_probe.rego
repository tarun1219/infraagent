package main

# Deployments must have readinessProbe on all containers.
# Without readinessProbe, Kubernetes routes traffic to containers that are not
# yet ready to serve requests, causing errors during rollouts.

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.readinessProbe
    msg := sprintf("Container '%v' in Deployment '%v' is missing readinessProbe", [container.name, input.metadata.name])
}
