package main

# Service selector must contain at least one label key.
# An empty or missing selector means the Service will not route traffic to
# any pods (or will match all pods, which is a security and reliability risk).

deny[msg] {
    input.kind == "Service"
    input.spec.type != "ExternalName"
    not input.spec.selector
    msg := sprintf("Service '%v' is missing spec.selector — it will not route traffic to any pods", [input.metadata.name])
}

deny[msg] {
    input.kind == "Service"
    input.spec.type != "ExternalName"
    count(input.spec.selector) == 0
    msg := sprintf("Service '%v' has an empty spec.selector — add at least one label key to select target pods", [input.metadata.name])
}

deny[msg] {
    input.kind == "Service"
    input.spec.type != "ExternalName"
    not input.spec.selector.app
    msg := sprintf("Service '%v' selector does not include the 'app' label key — ensure it matches pod template labels", [input.metadata.name])
}
