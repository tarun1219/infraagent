package main

# Deployment spec.selector.matchLabels must be a subset of (or match)
# the pod template labels. A mismatch causes the Deployment controller to
# fail to find its pods, which results in perpetually unavailable rollouts.

deny[msg] {
    input.kind == "Deployment"
    selector_label := input.spec.selector.matchLabels[key]
    template_labels := input.spec.template.metadata.labels
    not template_labels[key] == selector_label
    msg := sprintf("Deployment '%v': selector.matchLabels key '%v'='%v' not found in pod template labels", [input.metadata.name, key, selector_label])
}

deny[msg] {
    input.kind == "Deployment"
    not input.spec.selector.matchLabels
    msg := sprintf("Deployment '%v' is missing spec.selector.matchLabels", [input.metadata.name])
}

deny[msg] {
    input.kind == "Deployment"
    not input.spec.template.metadata.labels
    msg := sprintf("Deployment '%v' pod template is missing metadata.labels", [input.metadata.name])
}
