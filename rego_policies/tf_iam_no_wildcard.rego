package main

# IAM policies must not use "*" as an Action or Resource value.
# Wildcard actions (e.g. "Action": "*") grant full access to an AWS service
# or all services, violating the principle of least privilege. Always enumerate
# the specific actions required.

deny[msg] {
    policy := input.resource.aws_iam_policy[name]
    doc := json.unmarshal(policy.policy)
    statement := doc.Statement[_]
    action := statement.Action[_]
    action == "*"
    msg := sprintf("IAM policy '%v' uses wildcard Action '*' — enumerate specific actions instead", [name])
}

deny[msg] {
    policy := input.resource.aws_iam_policy[name]
    doc := json.unmarshal(policy.policy)
    statement := doc.Statement[_]
    statement.Action == "*"
    msg := sprintf("IAM policy '%v' uses wildcard Action '*' — enumerate specific actions instead", [name])
}

deny[msg] {
    role := input.resource.aws_iam_role[name]
    doc := json.unmarshal(role.assume_role_policy)
    statement := doc.Statement[_]
    statement.Principal == "*"
    msg := sprintf("IAM role '%v' assume_role_policy allows Principal '*' — restrict to specific principals", [name])
}

deny[msg] {
    policy := input.resource.aws_iam_role_policy[name]
    doc := json.unmarshal(policy.policy)
    statement := doc.Statement[_]
    statement.Resource == "*"
    statement.Action == "*"
    msg := sprintf("IAM role policy '%v' grants full access ('Action: *' on 'Resource: *') — use least-privilege permissions", [name])
}
