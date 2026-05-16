package main

# Dockerfiles must include a USER instruction that sets a non-root user.
# Running containers as root (UID 0) is a significant security risk.
# If the container is compromised, the attacker has root privileges on the host
# (subject only to container runtime restrictions). Always create and switch to
# a dedicated non-root user.

deny[msg] {
    input.Stages[_].Commands[_].Cmd == "user"
    not has_non_root_user
    msg := "Dockerfile sets USER but it may resolve to root — ensure USER is set to a non-root UID or username"
}

deny[msg] {
    not has_user_instruction
    msg := "Dockerfile is missing a USER instruction — add 'USER nonroot' or a specific UID (e.g. USER 1001) to avoid running as root"
}

has_user_instruction {
    cmd := input.Stages[_].Commands[_]
    cmd.Cmd == "user"
}

has_non_root_user {
    cmd := input.Stages[_].Commands[_]
    cmd.Cmd == "user"
    user := cmd.Value[0]
    user != "root"
    user != "0"
    user != "0:0"
    user != "root:root"
}
