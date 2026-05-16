package main

# Dockerfiles must include a HEALTHCHECK instruction.
# Without HEALTHCHECK, Docker and Kubernetes (when not using probes) cannot
# determine whether the application inside the container is actually ready to
# serve traffic. A HEALTHCHECK enables automatic detection of unhealthy
# containers and triggers restarts or removal from load balancer rotation.

deny[msg] {
    not has_healthcheck
    msg := "Dockerfile is missing a HEALTHCHECK instruction — add 'HEALTHCHECK CMD ...' to enable container health monitoring"
}

has_healthcheck {
    cmd := input.Stages[_].Commands[_]
    cmd.Cmd == "healthcheck"
}
