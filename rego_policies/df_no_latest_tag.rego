package main

# FROM instructions must not use the ':latest' tag or an untagged image.
# The 'latest' tag is mutable — the underlying image can change silently,
# breaking reproducibility and making vulnerability tracking impossible.
# Pin all base images to a specific version tag or SHA256 digest.

deny[msg] {
    stage := input.Stages[_]
    from := stage.From
    endswith(from.Image, ":latest")
    msg := sprintf("Dockerfile stage '%v' uses ':latest' tag for base image '%v' — pin to a specific version or digest", [stage.Name, from.Image])
}

deny[msg] {
    stage := input.Stages[_]
    from := stage.From
    not contains(from.Image, ":")
    not from.Scratch
    msg := sprintf("Dockerfile stage '%v' uses untagged base image '%v' — pin to a specific version or digest", [stage.Name, from.Image])
}
