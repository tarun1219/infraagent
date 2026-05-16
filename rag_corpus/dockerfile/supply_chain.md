# Dockerfile Supply Chain Security

## Overview
Supply chain security ensures that the images you build and deploy are trustworthy: the base image is authentic, the build process is reproducible, the artifacts have not been tampered with, and the provenance can be verified. Key tools: Sigstore/cosign for signing, SLSA for provenance, and SBOMs for transparency.

## Image Signing with cosign (Sigstore)

```bash
# Install cosign
brew install cosign    # macOS
# or: go install github.com/sigstore/cosign/v2/cmd/cosign@latest

# Generate key pair (for keyful signing)
cosign generate-key-pair
# Produces: cosign.key (private), cosign.pub (public)

# Sign an image (keyful)
cosign sign --key cosign.key my-registry/my-image:1.0.0

# Sign with keyless (OIDC — recommended for CI/CD)
cosign sign my-registry/my-image:1.0.0
# Uses Fulcio (CA) and Rekor (transparency log) — no key management needed

# Verify a signed image
cosign verify --key cosign.pub my-registry/my-image:1.0.0

# Verify keyless signature
cosign verify \
  --certificate-identity-regexp "https://github.com/my-org/my-repo" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  my-registry/my-image:1.0.0
```

## SBOM Attestation

```bash
# Generate SBOM with Syft
syft my-registry/my-image:1.0.0 -o cyclonedx-json=sbom.json

# Attach SBOM as a cosign attestation
cosign attest \
  --key cosign.key \
  --type cyclonedx \
  --predicate sbom.json \
  my-registry/my-image:1.0.0

# Verify SBOM attestation
cosign verify-attestation \
  --key cosign.pub \
  --type cyclonedx \
  my-registry/my-image:1.0.0 | jq '.payload | @base64d | fromjson'
```

## SLSA Provenance (GitHub Actions)

```yaml
name: Build and Sign

on:
  push:
    tags: ["v*"]

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      id-token: write      # For keyless signing with Sigstore
      packages: write
      contents: read

    outputs:
      image-digest: ${{ steps.build.outputs.digest }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.ref_name }}
          sbom: true         # Attach SBOM to image
          provenance: true   # Attach SLSA provenance

      - name: Sign image with cosign (keyless)
        uses: sigstore/cosign-installer@v3

      - run: |
          cosign sign --yes \
            ghcr.io/${{ github.repository }}@${{ steps.build.outputs.digest }}
```

## Verifying Base Images

```bash
# Verify the base image before using it in your build
cosign verify \
  --certificate-oidc-issuer https://accounts.google.com \
  gcr.io/distroless/static-debian12:nonroot

# Pin base image to verified digest in Dockerfile
FROM gcr.io/distroless/static-debian12@sha256:abc123...
```

## Kubernetes Admission Policy (Require Signed Images)

```yaml
# Using Kyverno to enforce image signing
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-signed-images
spec:
  validationFailureAction: Enforce
  rules:
    - name: check-image-signature
      match:
        resources:
          kinds: ["Pod"]
      verifyImages:
        - imageReferences: ["my-registry/*"]
          attestors:
            - count: 1
              entries:
                - keyless:
                    subject: "https://github.com/my-org/my-repo/.github/workflows/*"
                    issuer: "https://token.actions.githubusercontent.com"
```

## Dependency Pinning in Dockerfiles

```dockerfile
# Pin OS packages to specific versions
RUN apk add --no-cache \
    curl=8.5.0-r0 \
    openssl=3.1.4-r5

# Pin pip packages
RUN pip install --no-cache-dir \
    flask==3.0.2 \
    gunicorn==21.2.0

# Use lock files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt    # requirements.txt has pinned versions
```

## Common Mistakes
- Not signing images — unsigned images can be replaced by attackers with access to the registry
- Using mutable tags (`:latest`, `:main`) in Kubernetes — these can be silently updated; use digests or immutable tags
- No SBOM — without an SBOM, you cannot quickly determine if your deployed images are affected by a newly disclosed CVE
- Trusting base images without verification — even official images can be compromised; verify signatures when available
- Storing cosign private keys in plaintext files — use hardware tokens, cloud KMS, or keyless signing in CI/CD
