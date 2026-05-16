# Dockerfile Security Scanning

## Overview
Container image scanning identifies known CVEs in OS packages and language dependencies before images are deployed. Integrate scanning into CI/CD pipelines to catch vulnerabilities early.

## Trivy (Recommended — Open Source, Fast)

```bash
# Scan a local image
trivy image --exit-code 1 \
            --severity HIGH,CRITICAL \
            --ignore-unfixed \
            my-image:1.0.0

# Scan a Dockerfile for misconfigurations
trivy config ./Dockerfile

# Scan filesystem (for CI scanning source before build)
trivy fs --severity HIGH,CRITICAL .

# Output as JSON for downstream processing
trivy image --format json --output trivy-report.json my-image:1.0.0

# SBOM generation (CycloneDX format)
trivy image --format cyclonedx --output sbom.json my-image:1.0.0
```

## Grype (Anchore, Alternative to Trivy)

```bash
# Install
curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin

# Scan image
grype my-image:1.0.0 --fail-on high

# Scan SBOM
grype sbom:./sbom.json

# Output table
grype my-image:1.0.0 -o table
```

## GitHub Actions Integration

```yaml
name: Container Security Scan

on:
  push:
    branches: [main]
  pull_request:

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: docker build -t my-image:${{ github.sha }} .

      - name: Run Trivy vulnerability scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: my-image:${{ github.sha }}
          format: sarif
          output: trivy-results.sarif
          severity: HIGH,CRITICAL
          exit-code: 1

      - name: Upload Trivy scan results to GitHub Security
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif

      - name: Dockerfile lint (Hadolint)
        uses: hadolint/hadolint-action@v3.1.0
        with:
          dockerfile: Dockerfile
          failure-threshold: warning
```

## Hadolint (Dockerfile Linting)

```bash
# Lint Dockerfile
hadolint Dockerfile

# With ignored rules
hadolint --ignore DL3008 --ignore DL3013 Dockerfile

# JSON output for CI
hadolint --format json Dockerfile
```

Common Hadolint rules:
- `DL3008` — pin versions in apt-get install
- `DL3013` — pin versions in pip install
- `DL3018` — pin versions in apk add
- `DL3025` — use JSON notation for CMD and ENTRYPOINT
- `SC2086` — double quote shell variables to prevent word splitting

## SBOM Generation

```bash
# Syft SBOM generation
syft my-image:1.0.0 -o cyclonedx-json=sbom.json
syft my-image:1.0.0 -o spdx-json=sbom.spdx.json

# Attach SBOM as image attestation (with cosign)
cosign attest --type cyclonedx --predicate sbom.json my-image:1.0.0
```

## Vulnerability Exceptions (.trivyignore)

```
# Format: CVE-ID [expiry-date] [comment]
CVE-2023-12345 exp:2024-12-31 # Upstream patch pending, mitigated by WAF
CVE-2022-99999                # False positive — package not loaded at runtime
```

## Scanning in Registry (ECR)

```hcl
resource "aws_ecr_repository" "app" {
  name                 = "my-app"
  image_tag_mutability = "IMMUTABLE"    # Prevent tag overwriting

  image_scanning_configuration {
    scan_on_push = true    # Automatically scan images on push
  }

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.ecr.arn
  }
}
```

## Common Mistakes
- Only scanning at build time — images in registries accumulate CVEs over time; set up continuous registry scanning
- Ignoring HIGH severity in CI — HIGH CVEs are regularly exploitable; only ignore with documented justification and expiry
- No baseline — without a `.trivyignore` or accepted-risk process, teams disable scanning entirely due to alert fatigue
- Scanning without SBOM — SBOMs are needed for supply chain attestation and compliance (SLSA, NIST SSDF)
