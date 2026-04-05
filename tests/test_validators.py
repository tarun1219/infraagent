"""
Unit tests for IaCBench validators.
"""
import pytest
from iachench.validators.kubernetes_validator import KubernetesValidator
from iachench.validators.terraform_validator import TerraformValidator
from iachench.validators.dockerfile_validator import DockerfileValidator


# ── Kubernetes Validator Tests ───────────────────────────────────────────────

class TestKubernetesValidator:
    def setup_method(self):
        self.validator = KubernetesValidator()

    def test_valid_deployment(self):
        manifest = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: test-app
  template:
    metadata:
      labels:
        app: test-app
    spec:
      containers:
      - name: app
        image: nginx:1.25.3
        resources:
          limits:
            cpu: "500m"
            memory: "128Mi"
          requests:
            cpu: "250m"
            memory: "64Mi"
"""
        result = self.validator.validate(manifest)
        assert result["syntax_valid"] is True

    def test_invalid_yaml_syntax(self):
        manifest = "invalid: yaml: [unclosed"
        result = self.validator.validate(manifest)
        assert result["syntax_valid"] is False
        assert result["errors"]

    def test_deprecated_api_version_flagged(self):
        manifest = """
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: old-app
spec:
  template:
    spec:
      containers:
      - name: app
        image: nginx:latest
"""
        result = self.validator.validate(manifest)
        # Should flag deprecated API or not pass security checks
        assert "extensions/v1beta1" in str(result.get("warnings", []) + result.get("errors", []))

    def test_root_user_flagged(self):
        manifest = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: insecure-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: insecure-app
  template:
    metadata:
      labels:
        app: insecure-app
    spec:
      containers:
      - name: app
        image: nginx:1.25.3
        securityContext:
          runAsUser: 0
"""
        result = self.validator.validate(manifest)
        assert result.get("security_valid") is False or any(
            "root" in str(e).lower() or "CKV_K8S_30" in str(e)
            for e in result.get("errors", []) + result.get("warnings", [])
        )

    def test_missing_resource_limits_flagged(self):
        manifest = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: no-limits
spec:
  replicas: 1
  selector:
    matchLabels:
      app: no-limits
  template:
    metadata:
      labels:
        app: no-limits
    spec:
      containers:
      - name: app
        image: nginx:1.25.3
"""
        result = self.validator.validate(manifest)
        issues = result.get("errors", []) + result.get("warnings", [])
        assert any("limit" in str(i).lower() or "resource" in str(i).lower() for i in issues)


# ── Terraform Validator Tests ────────────────────────────────────────────────

class TestTerraformValidator:
    def setup_method(self):
        self.validator = TerraformValidator()

    def test_valid_s3_bucket(self):
        hcl = """
resource "aws_s3_bucket" "test" {
  bucket = "my-test-bucket"
}

resource "aws_s3_bucket_public_access_block" "test" {
  bucket                  = aws_s3_bucket.test.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
"""
        result = self.validator.validate(hcl)
        assert result["syntax_valid"] is True

    def test_wildcard_iam_action_flagged(self):
        hcl = """
resource "aws_iam_policy" "bad" {
  policy = jsonencode({
    Statement = [{
      Effect   = "Allow"
      Action   = "*"
      Resource = "*"
    }]
  })
}
"""
        result = self.validator.validate(hcl)
        issues = result.get("security_issues", []) + result.get("errors", [])
        assert any(
            "wildcard" in str(i).lower() or "CKV_AWS_40" in str(i) or "*" in str(i)
            for i in issues
        )

    def test_invalid_hcl_syntax(self):
        hcl = 'resource "aws_s3_bucket" { unclosed'
        result = self.validator.validate(hcl)
        assert result["syntax_valid"] is False


# ── Dockerfile Validator Tests ───────────────────────────────────────────────

class TestDockerfileValidator:
    def setup_method(self):
        self.validator = DockerfileValidator()

    def test_valid_dockerfile(self):
        dockerfile = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY --chown=appuser:appuser . .
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8080/health || exit 1
CMD ["python", "app.py"]
"""
        result = self.validator.validate(dockerfile)
        assert result["syntax_valid"] is True

    def test_root_user_flagged(self):
        dockerfile = """FROM python:3.11-slim
WORKDIR /app
CMD ["python", "app.py"]
"""
        result = self.validator.validate(dockerfile)
        # No USER directive — should flag root
        issues = result.get("security_issues", []) + result.get("errors", [])
        assert any(
            "root" in str(i).lower() or "USER" in str(i) or "CKV_DOCKER_8" in str(i)
            for i in issues
        )

    def test_latest_tag_flagged(self):
        dockerfile = "FROM nginx:latest\nCMD [\"nginx\", \"-g\", \"daemon off;\"]"
        result = self.validator.validate(dockerfile)
        issues = result.get("warnings", []) + result.get("errors", [])
        assert any("latest" in str(i).lower() or "CKV_DOCKER_7" in str(i) for i in issues)

    def test_add_vs_copy_flagged(self):
        dockerfile = """FROM ubuntu:22.04
ADD app.tar.gz /app/
CMD ["/app/start"]
"""
        result = self.validator.validate(dockerfile)
        issues = result.get("warnings", []) + result.get("errors", [])
        assert any("ADD" in str(i) or "COPY" in str(i) or "CKV_DOCKER_2" in str(i) for i in issues)
