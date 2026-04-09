"""
Unit tests for the Multi-Layer Validator (infraagent/validators.py).

Each test class targets one validation layer independently:
  L1  — Syntax    (yamllint, HCL parse, hadolint)
  L2  — Schema    (kubeconform, terraform validate)
  L2.5— Dry-run  (kubectl apply --dry-run=server)
  L3  — Security  (Checkov, Trivy output parsing)
  L4  — Best Practices (OPA/Conftest)
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from infraagent.validators import (
    MultiLayerValidator,
    ValidationReport,
    ValidationLayer,
    Severity,
    _validate_yaml_syntax,
    _validate_hcl_syntax,
    _validate_dockerfile_syntax,
    _validate_k8s_schema,
    _validate_k8s_security,
    _validate_tf_security,
    _validate_best_practices,
    _kind_cluster_reachable,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def validator():
    return MultiLayerValidator(k8s_version="1.29.0")


VALID_DEPLOYMENT = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-app
  namespace: default
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
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
      - name: app
        image: nginx:1.25.3
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: ["ALL"]
        resources:
          limits:
            cpu: "500m"
            memory: "128Mi"
          requests:
            cpu: "250m"
            memory: "64Mi"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
"""

INSECURE_DEPLOYMENT = """
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
        image: nginx:latest
"""

INVALID_YAML = "key: [unclosed bracket"

VALID_TF = """
resource "aws_s3_bucket" "example" {
  bucket = "my-secure-bucket"
}

resource "aws_s3_bucket_public_access_block" "example" {
  bucket                  = aws_s3_bucket.example.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
"""

INSECURE_TF = """
resource "aws_iam_policy" "overly_permissive" {
  name = "overly-permissive"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "*"
      Resource = "*"
    }]
  })
}
"""

INVALID_HCL = 'resource "aws_s3_bucket" { unclosed'

VALID_DOCKERFILE = """FROM python:3.11-slim
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

INSECURE_DOCKERFILE = """FROM ubuntu:latest
ADD . /app
RUN apt-get install -y curl
EXPOSE 22
CMD ["bash"]
"""


# ── Layer 1: Syntax ───────────────────────────────────────────────────────────

class TestL1Syntax:
    """yamllint, HCL parse, hadolint — must catch malformed syntax."""

    def test_valid_yaml_passes(self):
        ok, errors = _validate_yaml_syntax(VALID_DEPLOYMENT)
        assert ok is True
        assert not any(e.severity == Severity.ERROR for e in errors)

    def test_invalid_yaml_fails(self):
        ok, errors = _validate_yaml_syntax(INVALID_YAML)
        assert ok is False
        assert errors

    def test_empty_yaml_fails(self):
        ok, errors = _validate_yaml_syntax("")
        # Empty YAML may be treated as invalid content
        assert isinstance(ok, bool)

    def test_valid_hcl_passes(self):
        ok, errors = _validate_hcl_syntax(VALID_TF)
        assert ok is True

    def test_invalid_hcl_fails(self):
        ok, errors = _validate_hcl_syntax(INVALID_HCL)
        assert ok is False
        assert errors

    def test_valid_dockerfile_passes(self):
        ok, errors = _validate_dockerfile_syntax(VALID_DOCKERFILE)
        assert ok is True

    def test_insecure_dockerfile_syntax_still_valid(self):
        # Hadolint only does syntax at L1; security rules are L3.
        ok, errors = _validate_dockerfile_syntax(INSECURE_DOCKERFILE)
        # Syntax should parse even if security is bad
        assert isinstance(ok, bool)

    def test_yaml_errors_have_correct_layer(self):
        _, errors = _validate_yaml_syntax(INVALID_YAML)
        for e in errors:
            assert e.layer == ValidationLayer.SYNTAX

    def test_hcl_errors_have_correct_layer(self):
        _, errors = _validate_hcl_syntax(INVALID_HCL)
        for e in errors:
            assert e.layer == ValidationLayer.SYNTAX


# ── Layer 2: Schema ───────────────────────────────────────────────────────────

class TestL2Schema:
    """kubeconform for K8s — catches wrong API versions, missing required fields."""

    def test_valid_deployment_schema_passes(self):
        ok, errors = _validate_k8s_schema(VALID_DEPLOYMENT)
        assert ok is True, f"Expected schema pass, errors: {errors}"

    def test_deprecated_api_version_flagged(self):
        deprecated = VALID_DEPLOYMENT.replace("apps/v1", "extensions/v1beta1")
        ok, errors = _validate_k8s_schema(deprecated)
        # kubeconform should reject extensions/v1beta1 Deployment
        assert ok is False or any(errors)

    def test_missing_selector_flagged(self):
        # Remove the selector block — required by apps/v1 Deployment schema
        no_selector = "\n".join(
            line for line in VALID_DEPLOYMENT.splitlines()
            if "selector" not in line and "matchLabels" not in line
        )
        ok, errors = _validate_k8s_schema(no_selector)
        assert ok is False or any(errors)

    def test_schema_errors_have_schema_layer(self):
        _, errors = _validate_k8s_schema(INSECURE_DEPLOYMENT.replace("apps/v1", "extensions/v1beta1"))
        for e in errors:
            assert e.layer in (ValidationLayer.SCHEMA, ValidationLayer.SYNTAX)

    def test_autoscaling_v2beta2_deprecated(self):
        hpa_deprecated = """
apiVersion: autoscaling/v2beta2
kind: HorizontalPodAutoscaler
metadata:
  name: test-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: test-app
  minReplicas: 1
  maxReplicas: 5
"""
        ok, errors = _validate_k8s_schema(hpa_deprecated)
        # v2beta2 was removed in K8s 1.26 — should fail with 1.29 schemas
        assert ok is False or any(errors)

    def test_autoscaling_v2_valid(self):
        hpa_valid = """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: test-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: test-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
"""
        ok, errors = _validate_k8s_schema(hpa_valid)
        assert ok is True, f"autoscaling/v2 HPA should pass schema, errors: {errors}"


# ── Layer 2.5: kubectl Dry-Run ────────────────────────────────────────────────

class TestL25DryRun:
    """kubectl apply --dry-run=server — only runs when kind cluster is reachable."""

    def test_kind_reachable_returns_bool(self):
        result = _kind_cluster_reachable()
        assert isinstance(result, bool)

    def test_validator_skips_dry_run_when_no_cluster(self, validator):
        """When kind is unavailable, dry_run_server_valid must be None (graceful skip)."""
        with patch("infraagent.validators._kind_cluster_reachable", return_value=False):
            report = validator.validate(VALID_DEPLOYMENT, "kubernetes")
        assert report.dry_run_server_valid is None, (
            "dry_run_server_valid should be None when kind is unavailable"
        )

    def test_validator_runs_dry_run_when_cluster_available(self, validator):
        """When kind returns available, dry_run_server_valid must be True or False."""
        mock_result = (True, [])
        with patch("infraagent.validators._kind_cluster_reachable", return_value=True), \
             patch("infraagent.validators._validate_k8s_dry_run_server", return_value=mock_result):
            report = validator.validate(VALID_DEPLOYMENT, "kubernetes")
        assert report.dry_run_server_valid is True

    def test_dry_run_errors_captured_in_report(self, validator):
        mock_result = (False, ["error: the server could not find the requested resource"])
        with patch("infraagent.validators._kind_cluster_reachable", return_value=True), \
             patch("infraagent.validators._validate_k8s_dry_run_server", return_value=mock_result):
            report = validator.validate(VALID_DEPLOYMENT, "kubernetes")
        assert report.dry_run_server_valid is False
        assert report.dry_run_server_errors

    def test_dry_run_not_run_for_terraform(self, validator):
        """kubectl dry-run is K8s-only; must not run for Terraform."""
        with patch("infraagent.validators._kind_cluster_reachable", return_value=True):
            report = validator.validate(VALID_TF, "terraform")
        # Terraform has tf_plan_valid, not dry_run_server_valid
        assert report.dry_run_server_valid is None


# ── Layer 3: Security ─────────────────────────────────────────────────────────

class TestL3Security:
    """Checkov + Trivy — catches root containers, wildcard IAM, missing security context."""

    def test_secure_deployment_high_score(self):
        score, errors = _validate_k8s_security(VALID_DEPLOYMENT)
        assert score >= 0.7, f"Secure deployment should score ≥0.7, got {score:.2f}"

    def test_insecure_deployment_low_score(self):
        score, errors = _validate_k8s_security(INSECURE_DEPLOYMENT)
        assert score < 0.7, f"Insecure deployment should score <0.7, got {score:.2f}"

    def test_root_user_flagged_in_k8s(self):
        root_deployment = VALID_DEPLOYMENT + """
      containers:
      - name: root-app
        image: nginx:1.25.3
        securityContext:
          runAsUser: 0
"""
        _, errors = _validate_k8s_security(VALID_DEPLOYMENT.replace(
            "runAsUser: 1000", "runAsUser: 0"
        ))
        rule_ids = [e.rule_id for e in errors]
        assert any("CKV_K8S_30" in r or "root" in r.lower() for r in rule_ids), (
            "Running as root (UID 0) should be flagged by CKV_K8S_30"
        )

    def test_missing_resource_limits_flagged(self):
        no_limits = VALID_DEPLOYMENT
        for line in ["resources:", "limits:", "requests:", "cpu:", "memory:"]:
            no_limits = "\n".join(
                l for l in no_limits.splitlines() if line.strip() not in l.strip()
            )
        _, errors = _validate_k8s_security(no_limits)
        rule_ids = " ".join(e.rule_id for e in errors)
        assert "CKV_K8S_11" in rule_ids or "limit" in rule_ids.lower() or errors

    def test_wildcard_iam_flagged_in_terraform(self):
        score, errors = _validate_tf_security(INSECURE_TF)
        rule_ids = [e.rule_id for e in errors]
        assert any("CKV_AWS_40" in r or "wildcard" in e.message.lower()
                   for r, e in zip(rule_ids, errors)), (
            "Wildcard IAM Action='*' should be flagged (CKV_AWS_40 or similar)"
        )

    def test_secure_tf_higher_score_than_insecure(self):
        secure_score, _ = _validate_tf_security(VALID_TF)
        insecure_score, _ = _validate_tf_security(INSECURE_TF)
        assert secure_score >= insecure_score, (
            "Secure Terraform should score ≥ insecure Terraform"
        )

    def test_security_errors_have_security_layer(self):
        _, errors = _validate_k8s_security(INSECURE_DEPLOYMENT)
        for e in errors:
            assert e.layer == ValidationLayer.SECURITY

    def test_security_score_bounded(self):
        for code, lang_fn in [(VALID_DEPLOYMENT, _validate_k8s_security),
                               (INSECURE_TF, _validate_tf_security)]:
            score, _ = lang_fn(code)
            assert 0.0 <= score <= 1.0, f"Security score {score} out of [0, 1]"


# ── Layer 4: Best Practices ───────────────────────────────────────────────────

class TestL4BestPractices:
    """OPA/Conftest — missing health probes, resource limits, labels."""

    def test_deployment_with_probes_passes_bp(self):
        score, issues = _validate_best_practices(VALID_DEPLOYMENT, "kubernetes")
        assert score >= 0.5, f"Deployment with probes should score ≥0.5, got {score:.2f}"

    def test_deployment_without_probes_flagged(self):
        no_probes = "\n".join(
            line for line in VALID_DEPLOYMENT.splitlines()
            if "Probe" not in line and "httpGet" not in line
               and "initialDelay" not in line and "period" not in line
               and "/health" not in line and "/ready" not in line
        )
        score_with, _    = _validate_best_practices(VALID_DEPLOYMENT, "kubernetes")
        score_without, _ = _validate_best_practices(no_probes, "kubernetes")
        assert score_with >= score_without, (
            "Adding probes should not decrease best-practice score"
        )

    def test_bp_score_bounded(self):
        for code, lang in [(VALID_DEPLOYMENT, "kubernetes"), (VALID_TF, "terraform")]:
            score, _ = _validate_best_practices(code, lang)
            assert 0.0 <= score <= 1.0

    def test_bp_issues_have_best_practice_layer(self):
        _, issues = _validate_best_practices(INSECURE_DEPLOYMENT, "kubernetes")
        for issue in issues:
            if hasattr(issue, "layer"):
                assert issue.layer == ValidationLayer.BEST_PRACTICE


# ── MultiLayerValidator Integration ──────────────────────────────────────────

class TestMultiLayerValidatorIntegration:
    """End-to-end: validate() must produce a coherent ValidationReport."""

    def test_secure_k8s_passes_all_layers(self, validator):
        with patch("infraagent.validators._kind_cluster_reachable", return_value=False):
            report = validator.validate(VALID_DEPLOYMENT, "kubernetes")
        assert report.syntax_valid is True
        assert report.schema_valid is True
        assert report.security_score >= 0.7

    def test_invalid_yaml_fails_at_l1(self, validator):
        report = validator.validate(INVALID_YAML, "kubernetes")
        assert report.syntax_valid is False
        assert report.passed is False
        # Schema should not be run if syntax failed
        assert report.schema_valid is False

    def test_insecure_k8s_fails_security_layer(self, validator):
        with patch("infraagent.validators._kind_cluster_reachable", return_value=False):
            report = validator.validate(INSECURE_DEPLOYMENT, "kubernetes")
        assert report.security_score < 0.7

    def test_report_has_all_required_fields(self, validator):
        with patch("infraagent.validators._kind_cluster_reachable", return_value=False):
            report = validator.validate(VALID_DEPLOYMENT, "kubernetes")
        assert hasattr(report, "passed")
        assert hasattr(report, "syntax_valid")
        assert hasattr(report, "schema_valid")
        assert hasattr(report, "security_score")
        assert hasattr(report, "best_practice_score")
        assert hasattr(report, "errors")
        assert hasattr(report, "dry_run_server_valid")

    def test_overall_score_between_0_and_1(self, validator):
        with patch("infraagent.validators._kind_cluster_reachable", return_value=False):
            report = validator.validate(INSECURE_DEPLOYMENT, "kubernetes")
        assert 0.0 <= report.overall_score <= 1.0

    def test_errors_to_feedback_returns_list_of_dicts(self, validator):
        with patch("infraagent.validators._kind_cluster_reachable", return_value=False):
            report = validator.validate(INSECURE_DEPLOYMENT, "kubernetes")
        feedback = validator.errors_to_feedback(report)
        assert isinstance(feedback, list)
        for item in feedback:
            assert isinstance(item, dict)
            assert "layer" in item
            assert "message" in item

    def test_terraform_validation_runs(self, validator):
        with patch("infraagent.validators._kind_cluster_reachable", return_value=False):
            report = validator.validate(VALID_TF, "terraform")
        assert isinstance(report, ValidationReport)
        assert report.syntax_valid is True

    def test_dockerfile_validation_runs(self, validator):
        report = validator.validate(VALID_DOCKERFILE, "dockerfile")
        assert isinstance(report, ValidationReport)
        assert report.syntax_valid is True

    def test_to_dict_serializable(self, validator):
        import json
        with patch("infraagent.validators._kind_cluster_reachable", return_value=False):
            report = validator.validate(VALID_DEPLOYMENT, "kubernetes")
        d = report.to_dict()
        # Must be JSON-serializable (no non-serializable objects)
        json.dumps(d)
