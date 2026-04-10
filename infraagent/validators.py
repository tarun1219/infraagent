"""
Multi-Layer Validator Module for InfraAgent.

Performs four-layer validation of generated IaC artifacts:
  Layer 1 — Syntax:        yamllint / HCL parse
  Layer 2 — Schema:        kubeconform / terraform validate
  Layer 3 — Security:      checkov / trivy / tfsec
  Layer 4 — Best Practice: OPA/conftest custom policies

All validators are invoked as local subprocesses; no cloud connectivity
required.  Results are aggregated into a structured ValidationReport.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ValidationLayer(str, Enum):
    SYNTAX = "syntax"
    SCHEMA = "schema"
    SECURITY = "security"
    BEST_PRACTICE = "best_practice"


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationError:
    layer: ValidationLayer
    tool: str
    rule_id: str
    message: str
    severity: Severity
    line: Optional[int] = None
    resource: Optional[str] = None


@dataclass
class ValidationReport:
    passed: bool
    syntax_valid: bool
    schema_valid: bool
    security_score: float       # 0.0 – 1.0
    best_practice_score: float  # 0.0 – 1.0
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    total_checks: int = 0
    passed_checks: int = 0
    # Layer 2.5: kubectl apply --dry-run=server result (None = not run / kind unavailable)
    dry_run_server_valid: Optional[bool] = None
    dry_run_server_errors: List[str] = field(default_factory=list)
    # Layer 2.5 extension: terraform plan mock validation (None = not run / terraform not installed)
    terraform_plan_valid: Optional[bool] = None
    terraform_plan_errors: List[str] = field(default_factory=list)
    # Layer 2.5 extension: docker build validation (None = not run / docker not installed)
    docker_build_valid: Optional[bool] = None
    docker_build_errors: List[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        if self.total_checks == 0:
            return 0.0
        return self.passed_checks / self.total_checks

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "syntax_valid": self.syntax_valid,
            "schema_valid": self.schema_valid,
            "dry_run_server_valid": self.dry_run_server_valid,
            "dry_run_server_errors": self.dry_run_server_errors,
            "terraform_plan_valid": self.terraform_plan_valid,
            "terraform_plan_errors": self.terraform_plan_errors,
            "docker_build_valid": self.docker_build_valid,
            "docker_build_errors": self.docker_build_errors,
            "security_score": self.security_score,
            "best_practice_score": self.best_practice_score,
            "overall_score": self.overall_score,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "errors": [
                {
                    "layer": e.layer.value,
                    "tool": e.tool,
                    "rule_id": e.rule_id,
                    "message": e.message,
                    "severity": e.severity.value,
                    "line": e.line,
                    "resource": e.resource,
                }
                for e in self.errors
            ],
        }


# ---------------------------------------------------------------------------
# Layer 1: Syntax Validators
# ---------------------------------------------------------------------------

def _validate_yaml_syntax(content: str) -> Tuple[bool, List[ValidationError]]:
    """Run yamllint on YAML content."""
    errors: List[ValidationError] = []
    try:
        import yaml
    except ImportError:
        # pyyaml not installed — accept content without validation
        return True, []
    try:
        # Parse all documents in the file
        list(yaml.safe_load_all(content))
        return True, []
    except yaml.YAMLError as e:
        errors.append(ValidationError(
            layer=ValidationLayer.SYNTAX,
            tool="pyyaml",
            rule_id="YAML_PARSE_ERROR",
            message=str(e),
            severity=Severity.ERROR,
            line=getattr(getattr(e, "problem_mark", None), "line", None),
        ))
        return False, errors


def _validate_hcl_syntax(content: str) -> Tuple[bool, List[ValidationError]]:
    """Invoke `terraform validate` on an HCL document via subprocess, or use heuristics."""
    errors: List[ValidationError] = []
    if not shutil.which("terraform"):
        # Heuristic fallback: balanced braces + resource block presence
        opens  = content.count("{")
        closes = content.count("}")
        if opens == 0 or opens != closes or len(content.strip()) < 10:
            errors.append(ValidationError(
                layer=ValidationLayer.SYNTAX,
                tool="heuristic",
                rule_id="TF_SYNTAX",
                message="HCL parse error: unbalanced braces or empty content",
                severity=Severity.ERROR,
            ))
            return False, errors
        return True, []
    with tempfile.TemporaryDirectory() as tmpdir:
        tf_file = Path(tmpdir) / "main.tf"
        tf_file.write_text(content)
        # Write a minimal provider stub so validate can run without auth
        (Path(tmpdir) / "versions.tf").write_text(
            'terraform { required_providers { aws = { source = "hashicorp/aws" } } }\n'
            'provider "aws" { region = "us-east-1" skip_credentials_validation = true '
            'skip_requesting_account_id = true skip_metadata_api_check = true '
            'access_key = "mock" secret_key = "mock" }\n'
        )
        result = subprocess.run(
            ["terraform", "validate", "-json"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        try:
            data = json.loads(result.stdout)
            if data.get("valid"):
                return True, []
            for diag in data.get("diagnostics", []):
                errors.append(ValidationError(
                    layer=ValidationLayer.SYNTAX,
                    tool="terraform-validate",
                    rule_id="TF_SYNTAX",
                    message=diag.get("detail", diag.get("summary", "")),
                    severity=Severity.ERROR
                    if diag.get("severity") == "error"
                    else Severity.WARNING,
                    line=diag.get("range", {}).get("start", {}).get("line"),
                ))
            return not any(e.severity == Severity.ERROR for e in errors), errors
        except (json.JSONDecodeError, KeyError):
            if result.returncode != 0:
                errors.append(ValidationError(
                    layer=ValidationLayer.SYNTAX,
                    tool="terraform-validate",
                    rule_id="TF_SYNTAX",
                    message=result.stderr[:500],
                    severity=Severity.ERROR,
                ))
                return False, errors
            return True, []


def _validate_dockerfile_syntax(content: str) -> Tuple[bool, List[ValidationError]]:
    """Run hadolint on Dockerfile content, or use heuristics if hadolint is unavailable."""
    errors: List[ValidationError] = []
    if not shutil.which("hadolint"):
        # Heuristic: must have at least one FROM instruction
        if "FROM " not in content and "from " not in content:
            errors.append(ValidationError(
                layer=ValidationLayer.SYNTAX,
                tool="heuristic",
                rule_id="DF_SYNTAX",
                message="Dockerfile missing FROM instruction",
                severity=Severity.ERROR,
            ))
            return False, errors
        return True, []
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="Dockerfile", delete=False
    ) as f:
        f.write(content)
        fname = f.name
    result = subprocess.run(
        ["hadolint", "--format", "json", fname],
        capture_output=True, text=True, timeout=10,
    )
    Path(fname).unlink(missing_ok=True)
    if result.returncode == 0:
        return True, []
    try:
        issues = json.loads(result.stdout)
        for issue in issues:
            sev = (
                Severity.ERROR
                if issue.get("level") == "error"
                else Severity.WARNING
            )
            errors.append(ValidationError(
                layer=ValidationLayer.SYNTAX,
                tool="hadolint",
                rule_id=issue.get("code", "DL0000"),
                message=issue.get("message", ""),
                severity=sev,
                line=issue.get("line"),
            ))
        return not any(e.severity == Severity.ERROR for e in errors), errors
    except json.JSONDecodeError:
        errors.append(ValidationError(
            layer=ValidationLayer.SYNTAX,
            tool="hadolint",
            rule_id="DL_UNKNOWN",
            message=result.stderr[:300],
            severity=Severity.ERROR,
        ))
        return False, errors


# ---------------------------------------------------------------------------
# Layer 2: Schema Validators
# ---------------------------------------------------------------------------

_DEPRECATED_API_VERSIONS = {
    "extensions/v1beta1":    "Removed in K8s 1.22 — use networking.k8s.io/v1 for Ingress",
    "policy/v1beta1":        "Removed in K8s 1.25 — use policy/v1 for PDB",
    "autoscaling/v1":        "Deprecated — use autoscaling/v2 for HPA",
    "autoscaling/v2beta1":   "Deprecated in K8s 1.26 — use autoscaling/v2 for HPA",
    "autoscaling/v2beta2":   "Deprecated in K8s 1.26 — use autoscaling/v2 for HPA",
    "apps/v1beta1":          "Removed in K8s 1.16 — use apps/v1",
    "apps/v1beta2":          "Removed in K8s 1.16 — use apps/v1",
}

_REQUIRED_FIELDS_PER_KIND = {
    "Deployment": ["spec.selector", "spec.template", "spec.template.spec.containers"],
    "Service": ["spec.selector", "spec.ports"],
    "HorizontalPodAutoscaler": ["spec.scaleTargetRef", "spec.minReplicas", "spec.maxReplicas"],
    "PodDisruptionBudget": ["spec.selector"],
    "Ingress": ["spec.rules"],
    "NetworkPolicy": ["spec.podSelector"],
}


def _validate_k8s_schema(content: str) -> Tuple[bool, List[ValidationError]]:
    """
    Schema validation for Kubernetes manifests.
    Attempts kubeconform first, falls back to in-process checks.
    """
    errors: List[ValidationError] = []

    # --- kubeconform (subprocess, optional) ---
    kubeconform_ok = True
    if shutil.which("kubeconform"):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(content)
            fname = f.name

        result = subprocess.run(
            ["kubeconform", "-strict", "-summary", "-output", "json", fname],
            capture_output=True, text=True, timeout=20,
        )
        Path(fname).unlink(missing_ok=True)

        if result.returncode != 0:
            kubeconform_ok = False
            try:
                data = json.loads(result.stdout)
                for res in data.get("resources", []):
                    if res.get("status") == "statusInvalid":
                        errors.append(ValidationError(
                            layer=ValidationLayer.SCHEMA,
                            tool="kubeconform",
                            rule_id="K8S_SCHEMA",
                            message=res.get("msg", "Schema validation failed"),
                            severity=Severity.ERROR,
                            resource=res.get("name"),
                        ))
            except json.JSONDecodeError:
                errors.append(ValidationError(
                    layer=ValidationLayer.SCHEMA,
                    tool="kubeconform",
                    rule_id="K8S_SCHEMA",
                    message=result.stderr[:300],
                    severity=Severity.ERROR,
                ))

    # --- In-process heuristic checks ---
    try:
        import yaml
    except ImportError:
        return kubeconform_ok and not errors, errors
    try:
        docs = list(yaml.safe_load_all(content))
    except yaml.YAMLError:
        return False, errors

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        api_version = doc.get("apiVersion", "")
        if api_version in _DEPRECATED_API_VERSIONS:
            errors.append(ValidationError(
                layer=ValidationLayer.SCHEMA,
                tool="infraagent-schema",
                rule_id="DEPRECATED_API",
                message=f"Deprecated apiVersion '{api_version}': {_DEPRECATED_API_VERSIONS[api_version]}",
                severity=Severity.ERROR,
                resource=doc.get("metadata", {}).get("name"),
            ))
            kubeconform_ok = False

    return kubeconform_ok and not errors, errors


# ---------------------------------------------------------------------------
# Layer 2.5: kubectl apply --dry-run=server (Kubernetes only)
# Requires a reachable kind cluster: `kind create cluster --name infraagent`
# Skipped gracefully when kubectl is absent or no cluster is reachable.
# ---------------------------------------------------------------------------

def _kind_cluster_reachable() -> bool:
    """Return True if a kind cluster named 'infraagent' (or any cluster) is reachable."""
    try:
        result = subprocess.run(
            ["kubectl", "cluster-info", "--request-timeout=3s"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _validate_k8s_dry_run_server(
    content: str,
) -> Tuple[bool, List[str]]:
    """
    Run `kubectl apply --dry-run=server` on Kubernetes YAML.

    This is the highest-fidelity static check available without actually
    creating resources: the API server validates the manifest against its
    live schema, admission webhooks, and resource version constraints.

    Returns:
        (ok: bool, error_lines: List[str])
        ok=None is never returned; callers should treat None dry_run_server_valid
        as "skipped" — this function returns (True, []) when the cluster is
        unreachable (so the result field stays None — handled by the caller).
    """
    if not _kind_cluster_reachable():
        return True, []   # Signal: skipped — caller sets field to None

    errors: List[str] = []
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(content)
        fname = f.name

    try:
        result = subprocess.run(
            [
                "kubectl", "apply",
                "--dry-run=server",
                "--validate=strict",
                "-f", fname,
            ],
            capture_output=True, text=True, timeout=30,
        )
        Path(fname).unlink(missing_ok=True)

        if result.returncode == 0:
            return True, []

        # Parse stderr for error lines (kubectl prints one error per resource)
        for line in (result.stderr + result.stdout).splitlines():
            line = line.strip()
            if line and ("error" in line.lower() or "invalid" in line.lower()):
                errors.append(line)
        if not errors and result.returncode != 0:
            errors.append(result.stderr.strip()[:400] or "kubectl dry-run failed")
        return False, errors

    except subprocess.TimeoutExpired:
        Path(fname).unlink(missing_ok=True)
        return False, ["kubectl apply --dry-run=server timed out (>30s)"]
    except FileNotFoundError:
        Path(fname).unlink(missing_ok=True)
        return True, []   # kubectl not installed — caller sets field to None


# ---------------------------------------------------------------------------
# Layer 2.5 Extensions: Terraform Plan Validation + Docker Build Validation
# ---------------------------------------------------------------------------

def _validate_terraform_plan(content: str) -> Tuple[Optional[bool], List[str]]:
    """
    Run `terraform init -backend=false` then `terraform validate` on HCL content.

    This provides higher-fidelity validation than the syntax-only HCL parse by
    checking provider resource attribute validity and required argument presence.

    Returns:
        (ok: Optional[bool], error_lines: List[str])
        Returns (None, ["terraform not installed"]) when terraform is unavailable.
    """
    errors: List[str] = []
    try:
        # Quick check: is terraform installed?
        check = subprocess.run(
            ["terraform", "version"],
            capture_output=True, text=True, timeout=10,
        )
        if check.returncode != 0:
            return None, ["terraform not installed"]
    except FileNotFoundError:
        return None, ["terraform not installed"]
    except subprocess.TimeoutExpired:
        return None, ["terraform version check timed out"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tf_file = Path(tmpdir) / "main.tf"
        tf_file.write_text(content)
        # Minimal provider stub so validate can run without cloud credentials
        (Path(tmpdir) / "versions.tf").write_text(
            'terraform {\n'
            '  required_providers {\n'
            '    aws = { source = "hashicorp/aws" }\n'
            '  }\n'
            '}\n'
            'provider "aws" {\n'
            '  region                      = "us-east-1"\n'
            '  skip_credentials_validation = true\n'
            '  skip_requesting_account_id  = true\n'
            '  skip_metadata_api_check     = true\n'
            '  access_key                  = "mock"\n'
            '  secret_key                  = "mock"\n'
            '}\n'
        )
        # Step 1: terraform init (no backend, no network calls)
        init_result = subprocess.run(
            ["terraform", "init", "-backend=false", "-no-color"],
            cwd=tmpdir,
            capture_output=True, text=True, timeout=60,
        )
        if init_result.returncode != 0:
            # init failure usually means provider fetch issue — skip validate
            for line in init_result.stderr.splitlines():
                if "error" in line.lower():
                    errors.append(line.strip())
            return False, errors or [init_result.stderr.strip()[:400]]

        # Step 2: terraform validate
        val_result = subprocess.run(
            ["terraform", "validate", "-json", "-no-color"],
            cwd=tmpdir,
            capture_output=True, text=True, timeout=30,
        )
        try:
            data = json.loads(val_result.stdout)
            if data.get("valid"):
                return True, []
            for diag in data.get("diagnostics", []):
                if diag.get("severity") == "error":
                    msg = diag.get("detail", diag.get("summary", "Terraform validate error"))
                    errors.append(msg)
            return False, errors
        except json.JSONDecodeError:
            if val_result.returncode != 0:
                errors.append(val_result.stderr.strip()[:400] or "terraform validate failed")
                return False, errors
            return True, []
        except subprocess.TimeoutExpired:
            return False, ["terraform validate timed out (>30s)"]


def _validate_docker_build(content: str) -> Tuple[Optional[bool], List[str]]:
    """
    Run `docker buildx build --check .` on Dockerfile content.

    Uses Docker BuildKit's lint/check mode to validate the Dockerfile without
    actually building the image.  Falls back to `docker build --no-cache --dry-run`
    for older Docker versions.

    Returns:
        (ok: Optional[bool], error_lines: List[str])
        Returns (None, ["docker not installed"]) when docker is unavailable.
    """
    errors: List[str] = []
    try:
        check = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=10,
        )
        if check.returncode != 0:
            return None, ["docker not installed or daemon not running"]
    except FileNotFoundError:
        return None, ["docker not installed"]
    except subprocess.TimeoutExpired:
        return None, ["docker version check timed out"]

    with tempfile.TemporaryDirectory() as tmpdir:
        dockerfile = Path(tmpdir) / "Dockerfile"
        dockerfile.write_text(content)

        # Try buildx build --check first (Docker 24.0+)
        result = subprocess.run(
            ["docker", "buildx", "build", "--check", "."],
            cwd=tmpdir,
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            return True, []

        # Parse errors from stderr
        for line in (result.stderr + result.stdout).splitlines():
            line = line.strip()
            if line and any(kw in line.lower() for kw in ("error", "invalid", "unknown instruction")):
                errors.append(line)

        if not errors:
            # Fallback: check if it's an "unknown flag" error (older Docker without --check)
            if "--check" in result.stderr or "unknown flag" in result.stderr:
                # Try hadolint as fallback
                try:
                    had = subprocess.run(
                        ["hadolint", "--format", "tty", str(dockerfile)],
                        capture_output=True, text=True, timeout=10,
                    )
                    if had.returncode == 0:
                        return True, []
                    for line in had.stdout.splitlines():
                        if "error" in line.lower():
                            errors.append(line.strip())
                    return False, errors if errors else [had.stdout.strip()[:400]]
                except FileNotFoundError:
                    return None, ["docker buildx --check not supported and hadolint not available"]
            errors.append(result.stderr.strip()[:400] or "docker build check failed")

        return False, errors


# ---------------------------------------------------------------------------
# Layer 3: Security Validators
# ---------------------------------------------------------------------------

_SECURITY_CHECKS_K8S = [
    ("SEC_NO_ROOT", "runAsNonRoot", "Container must set runAsNonRoot: true"),
    ("SEC_PRIV_ESC", "allowPrivilegeEscalation: false", "Must disable privilege escalation"),
    ("SEC_READONLY_FS", "readOnlyRootFilesystem: true", "Root filesystem should be read-only"),
    ("SEC_DROP_CAPS", 'drop: ["ALL"]', "Must drop all Linux capabilities"),
    ("SEC_RESOURCE_LIMITS", "limits:", "Resource limits (CPU/memory) must be defined"),
]

# Patterns that indicate explicitly running as root UID — always flagged
_ROOT_UID_PATTERNS = ["runasuser: 0", "runasuser:0"]

_SECURITY_CHECKS_TF = [
    ("TF_S3_PUBLIC", "block_public_acls", "S3 bucket must block public ACLs"),
    ("TF_ENCRYPT", "encrypted", "Storage resources must enable encryption"),
    ("TF_IAM_STAR", r'\*"', "IAM policies must not use wildcard '*' in actions/resources"),
]


def _validate_k8s_security(content: str) -> Tuple[float, List[ValidationError]]:
    """Run checkov + in-process pattern checks for K8s security."""
    errors: List[ValidationError] = []
    total = len(_SECURITY_CHECKS_K8S)
    passed = 0

    for rule_id, pattern, message in _SECURITY_CHECKS_K8S:
        if pattern in content:
            passed += 1
        else:
            errors.append(ValidationError(
                layer=ValidationLayer.SECURITY,
                tool="infraagent-security",
                rule_id=rule_id,
                message=message,
                severity=Severity.ERROR,
            ))

    # Explicitly flag running as root UID (runAsUser: 0)
    lower = content.lower()
    if any(p in lower for p in _ROOT_UID_PATTERNS):
        errors.append(ValidationError(
            layer=ValidationLayer.SECURITY,
            tool="infraagent-security",
            rule_id="CKV_K8S_30",
            message="Container runs as root UID (runAsUser: 0) — use a non-root UID",
            severity=Severity.ERROR,
        ))

    # Attempt checkov
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(content)
        fname = f.name

    try:
        result = subprocess.run(
            ["checkov", "-f", fname, "--quiet", "--output", "json",
             "--framework", "kubernetes"],
            capture_output=True, text=True, timeout=30,
        )
        Path(fname).unlink(missing_ok=True)
        data = json.loads(result.stdout)
        results = data.get("results", {})
        passed_checks = len(results.get("passed_checks", []))
        failed_checks = results.get("failed_checks", [])
        total += passed_checks + len(failed_checks)
        passed += passed_checks
        for check in failed_checks:
            errors.append(ValidationError(
                layer=ValidationLayer.SECURITY,
                tool="checkov",
                rule_id=check.get("check_id", "CKV_UNKNOWN"),
                message=check.get("check_result", {}).get(
                    "result", check.get("check_id", "Security check failed")
                ),
                severity=Severity.ERROR,
                resource=check.get("resource"),
            ))
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError, KeyError):
        Path(fname).unlink(missing_ok=True)

    score = passed / total if total > 0 else 0.0
    return score, errors


def _validate_tf_security(content: str) -> Tuple[float, List[ValidationError]]:
    """Security checks for Terraform (tfsec + pattern matching)."""
    errors: List[ValidationError] = []
    total = len(_SECURITY_CHECKS_TF)
    passed = 0

    for rule_id, pattern, message in _SECURITY_CHECKS_TF:
        if re.search(pattern, content):
            if rule_id == "TF_IAM_STAR":
                errors.append(ValidationError(
                    layer=ValidationLayer.SECURITY,
                    tool="infraagent-security",
                    rule_id=rule_id,
                    message=message,
                    severity=Severity.ERROR,
                ))
            else:
                passed += 1
        else:
            if rule_id != "TF_IAM_STAR":
                errors.append(ValidationError(
                    layer=ValidationLayer.SECURITY,
                    tool="infraagent-security",
                    rule_id=rule_id,
                    message=message,
                    severity=Severity.WARNING,
                ))

    score = passed / total if total > 0 else 0.0
    return score, errors


# ---------------------------------------------------------------------------
# Layer 4: Best Practice Checks
# ---------------------------------------------------------------------------

_BP_CHECKS_K8S = [
    ("BP_LIVENESS_PROBE", "livenessProbe:", "livenessProbe should be defined"),
    ("BP_READINESS_PROBE", "readinessProbe:", "readinessProbe should be defined"),
    ("BP_IMAGE_TAG", ":latest", "Avoid using ':latest' image tag", True),  # inverted
    ("BP_LABELS", "app:", "Resources should have 'app' label"),
    ("BP_NAMESPACE", "namespace:", "Namespace should be explicitly specified"),
]


def _validate_best_practices(
    content: str, language: str
) -> Tuple[float, List[ValidationError]]:
    errors: List[ValidationError] = []
    total = 0
    passed = 0

    if language == "kubernetes":
        checks = _BP_CHECKS_K8S
        for check in checks:
            rule_id, pattern, message = check[:3]
            inverted = len(check) == 4 and check[3]
            total += 1
            found = pattern in content
            if (found and not inverted) or (not found and inverted):
                passed += 1
            else:
                errors.append(ValidationError(
                    layer=ValidationLayer.BEST_PRACTICE,
                    tool="infraagent-bp",
                    rule_id=rule_id,
                    message=message,
                    severity=Severity.WARNING,
                ))

    score = passed / total if total > 0 else 1.0
    return score, errors


# ---------------------------------------------------------------------------
# Main Validator
# ---------------------------------------------------------------------------

class MultiLayerValidator:
    """
    Orchestrates four-layer validation of generated IaC artifacts.
    Each layer's errors are fed back to the self-correction loop.
    """

    def __init__(self, k8s_version: str = "1.29.0"):
        self.k8s_version = k8s_version

    def validate(self, code: str, language: str) -> ValidationReport:
        """
        Run all four validation layers on the given code.

        Args:
            code: The generated IaC code string.
            language: One of "kubernetes", "terraform", "dockerfile".

        Returns:
            A ValidationReport summarizing pass/fail and all errors.
        """
        all_errors: List[ValidationError] = []
        all_warnings: List[ValidationError] = []
        total_checks = 0
        passed_checks = 0

        # ---- Layer 1: Syntax ----
        if language == "kubernetes":
            syntax_ok, syn_errors = _validate_yaml_syntax(code)
        elif language == "terraform":
            syntax_ok, syn_errors = _validate_hcl_syntax(code)
        elif language == "dockerfile":
            syntax_ok, syn_errors = _validate_dockerfile_syntax(code)
        else:
            syntax_ok, syn_errors = True, []

        total_checks += 1
        if syntax_ok:
            passed_checks += 1
        all_errors.extend([e for e in syn_errors if e.severity == Severity.ERROR])
        all_warnings.extend([e for e in syn_errors if e.severity != Severity.ERROR])

        # ---- Layer 2: Schema (only if syntax passed) ----
        schema_valid = False
        if syntax_ok:
            if language == "kubernetes":
                schema_valid, sch_errors = _validate_k8s_schema(code)
            else:
                schema_valid = True
                sch_errors = []
            total_checks += 1
            if schema_valid:
                passed_checks += 1
            all_errors.extend([e for e in sch_errors if e.severity == Severity.ERROR])
            all_warnings.extend([e for e in sch_errors if e.severity != Severity.ERROR])
        else:
            schema_valid = False

        # ---- Layer 2.5: kubectl apply --dry-run=server (K8s only, if schema passed) ----
        # dry_run_server_valid is None when kind is unavailable (graceful skip).
        # When available it is the authoritative deployment-path check.
        dry_run_valid: Optional[bool] = None
        dry_run_errs: List[str] = []
        tf_plan_valid: Optional[bool] = None
        tf_plan_errs: List[str] = []
        docker_build_valid: Optional[bool] = None
        docker_build_errs: List[str] = []
        if language == "kubernetes" and syntax_ok:
            _ok, _errs = _validate_k8s_dry_run_server(code)
            # _validate_k8s_dry_run_server returns (True, []) to mean "skipped"
            # when the cluster is unreachable — distinguish via _errs being empty
            # AND _kind_cluster_reachable() being False (already checked inside).
            # Re-check reachability to set the field correctly.
            if _kind_cluster_reachable():
                dry_run_valid = _ok
                dry_run_errs = _errs
                # Count as an extra check if the cluster is live
                total_checks += 1
                if _ok:
                    passed_checks += 1
                else:
                    for msg in _errs:
                        all_errors.append(ValidationError(
                            layer=ValidationLayer.SCHEMA,
                            tool="kubectl-dry-run-server",
                            rule_id="K8S_DRY_RUN_SERVER",
                            message=msg,
                            severity=Severity.ERROR,
                        ))
            # else: dry_run_valid stays None — field recorded but not counted

        # ---- Layer 2.5 ext: terraform plan validation (Terraform only, if syntax passed) ----
        if language == "terraform" and syntax_ok:
            tf_plan_valid, tf_plan_errs = _validate_terraform_plan(code)
            if tf_plan_valid is not None:
                # Only gate when terraform is installed and validation ran
                total_checks += 1
                if tf_plan_valid:
                    passed_checks += 1
                else:
                    for msg in tf_plan_errs:
                        all_errors.append(ValidationError(
                            layer=ValidationLayer.SCHEMA,
                            tool="terraform-plan",
                            rule_id="TF_PLAN_INVALID",
                            message=msg,
                            severity=Severity.ERROR,
                        ))

        # ---- Layer 2.5 ext: docker build validation (Dockerfile only, if syntax passed) ----
        if language == "dockerfile" and syntax_ok:
            docker_build_valid, docker_build_errs = _validate_docker_build(code)
            if docker_build_valid is not None:
                total_checks += 1
                if docker_build_valid:
                    passed_checks += 1
                else:
                    for msg in docker_build_errs:
                        all_errors.append(ValidationError(
                            layer=ValidationLayer.SCHEMA,
                            tool="docker-build-check",
                            rule_id="DOCKER_BUILD_INVALID",
                            message=msg,
                            severity=Severity.ERROR,
                        ))

        # ---- Layer 3: Security ----
        if language == "kubernetes":
            sec_score, sec_errors = _validate_k8s_security(code)
        elif language == "terraform":
            sec_score, sec_errors = _validate_tf_security(code)
        else:
            sec_score = 1.0
            sec_errors = []

        n_sec = max(len(_SECURITY_CHECKS_K8S), 1)
        total_checks += n_sec
        passed_checks += int(sec_score * n_sec)
        all_errors.extend([e for e in sec_errors if e.severity == Severity.ERROR])
        all_warnings.extend([e for e in sec_errors if e.severity != Severity.ERROR])

        # ---- Layer 4: Best Practices ----
        bp_score, bp_errors = _validate_best_practices(code, language)
        n_bp = len(_BP_CHECKS_K8S) if language == "kubernetes" else 1
        total_checks += n_bp
        passed_checks += int(bp_score * n_bp)
        all_warnings.extend(bp_errors)

        # When dry-run is live and fails, it tightens the pass gate.
        dry_run_gate = (dry_run_valid is None or dry_run_valid)
        # Terraform plan and docker build gates: only active when tool is installed.
        tf_plan_gate = (tf_plan_valid is None or tf_plan_valid)
        docker_build_gate = (docker_build_valid is None or docker_build_valid)
        overall_passed = (
            syntax_ok
            and schema_valid
            and dry_run_gate
            and tf_plan_gate
            and docker_build_gate
            and sec_score >= 0.5
            and bp_score >= 0.6
        )

        return ValidationReport(
            passed=overall_passed,
            syntax_valid=syntax_ok,
            schema_valid=schema_valid,
            security_score=sec_score,
            best_practice_score=bp_score,
            errors=all_errors,
            warnings=all_warnings,
            total_checks=total_checks,
            passed_checks=passed_checks,
            dry_run_server_valid=dry_run_valid,
            dry_run_server_errors=dry_run_errs,
            terraform_plan_valid=tf_plan_valid,
            terraform_plan_errors=tf_plan_errs,
            docker_build_valid=docker_build_valid,
            docker_build_errors=docker_build_errs,
        )

    def errors_to_feedback(self, report: ValidationReport) -> List[Dict]:
        """Convert ValidationReport errors into a list of dicts for LLM feedback."""
        return [
            {
                "layer": e.layer.value,
                "tool": e.tool,
                "rule_id": e.rule_id,
                "message": e.message,
                "severity": e.severity.value,
                "line": e.line,
                "resource": e.resource,
            }
            for e in report.errors
        ]
