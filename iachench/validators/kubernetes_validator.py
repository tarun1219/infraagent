"""Kubernetes manifest validation: pyyaml syntax → heuristic schema → kubectl dry-run=server."""
from __future__ import annotations
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional, List


def validate_syntax(content: str) -> Tuple[bool, List[str]]:
    """Layer 1: YAML syntax validation via pyyaml (falls back gracefully without yamllint)."""
    # Try yamllint first if available
    if shutil.which("yamllint"):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            fname = f.name
        result = subprocess.run(
            ["yamllint", "-d", "{extends: default, rules: {line-length: disable}}", fname],
            capture_output=True, text=True,
        )
        Path(fname).unlink(missing_ok=True)
        return result.returncode == 0, result.stdout.splitlines()

    # Fallback: pyyaml
    try:
        import yaml
        list(yaml.safe_load_all(content))
        return True, []
    except Exception as e:
        return False, [str(e)]


def validate_schema(content: str, k8s_version: str = "1.28.0") -> Tuple[bool, List[str]]:
    """Layer 2: Schema validation via kubeconform if available, otherwise heuristic."""
    if shutil.which("kubeconform"):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            fname = f.name
        result = subprocess.run(
            ["kubeconform", "-strict", "-kubernetes-version", k8s_version, fname],
            capture_output=True, text=True,
        )
        Path(fname).unlink(missing_ok=True)
        return result.returncode == 0, result.stderr.splitlines()

    # Heuristic: check for required K8s fields
    errors = []
    lower = content.lower()
    if "extensions/v1beta1" in content:
        errors.append("extensions/v1beta1 is deprecated — migrate to apps/v1")
    if "apiversion:" not in lower:
        errors.append("Missing apiVersion field")
    if "kind:" not in lower:
        errors.append("Missing kind field")
    # Deployment-specific checks
    if "kind: deployment" in lower:
        if "matchlabels:" not in lower and "matchlabels:" not in content.lower():
            errors.append("Deployment missing spec.selector.matchLabels")
    schema_ok = len(errors) == 0
    return schema_ok, errors


def validate_dry_run_server(content: str) -> Tuple[Optional[bool], List[str]]:
    """
    Layer 2.5: Server-side dry-run via kubectl apply --dry-run=server.

    Returns (None, reason) if no kind cluster is reachable — degrades gracefully.
    Requires: kind create cluster --name infraagent (see scripts/setup_kind_cluster.sh)
    """
    ping = subprocess.run(
        ["kubectl", "cluster-info", "--request-timeout=3s"],
        capture_output=True, text=True,
    )
    if ping.returncode != 0:
        return None, ["No kind cluster reachable; Layer 2.5 skipped (run setup_kind_cluster.sh)"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        fname = f.name
    result = subprocess.run(
        ["kubectl", "apply", "--dry-run=server", "--validate=strict", "-f", fname],
        capture_output=True, text=True,
    )
    Path(fname).unlink(missing_ok=True)
    return result.returncode == 0, result.stderr.splitlines()


class KubernetesValidator:
    """
    Class wrapper around module-level validation functions.
    Returns a dict with keys: syntax_valid, schema_valid, security_valid, errors, warnings.
    """

    def validate(self, content: str) -> dict:
        errors: list = []
        warnings: list = []

        # L1 Syntax
        syntax_ok, syntax_errs = validate_syntax(content)
        errors.extend(syntax_errs)

        # L2 Schema
        schema_ok = False
        if syntax_ok:
            schema_ok, schema_errs = validate_schema(content)
            errors.extend(schema_errs)

        # Heuristic security checks (no external tools required)
        security_issues = _heuristic_security(content)
        warnings.extend(security_issues)
        security_valid = len(security_issues) == 0

        return {
            "syntax_valid":   syntax_ok,
            "schema_valid":   schema_ok,
            "security_valid": security_valid,
            "errors":         errors,
            "warnings":       warnings,
        }


def _heuristic_security(content: str) -> list:
    """Quick keyword-based security checks — no external tools required."""
    issues = []
    lower = content.lower()
    if "extensions/v1beta1" in content:
        issues.append("extensions/v1beta1 is deprecated — use apps/v1")
    if "runasuserroot" in lower or "runasuser: 0" in lower or "runasuser:0" in lower:
        issues.append("CKV_K8S_30: Containers must not run as root (runAsUser: 0)")
    if "runasnonroot" not in lower:
        issues.append("CKV_K8S_30: runAsNonRoot not set — container may run as root")
    if "limits:" not in lower:
        issues.append("CKV_K8S_11: CPU resource limits not set")
    return issues
