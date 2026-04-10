"""Dockerfile validation: hadolint (syntax/lint) + trivy (security)."""
from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, List


def validate_syntax(content: str) -> Tuple[bool, List[str]]:
    """Layer 1: Dockerfile linting via hadolint (error-level only)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dockerfile", delete=False) as f:
        f.write(content)
        fname = f.name
    result = subprocess.run(
        ["hadolint", "--format", "tty", "--failure-threshold", "error", fname],
        capture_output=True, text=True,
    )
    Path(fname).unlink(missing_ok=True)
    return result.returncode == 0, result.stdout.splitlines()


def validate_security(content: str) -> Tuple[float, List[str]]:
    """Layer 3: Security scanning via trivy config. Returns (score 0-1, finding IDs)."""
    import json
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "Dockerfile").write_text(content)
        result = subprocess.run(
            ["trivy", "config", "--quiet", "--format", "json", tmpdir],
            capture_output=True, text=True,
        )
    try:
        data = json.loads(result.stdout)
        results = data.get("Results", [])
        misconfigs = [m for r in results for m in r.get("Misconfigurations", [])]
        high_crit = [m for m in misconfigs if m.get("Severity") in ("HIGH", "CRITICAL")]
        score = max(0.0, 1.0 - len(high_crit) * 0.15)
        return float(score), [m["ID"] for m in high_crit]
    except Exception:
        return 0.5, []


class DockerfileValidator:
    """
    Class wrapper around module-level Dockerfile validation functions.
    Returns a dict with keys: syntax_valid, security_issues, errors, warnings.
    """

    def validate(self, content: str) -> dict:
        errors: list = []
        security_issues: list = []
        warnings: list = []

        syntax_ok = _heuristic_syntax(content)
        if not syntax_ok:
            errors.append("Dockerfile missing FROM instruction")

        sec, warn = _heuristic_security_split(content)
        security_issues.extend(sec)
        warnings.extend(warn)

        return {
            "syntax_valid":    syntax_ok,
            "security_issues": security_issues,
            "errors":          errors,
            "warnings":        warnings,
        }


def _heuristic_syntax(content: str) -> bool:
    """A Dockerfile must contain at least one FROM instruction."""
    return "FROM " in content


def _heuristic_security(content: str) -> list:
    """Legacy flat list — returns all security + warning issues combined."""
    sec, warn = _heuristic_security_split(content)
    return sec + warn


def _heuristic_security_split(content: str):
    """Returns (security_issues, warnings) as separate lists.

    security_issues: high-severity findings (no USER, etc.)
    warnings: best-practice violations (:latest tag, ADD vs COPY)
    """
    security_issues = []
    warnings = []
    lines = content.splitlines()

    # No USER directive → runs as root (security issue)
    if not any(l.strip().upper().startswith("USER ") for l in lines):
        security_issues.append("CKV_DOCKER_8: No USER directive — container runs as root")

    # ADD instead of COPY (warning — ADD is not always wrong)
    if any(l.strip().upper().startswith("ADD ") for l in lines):
        warnings.append("CKV_DOCKER_2: Use COPY instead of ADD unless extracting archives")

    # latest tag (case-insensitive) — warning
    for line in lines:
        stripped_upper = line.strip().upper()
        if stripped_upper.startswith("FROM "):
            parts = stripped_upper.split()
            image = parts[1] if len(parts) > 1 else ""
            # Strip AS alias (e.g. "FROM node:20 AS builder" → "NODE:20")
            image = image.split()[0] if " " in image else image
            if image.endswith(":LATEST") or (":" not in image and image not in ("SCRATCH",)):
                warnings.append("CKV_DOCKER_7: Avoid using :latest tag — pin to a specific version")

    return security_issues, warnings
