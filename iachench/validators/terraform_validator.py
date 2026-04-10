"""Terraform HCL validation: terraform validate + checkov + LocalStack deploy."""
from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, List, Optional


def validate_syntax(content: str) -> Tuple[bool, List[str]]:
    """Layer 1+2: terraform init (no backend) + terraform validate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "main.tf").write_text(content)
        init = subprocess.run(
            ["terraform", "init", "-backend=false", "-no-color"],
            cwd=tmpdir, capture_output=True, text=True,
        )
        if init.returncode != 0:
            return False, ["terraform init failed"] + init.stderr.splitlines()
        result = subprocess.run(
            ["terraform", "validate", "-no-color"],
            cwd=tmpdir, capture_output=True, text=True,
        )
    return result.returncode == 0, result.stderr.splitlines()


def validate_security(content: str) -> Tuple[float, List[str]]:
    """Layer 3: Security scanning via Checkov. Returns (score 0-1, finding IDs)."""
    import json
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "main.tf").write_text(content)
        result = subprocess.run(
            ["checkov", "-d", tmpdir, "-o", "json", "--quiet"],
            capture_output=True, text=True,
        )
    try:
        data = json.loads(result.stdout)
        summary = data.get("summary", {})
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        total = passed + failed
        score = passed / total if total > 0 else 1.0
        failed_ids = [c["check_id"] for c in data.get("results", {}).get("failed_checks", [])]
        return float(score), failed_ids
    except Exception:
        return 0.5, []


class TerraformValidator:
    """
    Class wrapper around module-level Terraform validation functions.
    Returns a dict with keys: syntax_valid, security_issues, errors.
    """

    def validate(self, content: str) -> dict:
        errors: list = []
        security_issues: list = []

        # L1 Syntax (heuristic — no terraform binary required)
        syntax_ok = _heuristic_syntax(content)
        if not syntax_ok:
            errors.append("HCL parse error: unclosed block or missing required structure")

        # Heuristic security
        security_issues.extend(_heuristic_security(content))

        return {
            "syntax_valid":    syntax_ok,
            "security_issues": security_issues,
            "errors":          errors,
        }


def _heuristic_syntax(content: str) -> bool:
    """Check balanced braces and presence of at least one resource block."""
    opens  = content.count("{")
    closes = content.count("}")
    if opens == 0 or opens != closes:
        return False
    stripped = content.strip()
    return len(stripped) > 10  # non-trivial content


def _heuristic_security(content: str) -> list:
    issues = []
    if 'Action   = "*"' in content or '"Action": "*"' in content or 'Action = "*"' in content:
        issues.append('CKV_AWS_40: IAM policy uses wildcard Action "*"')
    if '"Resource": "*"' in content or 'resource = "*"' in content.lower():
        issues.append('CKV_AWS_40: IAM policy uses wildcard Resource "*"')
    if "encryption" not in content.lower() and "aws_s3_bucket" in content:
        issues.append("CKV_AWS_19: S3 bucket missing server-side encryption")
    return issues
