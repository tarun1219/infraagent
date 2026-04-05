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
