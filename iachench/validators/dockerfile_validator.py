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
