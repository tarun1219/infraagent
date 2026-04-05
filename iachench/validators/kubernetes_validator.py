"""Kubernetes manifest validation: yamllint → kubeconform → kubectl dry-run=server."""
from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional, List


def validate_syntax(content: str) -> Tuple[bool, List[str]]:
    """Layer 1: YAML syntax validation via yamllint."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        fname = f.name
    result = subprocess.run(
        ["yamllint", "-d", "{extends: default, rules: {line-length: disable}}", fname],
        capture_output=True, text=True,
    )
    Path(fname).unlink(missing_ok=True)
    return result.returncode == 0, result.stdout.splitlines()


def validate_schema(content: str, k8s_version: str = "1.28.0") -> Tuple[bool, List[str]]:
    """Layer 2: Schema validation via kubeconform (strict mode)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        fname = f.name
    result = subprocess.run(
        ["kubeconform", "-strict", "-kubernetes-version", k8s_version, fname],
        capture_output=True, text=True,
    )
    Path(fname).unlink(missing_ok=True)
    return result.returncode == 0, result.stderr.splitlines()


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
