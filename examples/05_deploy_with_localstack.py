"""
Example 05 — Deploy with LocalStack (Static vs Runtime Validation Gap)

Generates a Terraform configuration, validates it statically with
`terraform validate`, then deploys to LocalStack and captures runtime
failures that static validation misses.

Demonstrates the 8.3pp gap between static validation pass rate and
LocalStack deployment success rate reported in Figure 12 of the paper.

Prerequisites:
  bash scripts/setup_localstack.sh   # starts LocalStack container
  terraform init                     # downloads AWS provider

Run:
  python examples/05_deploy_with_localstack.py
  python examples/05_deploy_with_localstack.py --task s3-encrypted
  python examples/05_deploy_with_localstack.py --all-tasks
  python examples/05_deploy_with_localstack.py --stub
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from infraagent.agent import InfraAgent
from infraagent.generator import ModelID

# ── Task library ──────────────────────────────────────────────────────────────
TASKS: Dict[str, str] = {
    "s3-basic": (
        "Create an AWS S3 bucket named 'infraagent-demo-bucket' with versioning enabled."
    ),
    "s3-encrypted": (
        "Create an AWS S3 bucket 'infraagent-secure-bucket' with server-side encryption "
        "using aws:kms (key managed by AWS), versioning enabled, and public access fully blocked."
    ),
    "iam-role": (
        "Create an IAM role 'lambda-execution-role' with a trust policy allowing "
        "lambda.amazonaws.com to assume it, and attach the AWSLambdaBasicExecutionRole "
        "managed policy."
    ),
    "vpc-subnet": (
        "Create a VPC with CIDR 10.0.0.0/16, two public subnets in us-east-1a and "
        "us-east-1b (10.0.1.0/24, 10.0.2.0/24), an internet gateway, and a route table "
        "associating both subnets to the gateway."
    ),
    "rds-postgres": (
        "Create an RDS PostgreSQL 15.4 db.t3.micro instance named 'infraagent-db' with "
        "encrypted storage (storage_encrypted=true), deletion protection, and a security "
        "group allowing port 5432 only from the VPC CIDR 10.0.0.0/16."
    ),
}

LOCALSTACK_ENDPOINT = os.getenv("LOCALSTACK_ENDPOINT", "http://localhost:4566")
AWS_REGION          = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

TF_PROVIDER_OVERRIDE = """
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  access_key                  = "test"
  secret_key                  = "test"
  region                      = "{region}"
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  endpoints {{
    s3             = "{endpoint}"
    iam            = "{endpoint}"
    ec2            = "{endpoint}"
    rds            = "{endpoint}"
    sts            = "{endpoint}"
  }}
}
"""


@dataclass
class ValidationGapResult:
    task_name:           str
    generated_tf:        str
    static_valid:        bool          # terraform validate passed
    static_errors:       List[str]
    deploy_success:      bool          # terraform apply to LocalStack succeeded
    deploy_errors:       List[str]
    static_time_s:       float
    deploy_time_s:       float
    gap:                 bool = field(init=False)  # static pass but deploy fail

    def __post_init__(self):
        self.gap = self.static_valid and not self.deploy_success

    def summary_line(self) -> str:
        sv = "PASS" if self.static_valid  else "FAIL"
        dv = "PASS" if self.deploy_success else "FAIL"
        gap = " ← GAP" if self.gap else ""
        return (
            f"  {self.task_name:<22}  static={sv}  deploy={dv}"
            f"  ({self.static_time_s:.1f}s / {self.deploy_time_s:.1f}s){gap}"
        )


def _run(cmd: List[str], cwd: Path, timeout: int = 120) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after {timeout}s"
    except FileNotFoundError:
        return 1, "", f"Command not found: {cmd[0]}"


def check_terraform() -> bool:
    rc, _, _ = _run(["terraform", "version"], cwd=Path("."))
    return rc == 0


def check_localstack() -> bool:
    """Check if LocalStack is accepting connections."""
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(
            f"{LOCALSTACK_ENDPOINT}/_localstack/health", timeout=3
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


def static_validate(tf_code: str, workdir: Path) -> Tuple[bool, List[str], float]:
    """Run terraform validate on generated code. Returns (passed, errors, elapsed_s)."""
    tf_file = workdir / "main.tf"
    tf_file.write_text(tf_code)

    t0 = time.perf_counter()

    # terraform init (no-lock, local cache preferred)
    rc, out, err = _run(
        ["terraform", "init", "-no-color", "-input=false", "-upgrade=false"],
        cwd=workdir, timeout=60,
    )
    if rc != 0:
        elapsed = time.perf_counter() - t0
        return False, [f"init failed: {err.strip()[:200]}"], elapsed

    rc, out, err = _run(
        ["terraform", "validate", "-no-color", "-json"],
        cwd=workdir, timeout=30,
    )
    elapsed = time.perf_counter() - t0

    if rc == 0:
        return True, [], elapsed

    errors: List[str] = []
    try:
        result = json.loads(out)
        for diag in result.get("diagnostics", []):
            errors.append(diag.get("summary", str(diag))[:120])
    except json.JSONDecodeError:
        errors = [err.strip()[:200]] if err.strip() else ["terraform validate failed"]
    return False, errors, elapsed


def localstack_deploy(tf_code: str, workdir: Path) -> Tuple[bool, List[str], float]:
    """
    Deploy to LocalStack using terraform apply.
    Injects LocalStack provider override before applying.
    Returns (success, errors, elapsed_s).
    """
    # Inject LocalStack provider block
    provider_block = TF_PROVIDER_OVERRIDE.format(
        region=AWS_REGION,
        endpoint=LOCALSTACK_ENDPOINT,
    )
    full_tf = provider_block + "\n" + tf_code

    tf_file = workdir / "main.tf"
    tf_file.write_text(full_tf)

    t0 = time.perf_counter()

    rc, out, err = _run(
        ["terraform", "init", "-no-color", "-input=false", "-upgrade=false"],
        cwd=workdir, timeout=60,
    )
    if rc != 0:
        elapsed = time.perf_counter() - t0
        return False, [f"init failed: {err.strip()[:200]}"], elapsed

    rc, out, err = _run(
        ["terraform", "apply", "-auto-approve", "-no-color", "-json"],
        cwd=workdir, timeout=90,
    )
    elapsed = time.perf_counter() - t0

    if rc == 0:
        return True, [], elapsed

    errors: List[str] = []
    for line in out.splitlines():
        try:
            obj = json.loads(line)
            if obj.get("type") == "diagnostic" and obj.get("level") == "error":
                errors.append(obj.get("message", "")[:120])
        except json.JSONDecodeError:
            pass
    if not errors and err.strip():
        errors = [err.strip()[:200]]
    if not errors:
        errors = ["terraform apply failed (no error details)"]
    return False, errors, elapsed


def run_task(
    task_name: str,
    task_intent: str,
    model_id: ModelID,
    stub: bool,
    verbose: bool,
) -> Optional[ValidationGapResult]:
    """Generate, statically validate, and deploy one Terraform task."""
    print(f"\n  Task: {task_name}")
    print(f"  Intent: {task_intent[:80]}...")

    agent = InfraAgent(
        model=model_id,
        max_rounds=3,
        use_rag=True,
        use_stub=stub,
        verbose=verbose,
    )

    print("  Generating IaC...", end="", flush=True)
    result = agent.run(task_intent, task_id=f"tf-ls-{task_name}")
    tf_code = result.final_code
    print(f" done ({result.total_duration_s:.1f}s)")

    if verbose:
        print(f"\n  Generated ({len(tf_code)} chars):")
        print("  " + "\n  ".join(tf_code[:600].splitlines()))
        if len(tf_code) > 600:
            print("  [... truncated ...]")

    has_terraform = shutil.which("terraform") is not None

    # Static validation
    print("  Static validate...", end="", flush=True)
    with tempfile.TemporaryDirectory(prefix="ia_static_") as tmp_static:
        if has_terraform:
            static_ok, static_errs, static_t = static_validate(
                tf_code, Path(tmp_static)
            )
        else:
            # Stub: basic heuristic check
            static_ok   = "resource" in tf_code and "}" in tf_code
            static_errs = [] if static_ok else ["No 'resource' block found"]
            static_t    = 0.0
    sv_str = "PASS" if static_ok else "FAIL"
    print(f" [{sv_str}] ({static_t:.1f}s)")
    if not static_ok and static_errs:
        for e in static_errs[:3]:
            print(f"    ✗ {e}")

    # LocalStack deployment
    ls_ok = True
    ls_errs: List[str] = []
    ls_t = 0.0
    ls_available = check_localstack()

    print("  LocalStack deploy...", end="", flush=True)
    if not ls_available:
        print(" [SKIP] LocalStack not running")
        print("  Run: bash scripts/setup_localstack.sh")
        ls_ok = False
        ls_errs = ["LocalStack not available"]
    elif not has_terraform:
        print(" [SKIP] terraform not installed")
        ls_ok = False
        ls_errs = ["terraform not installed"]
    else:
        with tempfile.TemporaryDirectory(prefix="ia_deploy_") as tmp_deploy:
            ls_ok, ls_errs, ls_t = localstack_deploy(tf_code, Path(tmp_deploy))
        dv_str = "PASS" if ls_ok else "FAIL"
        print(f" [{dv_str}] ({ls_t:.1f}s)")
        if not ls_ok and ls_errs:
            for e in ls_errs[:3]:
                print(f"    ✗ {e}")

    return ValidationGapResult(
        task_name=task_name,
        generated_tf=tf_code,
        static_valid=static_ok,
        static_errors=static_errs,
        deploy_success=ls_ok,
        deploy_errors=ls_errs,
        static_time_s=static_t,
        deploy_time_s=ls_t,
    )


def print_gap_analysis(results: List[ValidationGapResult]) -> None:
    total   = len(results)
    s_pass  = sum(1 for r in results if r.static_valid)
    d_pass  = sum(1 for r in results if r.deploy_success and
                  r.deploy_errors != ["LocalStack not available"] and
                  r.deploy_errors != ["terraform not installed"])
    gaps    = sum(1 for r in results if r.gap)

    print(f"\n{'═' * 72}")
    print(f"  Validation Gap Analysis")
    print(f"{'─' * 72}")
    for r in results:
        print(r.summary_line())
    print(f"{'─' * 72}")

    if total > 0:
        s_rate = s_pass / total * 100
        print(f"\n  Static validation pass rate:    {s_pass}/{total}  ({s_rate:.1f}%)")
        if any(not (r.deploy_errors in [["LocalStack not available"],
                                         ["terraform not installed"]])
               for r in results):
            d_rate = d_pass / total * 100 if total > 0 else 0
            gap_pp = s_rate - d_rate
            print(f"  LocalStack deploy pass rate:    {d_pass}/{total}  ({d_rate:.1f}%)")
            print(f"  Validation gap (static − runtime): {gap_pp:+.1f}pp")
            print(f"  Tasks showing gap:              {gaps}/{total}")

    print(f"\n  Paper reference (300 Terraform tasks, sc+rag+3r):")
    print(f"  Static pass rate:  91.2%")
    print(f"  Runtime pass rate: 82.9%")
    print(f"  Gap:               8.3pp  (Figure 12)")
    print()
    print(f"  Gap causes (Figure 13):")
    print(f"    IAM boundary policy enforcement  — 3.1pp")
    print(f"    KMS key existence check           — 2.4pp")
    print(f"    Resource dependency ordering      — 1.8pp")
    print(f"    Region endpoint limitations       — 1.0pp")
    print(f"{'═' * 72}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy Terraform to LocalStack and measure validation gap"
    )
    parser.add_argument(
        "--task",
        choices=list(TASKS),
        default="s3-encrypted",
        help="Single task to run (default: s3-encrypted)",
    )
    parser.add_argument(
        "--all-tasks", action="store_true",
        help="Run all 5 tasks and compute aggregate gap",
    )
    parser.add_argument(
        "--model",
        choices=["deepseek", "codellama", "mistral", "phi3", "gpt4o", "claude"],
        default="deepseek",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--stub", action="store_true",
        help="Stub mode: no Ollama required (static check uses heuristics)",
    )
    args = parser.parse_args()

    model_map = {
        "deepseek": ModelID.DEEPSEEK_CODER, "codellama": ModelID.CODELLAMA,
        "mistral":  ModelID.MISTRAL,        "phi3":      ModelID.PHI3,
        "gpt4o":    ModelID.GPT4O,          "claude":    ModelID.CLAUDE,
    }
    model_id = model_map[args.model]

    print("=" * 72)
    print("  InfraAgent — Terraform + LocalStack Validation Gap")
    print(f"  Model: {args.model}  |  LocalStack: {LOCALSTACK_ENDPOINT}")
    ls_ok = check_localstack()
    tf_ok = shutil.which("terraform") is not None
    print(f"  LocalStack: {'✓ running' if ls_ok else '✗ not running (deploy tests skipped)'}")
    print(f"  Terraform:  {'✓ found' if tf_ok else '✗ not found (install from terraform.io)'}")
    print("=" * 72)

    tasks_to_run = list(TASKS.items()) if args.all_tasks else [(args.task, TASKS[args.task])]
    results: List[ValidationGapResult] = []

    for task_name, task_intent in tasks_to_run:
        r = run_task(task_name, task_intent, model_id, args.stub, args.verbose)
        if r:
            results.append(r)

    if results:
        print_gap_analysis(results)

    print("\n  Setup tips:")
    print("    LocalStack: bash scripts/setup_localstack.sh")
    print("    Terraform:  brew install terraform  # macOS")
    print("    Init:       cd /tmp && terraform init (once, to cache providers)")


if __name__ == "__main__":
    main()
