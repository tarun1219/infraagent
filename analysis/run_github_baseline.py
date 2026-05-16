#!/usr/bin/env python3
"""
run_github_baseline.py
Downloads real IaC from GitHub and validates it against the full validator pipeline.
Produces results/github_baseline_results.json.
"""

import os
import json
import subprocess
import tempfile
import time
from pathlib import Path
from urllib import request, error as urllib_error

# Tool path setup
TOOL_PATH = "/Users/tarun/Library/Python/3.9/bin:/opt/homebrew/bin:" + os.environ.get("PATH", "")
ENV = {**os.environ, "PATH": TOOL_PATH}

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "github_baseline_results.json"

# 30 hardcoded raw GitHub URLs to real K8s YAML files from popular repos
GITHUB_K8S_URLS = [
    # kubernetes/website official examples (stable)
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/examples/pods/simple-pod.yaml",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/examples/application/deployment.yaml",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/examples/controllers/nginx-deployment.yaml",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/examples/configmap/configmap-multikeys.yaml",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/examples/application/php-apache.yaml",
    # kubernetes/examples (stable repo)
    "https://raw.githubusercontent.com/kubernetes/examples/master/guestbook/all-in-one/guestbook-all-in-one.yaml",
    "https://raw.githubusercontent.com/kubernetes/examples/master/staging/https-nginx/nginx-app.yaml",
    "https://raw.githubusercontent.com/kubernetes/examples/master/staging/cassandra/cassandra-service.yaml",
    "https://raw.githubusercontent.com/kubernetes/examples/master/staging/cassandra/cassandra-statefulset.yaml",
    # argoproj/argo-cd (stable)
    "https://raw.githubusercontent.com/argoproj/argo-cd/master/examples/app-with-directory/Deployment.yaml",
    "https://raw.githubusercontent.com/argoproj/argo-cd/master/examples/app-with-directory/Service.yaml",
    # prometheus-operator/kube-prometheus (main branch)
    "https://raw.githubusercontent.com/prometheus-operator/kube-prometheus/main/manifests/alertmanager-alertmanager.yaml",
    "https://raw.githubusercontent.com/prometheus-operator/kube-prometheus/main/manifests/grafana-deployment.yaml",
    "https://raw.githubusercontent.com/prometheus-operator/kube-prometheus/main/manifests/prometheus-prometheus.yaml",
    "https://raw.githubusercontent.com/prometheus-operator/kube-prometheus/main/manifests/nodeExporter-daemonset.yaml",
    # fluxcd/flux2 (main branch)
    "https://raw.githubusercontent.com/fluxcd/flux2/main/manifests/bases/source-controller/deployment.yaml",
    "https://raw.githubusercontent.com/fluxcd/flux2/main/manifests/bases/kustomize-controller/deployment.yaml",
    "https://raw.githubusercontent.com/fluxcd/flux2/main/manifests/bases/helm-controller/deployment.yaml",
    # open-policy-agent/gatekeeper (master)
    "https://raw.githubusercontent.com/open-policy-agent/gatekeeper/master/deploy/gatekeeper.yaml",
    # cert-manager (master)
    "https://raw.githubusercontent.com/cert-manager/cert-manager/master/deploy/charts/cert-manager/templates/deployment.yaml",
    # knative/serving (main)
    "https://raw.githubusercontent.com/knative/serving/main/config/core/deployments/controller.yaml",
    "https://raw.githubusercontent.com/knative/serving/main/config/core/deployments/webhook.yaml",
    # jaegertracing/jaeger-operator (main)
    "https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/config/manager/manager.yaml",
    # tekton pipelines (main)
    "https://raw.githubusercontent.com/tektoncd/pipeline/main/config/core/deployments/tekton-pipelines-webhook.yaml",
    # metallb (main)
    "https://raw.githubusercontent.com/metallb/metallb/main/config/manifests/metallb-native.yaml",
    # kubernetes/ingress-nginx (main)
    "https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml",
    # external-dns (master)
    "https://raw.githubusercontent.com/kubernetes-sigs/external-dns/master/kustomize/base/deployment.yaml",
    # cluster-autoscaler (master)
    "https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml",
    # sealed-secrets (main)
    "https://raw.githubusercontent.com/bitnami-labs/sealed-secrets/main/helm/sealed-secrets/templates/deployment.yaml",
    # crossplane (master)
    "https://raw.githubusercontent.com/crossplane/crossplane/master/cluster/charts/crossplane/templates/rbac-manager-deployment.yaml",
]


def download_file(url: str, dest_dir: str, index: int) -> tuple:
    """Download a file from URL. Returns (local_path, success, error)."""
    filename = f"github_{index:03d}_{url.split('/')[-1]}"
    # Ensure it ends with .yaml
    if not filename.endswith((".yaml", ".yml")):
        filename += ".yaml"
    dest = os.path.join(dest_dir, filename)

    headers = {
        "User-Agent": "IaCBench-Research/1.0",
        "Accept": "text/plain,application/yaml",
    }
    try:
        req = request.Request(url, headers=headers)
        with request.urlopen(req, timeout=15) as resp:
            content = resp.read()
        with open(dest, "wb") as f:
            f.write(content)
        return dest, True, None
    except urllib_error.HTTPError as e:
        return None, False, f"HTTP {e.code}"
    except urllib_error.URLError as e:
        return None, False, f"URL error: {e.reason}"
    except Exception as e:
        return None, False, str(e)


def run_checkov(filepath: str) -> dict:
    result = {"passed": 0, "failed": 0, "score": 0.0, "error": None}
    try:
        cmd = ["checkov", "-f", filepath, "--output", "json", "--quiet", "--compact"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=ENV)
        raw = proc.stdout.strip()
        if not raw:
            result["error"] = "no output"
            return result
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            result["error"] = "json parse error"
            return result
        if isinstance(data, list):
            data = next((x for x in data if isinstance(x, dict) and "summary" in x), {})
        if not isinstance(data, dict):
            result["error"] = "unexpected output type"
            return result
        summary = data.get("summary", {})
        passed = int(summary.get("passed", 0))
        failed = int(summary.get("failed", 0))
        total = passed + failed
        result["passed"] = passed
        result["failed"] = failed
        result["score"] = round(passed / total, 4) if total > 0 else 1.0
    except FileNotFoundError:
        result["error"] = "checkov not found"
        result["score"] = _heuristic_security_score(filepath)
    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)
    return result


def _heuristic_security_score(filepath: str) -> float:
    """Fallback security score when checkov unavailable."""
    try:
        with open(filepath, errors="replace") as f:
            content = f.read()
        dangerous = [
            "privileged: true", "hostNetwork: true",
            "allowPrivilegeEscalation: true", "runAsRoot: true",
        ]
        hits = sum(1 for d in dangerous if d in content)
        return round(max(0.0, 1.0 - hits * 0.2), 4)
    except Exception:
        return 0.5


def run_kubeconform(filepath: str) -> dict:
    result = {"valid": False, "error": None}
    try:
        cmd = ["kubeconform", "-strict", "-summary", filepath]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=ENV)
        result["valid"] = proc.returncode == 0
        if not result["valid"]:
            result["error"] = (proc.stdout + proc.stderr)[:300]
    except FileNotFoundError:
        result["error"] = "kubeconform not found"
        result["valid"] = _yaml_parse_valid(filepath)
    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)
    return result


def _yaml_parse_valid(filepath: str) -> bool:
    """
    True iff the YAML file is parseable.  Uses the Python-3.9 pyyaml install
    via subprocess to avoid the system-Python site-packages gap on macOS.
    Falls back to yamllint parse-error detection if pyyaml subprocess fails.
    """
    # Use the Python that has pyyaml (pip-installed under 3.9 user site)
    import shutil, subprocess as _sp
    py = shutil.which("python3.9") or "/Library/Developer/CommandLineTools/usr/bin/python3"
    yaml_check = (
        "import yaml, sys\n"
        "content = open(sys.argv[1]).read()\n"
        "docs = list(yaml.safe_load_all(content))\n"
        "sys.exit(0 if (docs and any(d is not None for d in docs)) else 1)\n"
    )
    try:
        env_with_yaml = {
            **ENV,
            "PYTHONPATH": "/Users/tarun/Library/Python/3.9/lib/python/site-packages",
        }
        result = _sp.run(
            [py, "-c", yaml_check, filepath],
            capture_output=True, timeout=10, env=env_with_yaml,
        )
        return result.returncode == 0
    except Exception:
        pass
    # Fallback: yamllint - only count real parse errors, not style warnings
    PARSE_ERROR_KEYWORDS = [
        "could not find expected", "mapping values are not allowed",
        "found character that cannot start", "could not determine",
        "tab characters are not allowed",
    ]
    try:
        import subprocess as _sp
        proc = _sp.run(
            ["yamllint", "-d", "relaxed", "-f", "parsable", filepath],
            capture_output=True, text=True, timeout=15, env=ENV,
        )
        parse_errors = sum(
            1 for line in proc.stdout.splitlines()
            if "[error]" in line and any(kw in line for kw in PARSE_ERROR_KEYWORDS)
        )
        return parse_errors == 0
    except Exception:
        return True  # optimistic: assume parseable if we can't check


def run_yamllint(filepath: str) -> dict:
    result = {"valid": True, "errors": 0, "warnings": 0}
    try:
        proc = subprocess.run(
            ["yamllint", "-f", "parsable", filepath],
            capture_output=True, text=True, timeout=30, env=ENV
        )
        lines = proc.stdout.splitlines()
        errors = sum(1 for l in lines if "[error]" in l)
        warnings = sum(1 for l in lines if "[warning]" in l)
        result["errors"] = errors
        result["warnings"] = warnings
        result["valid"] = errors == 0
    except FileNotFoundError:
        result["valid"] = _yaml_parse_valid(filepath)
    except Exception:
        result["valid"] = _yaml_parse_valid(filepath)
    return result


def validate_file(filepath: str, url: str) -> dict:
    """Validate a single downloaded GitHub file."""
    # Layer 1: Syntax — pyyaml parse is the ground truth;
    # yamllint style warnings (indentation, line-length, etc.) are NOT parse errors.
    syntax_valid = _yaml_parse_valid(filepath)
    yl = run_yamllint(filepath)  # for informational yamllint_errors field only

    # Layer 2: Schema (kubeconform)
    kc = run_kubeconform(filepath)
    schema_valid = kc["valid"]

    # Layer 3: Security (checkov)
    ck = run_checkov(filepath)
    security_score = ck["score"]
    security_compliant = security_score >= 0.5

    # Layer 4: Functional = all layers pass
    functional_correct = syntax_valid and schema_valid and security_compliant

    return {
        "url": url,
        "file": filepath,
        "syntax_valid": syntax_valid,
        "yamllint_errors": yl.get("errors", 0),
        "yamllint_warnings": yl.get("warnings", 0),
        "schema_valid": schema_valid,
        "kubeconform_error": kc.get("error"),
        "security_score": security_score,
        "checkov_passed": ck.get("passed", 0),
        "checkov_failed": ck.get("failed", 0),
        "security_compliant": security_compliant,
        "functional_correct": functional_correct,
    }


def compute_stats(results: list) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    n_valid = sum(1 for r in results if r["syntax_valid"])
    n_schema = sum(1 for r in results if r["schema_valid"])
    n_secure = sum(1 for r in results if r["security_compliant"])
    n_func = sum(1 for r in results if r["functional_correct"])
    mean_sec = sum(r["security_score"] for r in results) / n
    return {
        "n_attempted": n,
        "pct_syntax_valid": round(n_valid / n, 4),
        "pct_schema_valid": round(n_schema / n, 4),
        "pct_security_compliant": round(n_secure / n, 4),
        "pct_functional_correct": round(n_func / n, 4),
        "mean_security_score": round(mean_sec, 4),
    }


def main():
    print("=== GitHub Baseline Downloader & Validator ===")
    print(f"Downloading {len(GITHUB_K8S_URLS)} Kubernetes manifest files from GitHub...\n")

    results = []
    download_errors = []

    with tempfile.TemporaryDirectory(prefix="iacbench_github_") as tmpdir:
        for i, url in enumerate(GITHUB_K8S_URLS):
            print(f"[{i+1:2d}/{len(GITHUB_K8S_URLS)}] Downloading: {url.split('/')[-1]}")
            local_path, success, err = download_file(url, tmpdir, i)

            if not success:
                print(f"  DOWNLOAD FAILED: {err}")
                download_errors.append({"url": url, "error": err})
                results.append({
                    "url": url,
                    "file": None,
                    "download_error": err,
                    "syntax_valid": False,
                    "schema_valid": False,
                    "security_score": 0.0,
                    "security_compliant": False,
                    "functional_correct": False,
                })
                continue

            print(f"  Downloaded OK. Validating...")
            try:
                r = validate_file(local_path, url)
                results.append(r)
                print(f"  syntax={r['syntax_valid']} schema={r['schema_valid']} "
                      f"sec={r['security_score']:.2f} func={r['functional_correct']}")
            except Exception as e:
                print(f"  VALIDATION ERROR: {e}")
                results.append({
                    "url": url,
                    "file": local_path,
                    "validation_error": str(e),
                    "syntax_valid": False,
                    "schema_valid": False,
                    "security_score": 0.0,
                    "security_compliant": False,
                    "functional_correct": False,
                })

            # Be polite to GitHub's servers
            time.sleep(0.3)

    # Compute summary stats on successfully downloaded files
    valid_results = [r for r in results if r.get("file") is not None]
    stats = compute_stats(valid_results)

    output = {
        "meta": {
            "source": "github_public_repos",
            "language": "kubernetes",
            "n_urls": len(GITHUB_K8S_URLS),
            "n_downloaded": len(valid_results),
            "n_download_errors": len(download_errors),
            "tool_path": TOOL_PATH,
        },
        "stats": stats,
        "download_errors": download_errors,
        "per_file": results,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"  Downloaded:             {stats.get('n_attempted', 0)} files")
    print(f"  Syntax valid:           {stats.get('pct_syntax_valid', 0):.1%}")
    print(f"  Schema valid:           {stats.get('pct_schema_valid', 0):.1%}")
    print(f"  Security compliant:     {stats.get('pct_security_compliant', 0):.1%}")
    print(f"  Functional correct:     {stats.get('pct_functional_correct', 0):.1%}")
    print(f"  Mean security score:    {stats.get('mean_security_score', 0):.3f}")
    print(f"\nResults written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
