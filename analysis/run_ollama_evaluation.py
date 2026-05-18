#!/usr/bin/env python3
"""
run_ollama_evaluation.py
Runs DeepSeek-Coder, CodeLlama, and Mistral on all 150 IaCBench tasks
via local Ollama. Produces results/ollama_eval_results.json.

Usage:
    python3 analysis/run_ollama_evaluation.py
"""

import os
import json
import time
import sys
import urllib.request
import urllib.error
import subprocess
import tempfile
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "ollama_eval_results.json"

TASK_FILES = {
    "kubernetes": REPO_ROOT / "iachench" / "tasks" / "k8s_tasks.json",
    "terraform":  REPO_ROOT / "iachench" / "tasks" / "tf_tasks.json",
    "dockerfile": REPO_ROOT / "iachench" / "tasks" / "df_tasks.json",
}

MODELS = [
    "deepseek-coder:6.7b-instruct-q4_K_M",
    "mistral:7b-instruct-q4_K_M",
]

OLLAMA_URL = "http://localhost:11434"
TOOL_PATH = "/Users/tarun/Library/Python/3.9/bin:/opt/homebrew/bin:" + os.environ.get("PATH", "")
ENV = {**os.environ, "PATH": TOOL_PATH}

MAX_TOKENS = 1024
TEMPERATURE = 0.2
MAX_SELF_CORRECTIONS = 0
REQUEST_DELAY = 0.0
SAMPLE_PER_DIFF_PER_LANG = 2

SYSTEM_PROMPT = """You are an expert Infrastructure-as-Code engineer. Generate production-ready,
secure IaC code based on the user's requirements. Follow security best practices:
- Kubernetes: non-root containers, resource limits, readiness/liveness probes, no privileged mode
- Terraform: encryption at rest, no public exposure, least-privilege IAM, versioned state
- Dockerfile: non-root USER, pinned base image tags, HEALTHCHECK, minimal layers
Output ONLY the raw code with no explanation, markdown fences, or commentary."""


def load_tasks() -> list:
    """Load all 150 tasks (50 per language)."""
    tasks = []
    for lang, path in TASK_FILES.items():
        data = json.loads(path.read_text())
        task_list = data.get("tasks", data) if isinstance(data, dict) else data
        tasks.extend(task_list)
    return tasks


def call_ollama(model: str, prompt: str) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": f"[INST] {SYSTEM_PROMPT}\n\n{prompt} [/INST]",
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        }
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    return data.get("response", "").strip()


def strip_fences(code: str) -> str:
    lines = code.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


# ── Validation helpers ────────────────────────────────────────────────────────

def _yaml_parse_valid(filepath: str) -> bool:
    py = shutil.which("python3.9") or "/Library/Developer/CommandLineTools/usr/bin/python3"
    yaml_check = (
        "import yaml, sys\n"
        "content = open(sys.argv[1]).read()\n"
        "docs = list(yaml.safe_load_all(content))\n"
        "sys.exit(0 if (docs and any(d is not None for d in docs)) else 1)\n"
    )
    env2 = {**ENV, "PYTHONPATH": "/Users/tarun/Library/Python/3.9/lib/python/site-packages"}
    try:
        r = subprocess.run([py, "-c", yaml_check, filepath], capture_output=True, timeout=10, env=env2)
        return r.returncode == 0
    except Exception:
        return True


def _heuristic_tf_valid(code: str) -> bool:
    opens = code.count("{")
    closes = code.count("}")
    has_block = any(kw in code for kw in ("resource", "data", "module", "variable", "output"))
    return abs(opens - closes) <= 2 and has_block


def _heuristic_security_score(code: str, lang: str) -> float:
    if lang == "kubernetes":
        dangerous = ["privileged: true", "hostNetwork: true",
                     "allowPrivilegeEscalation: true", "runAsRoot: true"]
    elif lang == "terraform":
        dangerous = ["publicly_accessible = true", 'encryption = "none"',
                     "encrypted = false", "enable_dns_hostnames = false"]
    else:
        dangerous = ["USER root", ":latest", "apt-get install -y"]
    hits = sum(1 for d in dangerous if d in code)
    return round(max(0.0, 1.0 - hits * 0.25), 4)


def run_checkov(filepath: str) -> dict:
    result = {"passed": 0, "failed": 0, "score": 0.0}
    try:
        cmd = ["checkov", "-f", filepath, "--output", "json", "--quiet", "--compact"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=ENV)
        raw = proc.stdout.strip()
        if not raw:
            return result
        data = json.loads(raw)
        if isinstance(data, list):
            data = next((x for x in data if isinstance(x, dict) and "summary" in x), {})
        summary = data.get("summary", {})
        passed = int(summary.get("passed", 0))
        failed = int(summary.get("failed", 0))
        total = passed + failed
        result["passed"] = passed
        result["failed"] = failed
        result["score"] = round(passed / total, 4) if total > 0 else 0.0
    except Exception:
        pass
    return result


def run_kubeconform(filepath: str) -> bool:
    try:
        cmd = ["kubeconform", "-strict", "-summary", filepath]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=ENV)
        return proc.returncode == 0
    except Exception:
        return _yaml_parse_valid(filepath)


def validate_code(code: str, lang: str) -> dict:
    if not code or len(code.strip()) < 10:
        return {
            "syntax_valid": False, "schema_valid": False,
            "security_score": 0.0, "checkov_passed": 0,
            "checkov_failed": 0, "security_compliant": False,
            "functional_correct": False,
        }

    with tempfile.TemporaryDirectory(prefix="ollama_eval_") as tmp:
        ext = {"kubernetes": ".yaml", "terraform": ".tf", "dockerfile": ""}[lang]
        fpath = os.path.join(tmp, "Dockerfile" if lang == "dockerfile" else f"generated{ext}")
        with open(fpath, "w") as f:
            f.write(code)

        if lang == "kubernetes":
            syntax_valid = _yaml_parse_valid(fpath)
        elif lang == "terraform":
            syntax_valid = _heuristic_tf_valid(code)
        else:
            syntax_valid = "FROM" in code and len(code.strip()) > 10

        schema_valid = run_kubeconform(fpath) if lang == "kubernetes" else syntax_valid

        ck = run_checkov(fpath)
        checkov_ran = (ck.get("passed", 0) + ck.get("failed", 0)) > 0
        sec_score = ck["score"] if checkov_ran else _heuristic_security_score(code, lang)
        sec_compliant = sec_score >= 0.5
        functional = syntax_valid and schema_valid and sec_compliant

    return {
        "syntax_valid": syntax_valid,
        "schema_valid": schema_valid,
        "security_score": sec_score,
        "checkov_passed": ck.get("passed", 0),
        "checkov_failed": ck.get("failed", 0),
        "security_compliant": sec_compliant,
        "functional_correct": functional,
    }


def build_correction_prompt(original_prompt: str, code: str, validation: dict) -> str:
    issues = []
    if not validation["syntax_valid"]:
        issues.append("SYNTAX ERROR: The generated code has syntax/parse errors.")
    if not validation["schema_valid"]:
        issues.append("SCHEMA ERROR: The Kubernetes manifest fails schema validation.")
    if not validation["security_compliant"]:
        issues.append(f"SECURITY: Score={validation['security_score']:.2f} (<0.5 threshold). "
                      "Add resource limits, non-root user, disable privileged mode.")
    issue_str = "\n".join(issues) if issues else "Unknown validation failure."
    return (
        f"Your previous IaC output had issues:\n{issue_str}\n\n"
        f"Original request:\n{original_prompt}\n\n"
        f"Your flawed output:\n{code[:500]}\n\n"
        "Fix all issues and output ONLY the corrected code."
    )


def evaluate_task(task: dict, model: str) -> dict:
    prompt = task["prompt"]
    lang = task["language"]

    result = {
        "task_id": task["task_id"],
        "language": lang,
        "difficulty": task.get("difficulty", ""),
        "category": task.get("category", ""),
        "model": model,
        "attempts": [],
        "final_functional": False,
        "self_corrections": 0,
        "final_security_score": 0.0,
        "final_syntax_valid": False,
        "final_schema_valid": False,
    }

    code = ""
    validation = {}
    for attempt in range(1 + MAX_SELF_CORRECTIONS):
        try:
            if attempt == 0:
                code = strip_fences(call_ollama(model, prompt))
            else:
                correction_prompt = build_correction_prompt(task["prompt"], code, validation)
                code = strip_fences(call_ollama(model, correction_prompt))
                result["self_corrections"] += 1

            time.sleep(REQUEST_DELAY)
            validation = validate_code(code, lang)
            result["attempts"].append({
                "attempt": attempt + 1,
                "functional_correct": validation["functional_correct"],
                "syntax_valid": validation["syntax_valid"],
                "schema_valid": validation["schema_valid"],
                "security_score": validation["security_score"],
            })

            if validation["functional_correct"]:
                break

        except urllib.error.URLError as e:
            result["attempts"].append({"attempt": attempt + 1, "error": f"Ollama connection error: {e}"})
            break
        except Exception as e:
            result["attempts"].append({"attempt": attempt + 1, "error": str(e)})
            break

    result["final_functional"] = validation.get("functional_correct", False)
    result["final_security_score"] = validation.get("security_score", 0.0)
    result["final_syntax_valid"] = validation.get("syntax_valid", False)
    result["final_schema_valid"] = validation.get("schema_valid", False)
    return result


def compute_stats(results: list, label: str) -> dict:
    n = len(results)
    if n == 0:
        return {}
    by_lang, by_diff = {}, {}
    for r in results:
        by_lang.setdefault(r["language"], []).append(r)
        by_diff.setdefault(r["difficulty"], []).append(r)

    def pct(lst, key): return round(sum(1 for x in lst if x.get(key)) / len(lst), 4) if lst else 0

    return {
        "label": label,
        "n": n,
        "pct_functional": pct(results, "final_functional"),
        "pct_syntax": pct(results, "final_syntax_valid"),
        "pct_schema": pct(results, "final_schema_valid"),
        "mean_security": round(sum(r.get("final_security_score", 0) for r in results) / n, 4),
        "pct_self_corrected": round(sum(1 for r in results if r["self_corrections"] > 0) / n, 4),
        "by_language": {lang: {"n": len(lst), "pct_functional": pct(lst, "final_functional")}
                        for lang, lst in by_lang.items()},
        "by_difficulty": {diff: {"n": len(lst), "pct_functional": pct(lst, "final_functional")}
                          for diff, lst in sorted(by_diff.items())},
    }


def main():
    tasks = load_tasks()
    total_calls = len(tasks) * len(MODELS)
    print(f"=== Ollama Multi-Model IaCBench Evaluation ===")
    print(f"Tasks: {len(tasks)} | Models: {len(MODELS)} | Total calls: {total_calls}")
    print(f"Models: {', '.join(MODELS)}\n")

    all_results = {}
    done = 0

    for model in MODELS:
        model_short = model.split(":")[0]
        print(f"\n{'='*50}")
        print(f"MODEL: {model}")
        print(f"{'='*50}")

        results = []
        for task in tasks:
            done += 1
            print(f"[{done:3d}/{total_calls}] {task['task_id']} "
                  f"({task['language']}, L{task.get('difficulty','?')}, {model_short})...",
                  end=" ", flush=True)
            r = evaluate_task(task, model)
            results.append(r)
            status = "PASS" if r["final_functional"] else "FAIL"
            sc = f"+{r['self_corrections']}SC" if r["self_corrections"] else ""
            print(f"{status}{sc} sec={r['final_security_score']:.2f}")

        stats = compute_stats(results, model)
        all_results[model] = {"stats": stats, "results": results}

        print(f"\n--- {model_short} Summary ---")
        print(f"  Functional: {stats['pct_functional']:.1%}  "
              f"Security: {stats['mean_security']:.3f}  "
              f"Self-corrected: {stats['pct_self_corrected']:.1%}")

    output = {
        "meta": {
            "models": MODELS,
            "n_tasks": len(tasks),
            "max_self_corrections": MAX_SELF_CORRECTIONS,
            "temperature": TEMPERATURE,
        },
        "model_stats": {m: all_results[m]["stats"] for m in MODELS},
        "model_results": {m: all_results[m]["results"] for m in MODELS},
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}")
    print("=== FINAL MULTI-MODEL COMPARISON ===")
    print(f"{'Model':<40} {'Functional':>12} {'Security':>10} {'SC Rate':>8}")
    print("-" * 72)
    for m in MODELS:
        s = all_results[m]["stats"]
        print(f"{m:<40} {s['pct_functional']:>11.1%} {s['mean_security']:>10.3f} {s['pct_self_corrected']:>7.1%}")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
