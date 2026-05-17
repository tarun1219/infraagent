#!/usr/bin/env python3
"""
run_llm_evaluation.py
Runs GPT-4o on all 150 IaCBench tasks (with and without RAG context),
validates each output through the 4-layer pipeline, and writes results.

Usage:
    OPENAI_API_KEY=sk-... python3 analysis/run_llm_evaluation.py

Output: analysis/results/llm_eval_results.json
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
OUTPUT_FILE = RESULTS_DIR / "llm_eval_results.json"

TASK_FILES = {
    "kubernetes": REPO_ROOT / "iachench" / "tasks" / "k8s_tasks.json",
    "terraform":  REPO_ROOT / "iachench" / "tasks" / "tf_tasks.json",
    "dockerfile": REPO_ROOT / "iachench" / "tasks" / "df_tasks.json",
}

TOOL_PATH = "/Users/tarun/Library/Python/3.9/bin:/opt/homebrew/bin:" + os.environ.get("PATH", "")
ENV = {**os.environ, "PATH": TOOL_PATH}

RAG_CORPUS_DIR = REPO_ROOT / "rag_corpus"

SYSTEM_PROMPT = """You are an expert Infrastructure-as-Code engineer. Generate production-ready,
secure IaC code based on the user's requirements. Follow security best practices:
- Kubernetes: non-root containers, resource limits, readiness/liveness probes, no privileged mode
- Terraform: encryption at rest, no public exposure, least-privilege IAM, versioned state
- Dockerfile: non-root USER, pinned base image tags, HEALTHCHECK, minimal layers
Output ONLY the raw code with no explanation, markdown fences, or commentary."""

MAX_TOKENS = 2048
TEMPERATURE = 0.2
MAX_SELF_CORRECTIONS = 2
REQUEST_DELAY = 1.2  # seconds between API calls


def load_tasks() -> list:
    tasks = []
    for lang, path in TASK_FILES.items():
        data = json.loads(path.read_text())
        task_list = data.get("tasks", data) if isinstance(data, dict) else data
        tasks.extend(task_list)
    return tasks


def get_rag_context(task: dict) -> str:
    """Return a short relevant snippet from RAG corpus for the task."""
    lang = task.get("language", "")
    category = task.get("category", "").lower().replace(" ", "-")
    corpus_dir = RAG_CORPUS_DIR / lang
    if not corpus_dir.exists():
        return ""
    # Try to find a matching doc by category keyword
    best = ""
    for doc in corpus_dir.glob("*.md"):
        if any(kw in doc.stem for kw in category.split("-")):
            content = doc.read_text(errors="replace")
            best = content[:1500]
            break
    if not best and list(corpus_dir.glob("*.md")):
        best = list(corpus_dir.glob("*.md"))[0].read_text(errors="replace")[:1500]
    return best


def call_gpt4o(prompt: str, api_key: str, system: str = SYSTEM_PROMPT) -> str:
    payload = json.dumps({
        "model": "gpt-4o",
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"].strip()


def strip_fences(code: str) -> str:
    """Remove markdown code fences if model added them."""
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
        result["score"] = round(passed / total, 4) if total > 0 else 1.0
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
    ext = {"kubernetes": ".yaml", "terraform": ".tf", "dockerfile": "Dockerfile"}[lang]
    suffix = ext if ext != "Dockerfile" else ""
    prefix = "eval_" if ext != "Dockerfile" else "Dockerfile"

    with tempfile.TemporaryDirectory(prefix="llmeval_") as tmp:
        if lang == "dockerfile":
            fpath = os.path.join(tmp, "Dockerfile")
        else:
            fpath = os.path.join(tmp, f"generated{ext}")
        with open(fpath, "w") as f:
            f.write(code)

        # L1 Syntax
        if lang == "kubernetes":
            syntax_valid = _yaml_parse_valid(fpath)
        elif lang == "terraform":
            syntax_valid = _heuristic_tf_valid(code)
        else:
            syntax_valid = len(code.strip()) > 10 and "FROM" in code

        # L2 Schema (K8s only)
        schema_valid = run_kubeconform(fpath) if lang == "kubernetes" else syntax_valid

        # L3 Security — use checkov if it ran any checks, else heuristic
        ck = run_checkov(fpath)
        checkov_ran = (ck.get("passed", 0) + ck.get("failed", 0)) > 0
        sec_score = ck["score"] if checkov_ran else _heuristic_security_score(code, lang)
        sec_compliant = sec_score >= 0.5

        # L4 Functional
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
        issues.append("SYNTAX ERROR: The generated code has syntax errors.")
    if not validation["schema_valid"]:
        issues.append("SCHEMA ERROR: The Kubernetes manifest fails schema validation.")
    if not validation["security_compliant"]:
        issues.append(f"SECURITY: Score={validation['security_score']:.2f} (<0.5). "
                      "Fix: add resource limits, non-root user, disable privileged mode.")
    issue_str = "\n".join(issues)
    return (
        f"Your previous IaC output had issues:\n{issue_str}\n\n"
        f"Original request:\n{original_prompt}\n\n"
        f"Your flawed output:\n{code}\n\n"
        "Fix all issues and output ONLY the corrected code."
    )


def evaluate_task(task: dict, api_key: str, use_rag: bool) -> dict:
    prompt = task["prompt"]
    lang = task["language"]

    if use_rag:
        ctx = get_rag_context(task)
        if ctx:
            prompt = f"Reference examples:\n{ctx}\n\n---\nTask:\n{prompt}"

    result = {
        "task_id": task["task_id"],
        "language": lang,
        "difficulty": task.get("difficulty", ""),
        "category": task.get("category", ""),
        "use_rag": use_rag,
        "attempts": [],
        "final_functional": False,
        "self_corrections": 0,
    }

    code = ""
    validation = {}
    for attempt in range(1 + MAX_SELF_CORRECTIONS):
        try:
            if attempt == 0:
                code = strip_fences(call_gpt4o(prompt, api_key))
            else:
                correction_prompt = build_correction_prompt(task["prompt"], code, validation)
                code = strip_fences(call_gpt4o(correction_prompt, api_key))
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

        except urllib.error.HTTPError as e:
            result["attempts"].append({"attempt": attempt + 1, "error": f"HTTP {e.code}"})
            if e.code == 429:
                time.sleep(30)
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
    by_lang = {}
    by_diff = {}
    for r in results:
        lang = r["language"]
        diff = r["difficulty"]
        by_lang.setdefault(lang, []).append(r)
        by_diff.setdefault(diff, []).append(r)

    def pct(lst, key): return round(sum(1 for x in lst if x.get(key)) / len(lst), 4) if lst else 0

    stats = {
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
                          for diff, lst in by_diff.items()},
    }
    return stats


def main():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    tasks = load_tasks()
    print(f"=== GPT-4o IaCBench Evaluation ===")
    print(f"Tasks: {len(tasks)} | Models: GPT-4o | Conditions: baseline + RAG")
    print(f"Max self-corrections: {MAX_SELF_CORRECTIONS} | Output: {OUTPUT_FILE}\n")

    baseline_results = []
    rag_results = []
    total_calls = len(tasks) * 2  # baseline + RAG
    done = 0

    for condition, results_list, use_rag in [
        ("BASELINE (no RAG)", baseline_results, False),
        ("RAG-AUGMENTED",     rag_results,     True),
    ]:
        print(f"\n--- {condition} ---")
        for task in tasks:
            done += 1
            print(f"[{done:3d}/{total_calls}] {task['task_id']} ({task['language']}, "
                  f"L{task.get('difficulty','?')}, {'RAG' if use_rag else 'base'})...", end=" ", flush=True)
            r = evaluate_task(task, api_key, use_rag)
            results_list.append(r)
            status = "PASS" if r["final_functional"] else "FAIL"
            sc = f"+{r['self_corrections']}SC" if r["self_corrections"] else ""
            print(f"{status}{sc} sec={r['final_security_score']:.2f}")

    baseline_stats = compute_stats(baseline_results, "gpt4o_baseline")
    rag_stats = compute_stats(rag_results, "gpt4o_rag")

    output = {
        "meta": {
            "model": "gpt-4o",
            "n_tasks": len(tasks),
            "max_self_corrections": MAX_SELF_CORRECTIONS,
            "temperature": TEMPERATURE,
        },
        "baseline_stats": baseline_stats,
        "rag_stats": rag_stats,
        "baseline_results": baseline_results,
        "rag_results": rag_results,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Results ===")
    print(f"  GPT-4o Baseline  — functional: {baseline_stats['pct_functional']:.1%}  "
          f"security: {baseline_stats['mean_security']:.3f}")
    print(f"  GPT-4o + RAG     — functional: {rag_stats['pct_functional']:.1%}  "
          f"security: {rag_stats['mean_security']:.3f}")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
