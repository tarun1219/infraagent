#!/usr/bin/env python3
"""
run_groq_evaluation.py
Runs Llama-3.3-70B and Llama-3.1-8B on all 150 IaCBench tasks
(baseline + RAG) via Groq cloud API, with parallel execution.
Produces results/groq_eval_results.json.

Usage:
    GROQ_API_KEY=gsk_... python3 analysis/run_groq_evaluation.py
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / f"{os.environ.get('PROVIDER', 'groq')}_eval_results.json"

TASK_FILES = {
    "kubernetes": REPO_ROOT / "iachench" / "tasks" / "k8s_tasks.json",
    "terraform":  REPO_ROOT / "iachench" / "tasks" / "tf_tasks.json",
    "dockerfile": REPO_ROOT / "iachench" / "tasks" / "df_tasks.json",
}

RAG_CORPUS_DIR = REPO_ROOT / "rag_corpus"

# ── Provider config ───────────────────────────────────────────────────────────
# Set PROVIDER env var to: groq | together | openrouter  (default: groq)
PROVIDER = os.environ.get("PROVIDER", "groq")

PROVIDER_CONFIG = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    },
    "together": {
        "url": "https://api.together.xyz/v1/chat/completions",
        "key_env": "TOGETHER_API_KEY",
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "models": [
            "deepseek/deepseek-v4-flash:free",
            "google/gemma-4-26b-a4b-it:free",
        ],
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "key_env": "ANTHROPIC_API_KEY",
        "models": [
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5-20250929",
        ],
    },
}

cfg = PROVIDER_CONFIG[PROVIDER]
MODELS = cfg["models"]
GROQ_API_URL = cfg["url"]
TOOL_PATH = os.environ.get("PATH", "")
ENV = {**os.environ, "PATH": TOOL_PATH}

MAX_TOKENS = 2048
TEMPERATURE = 0.2
_prov = os.environ.get("PROVIDER", "groq")
# OpenRouter free tier: 0 self-corrections, 1 worker, 10s delay
# Anthropic paid: 2 self-corrections, 5 workers, 0.3s delay
# Groq/Together: 2 self-corrections, 8 workers, 0.2s delay
MAX_SELF_CORRECTIONS = 0 if _prov == "openrouter" else 2
MAX_WORKERS = 1 if _prov == "openrouter" else (5 if _prov == "anthropic" else 8)
REQUEST_DELAY = 10.0 if _prov == "openrouter" else (0.3 if _prov == "anthropic" else 0.2)

SYSTEM_PROMPT = """You are an expert Infrastructure-as-Code engineer. Generate production-ready,
secure IaC code based on the user's requirements. Follow security best practices:
- Kubernetes: non-root containers, resource limits, readiness/liveness probes, no privileged mode
- Terraform: encryption at rest, no public exposure, least-privilege IAM, versioned state
- Dockerfile: non-root USER, pinned base image tags, HEALTHCHECK, minimal layers
Output ONLY the raw code with no explanation, markdown fences, or commentary."""

print_lock = threading.Lock()


def load_tasks() -> list:
    tasks = []
    for lang, path in TASK_FILES.items():
        data = json.loads(path.read_text())
        task_list = data.get("tasks", data) if isinstance(data, dict) else data
        tasks.extend(task_list)
    return tasks


def get_rag_context(task: dict) -> str:
    lang = task.get("language", "")
    category = task.get("category", "").lower().replace(" ", "-")
    corpus_dir = RAG_CORPUS_DIR / lang
    if not corpus_dir.exists():
        return ""
    for doc in corpus_dir.glob("*.md"):
        if any(kw in doc.stem for kw in category.split("-")):
            return doc.read_text(errors="replace")[:1500]
    docs = list(corpus_dir.glob("*.md"))
    return docs[0].read_text(errors="replace")[:1500] if docs else ""


def call_anthropic(model: str, prompt: str, api_key: str) -> str:
    payload = json.dumps({
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return data["content"][0]["text"].strip()


def call_groq(model: str, prompt: str, api_key: str) -> str:
    payload = json.dumps({
        "model": model,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }).encode()
    req = urllib.request.Request(
        GROQ_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"].strip()


def call_api_with_retry(model: str, prompt: str, api_key: str, max_retries: int = 6) -> str:
    """Dispatch to the right API and retry on 429."""
    for attempt in range(max_retries):
        try:
            if PROVIDER == "anthropic":
                return call_anthropic(model, prompt, api_key)
            return call_groq(model, prompt, api_key)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()
            except Exception:
                pass
            if e.code == 429 and attempt < max_retries - 1:
                wait = min(15 * (attempt + 1), 60)
                with print_lock:
                    print(f"  429 rate-limit — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise RuntimeError(f"HTTP {e.code}: {body[:300]}")
    raise RuntimeError("Exhausted retries")


def strip_fences(code: str) -> str:
    lines = code.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _yaml_parse_valid(filepath: str) -> bool:
    py = shutil.which("python3.9") or "/Library/Developer/CommandLineTools/usr/bin/python3"
    yaml_check = (
        "import yaml, sys\n"
        "content = open(sys.argv[1]).read()\n"
        "docs = list(yaml.safe_load_all(content))\n"
        "sys.exit(0 if (docs and any(d is not None for d in docs)) else 1)\n"
    )
    env2 = {**ENV}
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
    with tempfile.TemporaryDirectory(prefix="groq_eval_") as tmp:
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
        issues.append("SYNTAX ERROR: The generated code has syntax errors.")
    if not validation["schema_valid"]:
        issues.append("SCHEMA ERROR: The Kubernetes manifest fails schema validation.")
    if not validation["security_compliant"]:
        issues.append(f"SECURITY: Score={validation['security_score']:.2f} (<0.5). "
                      "Fix: add resource limits, non-root user, disable privileged mode.")
    issue_str = "\n".join(issues) if issues else "Unknown validation failure."
    return (
        f"Your previous IaC output had issues:\n{issue_str}\n\n"
        f"Original request:\n{original_prompt}\n\n"
        f"Your flawed output:\n{code[:500]}\n\n"
        "Fix all issues and output ONLY the corrected code."
    )


def evaluate_task(task: dict, model: str, use_rag: bool, api_key: str, task_num: int, total: int) -> dict:
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
        "model": model,
        "use_rag": use_rag,
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
                code = strip_fences(call_api_with_retry(model, prompt, api_key))
            else:
                correction_prompt = build_correction_prompt(task["prompt"], code, validation)
                code = strip_fences(call_api_with_retry(model, correction_prompt, api_key))
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

        except Exception as e:
            result["attempts"].append({"attempt": attempt + 1, "error": str(e)})
            break

    result["final_functional"] = validation.get("functional_correct", False)
    result["final_security_score"] = validation.get("security_score", 0.0)
    result["final_syntax_valid"] = validation.get("syntax_valid", False)
    result["final_schema_valid"] = validation.get("schema_valid", False)

    status = "PASS" if result["final_functional"] else "FAIL"
    sc = f"+{result['self_corrections']}SC" if result["self_corrections"] else ""
    model_short = model.split("-")[0] + "-" + model.split("-")[2] if "-" in model else model[:10]
    cond = "RAG" if use_rag else "base"
    with print_lock:
        print(f"[{task_num:3d}/{total}] {task['task_id']} ({lang}, L{task.get('difficulty','?')}, {model_short}, {cond})... "
              f"{status}{sc} sec={result['final_security_score']:.2f}")

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


def run_condition_parallel(tasks, model, use_rag, api_key, offset, total):
    """Run all tasks for one model+condition in parallel."""
    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(evaluate_task, task, model, use_rag, api_key,
                            offset + i + 1, total): i
            for i, task in enumerate(tasks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                task = tasks[idx]
                with print_lock:
                    print(f"  ERROR on {task['task_id']}: {e}")
                results[idx] = {
                    "task_id": task["task_id"], "language": task["language"],
                    "difficulty": task.get("difficulty", ""), "category": task.get("category", ""),
                    "model": model, "use_rag": use_rag, "attempts": [{"error": str(e)}],
                    "final_functional": False, "self_corrections": 0,
                    "final_security_score": 0.0, "final_syntax_valid": False, "final_schema_valid": False,
                }
    return results


def main():
    api_key = os.environ.get(cfg["key_env"], "")
    if not api_key:
        print(f"ERROR: {cfg['key_env']} not set")
        sys.exit(1)
    print(f"Provider: {PROVIDER}  |  Key env: {cfg['key_env']}")

    tasks = load_tasks()
    total_calls = len(tasks) * len(MODELS) * 2  # baseline + RAG per model
    print(f"=== Groq Multi-Model IaCBench Evaluation ===")
    print(f"Tasks: {len(tasks)} | Models: {len(MODELS)} | Conditions: baseline+RAG | Total: {total_calls}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Parallel workers: {MAX_WORKERS} per condition\n")

    all_results = {}

    for model in MODELS:
        model_short = model.replace("llama-", "llama").replace("-versatile", "").replace("-instant", "")
        print(f"\n{'='*60}")
        print(f"MODEL: {model}")
        print(f"{'='*60}")

        baseline_results = []
        rag_results = []
        done_so_far = 0

        for condition, results_list, use_rag in [
            ("BASELINE", baseline_results, False),
            ("RAG",      rag_results,      True),
        ]:
            print(f"\n--- {condition} (parallel x{MAX_WORKERS}) ---")
            offset = done_so_far
            batch = run_condition_parallel(tasks, model, use_rag, api_key,
                                           offset, total_calls // len(MODELS))
            results_list.extend(batch)
            done_so_far += len(tasks)

        bs = compute_stats(baseline_results, f"{model}_baseline")
        rs = compute_stats(rag_results, f"{model}_rag")
        all_results[model] = {
            "baseline_stats": bs,
            "rag_stats": rs,
            "baseline_results": baseline_results,
            "rag_results": rag_results,
        }

        print(f"\n--- {model_short} Summary ---")
        print(f"  Baseline: functional={bs['pct_functional']:.1%}  security={bs['mean_security']:.3f}  SC={bs['pct_self_corrected']:.1%}")
        print(f"  RAG:      functional={rs['pct_functional']:.1%}  security={rs['mean_security']:.3f}  SC={rs['pct_self_corrected']:.1%}")

    output = {
        "meta": {
            "models": MODELS,
            "n_tasks": len(tasks),
            "max_self_corrections": MAX_SELF_CORRECTIONS,
            "temperature": TEMPERATURE,
            "parallel_workers": MAX_WORKERS,
        },
        "model_stats": {m: {
            "baseline": all_results[m]["baseline_stats"],
            "rag": all_results[m]["rag_stats"],
        } for m in MODELS},
        "model_results": {m: {
            "baseline": all_results[m]["baseline_results"],
            "rag": all_results[m]["rag_results"],
        } for m in MODELS},
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("=== FINAL COMPARISON ===")
    print(f"{'Model':<35} {'Cond':<10} {'Functional':>12} {'Security':>10}")
    print("-" * 70)
    for m in MODELS:
        bs = all_results[m]["baseline_stats"]
        rs = all_results[m]["rag_stats"]
        print(f"{m:<35} {'baseline':<10} {bs['pct_functional']:>11.1%} {bs['mean_security']:>10.3f}")
        print(f"{'':<35} {'RAG':<10} {rs['pct_functional']:>11.1%} {rs['mean_security']:>10.3f}")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
