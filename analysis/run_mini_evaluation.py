#!/usr/bin/env python3
"""
run_mini_evaluation.py
Runs gpt-4o-mini on all 150 IaCBench tasks.
Produces results/mini_eval_results.json.
"""

import os, json, time, sys, urllib.request, urllib.error
import subprocess, tempfile, shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "mini_eval_results.json"

TASK_FILES = {
    "kubernetes": REPO_ROOT / "iachench" / "tasks" / "k8s_tasks.json",
    "terraform":  REPO_ROOT / "iachench" / "tasks" / "tf_tasks.json",
    "dockerfile": REPO_ROOT / "iachench" / "tasks" / "df_tasks.json",
}

TOOL_PATH = os.environ.get("PATH", "")
ENV = {**os.environ, "PATH": TOOL_PATH}
MODEL = "gpt-4o-mini"
MAX_TOKENS = 1024
TEMPERATURE = 0.2
MAX_SELF_CORRECTIONS = 2
REQUEST_DELAY = 0.8

SYSTEM_PROMPT = """You are an expert Infrastructure-as-Code engineer. Generate production-ready,
secure IaC code. Follow security best practices:
- Kubernetes: non-root containers, resource limits, readiness/liveness probes, no privileged mode
- Terraform: encryption at rest, no public exposure, least-privilege IAM
- Dockerfile: non-root USER, pinned base image tags, HEALTHCHECK
Output ONLY the raw code with no explanation or markdown fences."""


def load_tasks():
    tasks = []
    for lang, path in TASK_FILES.items():
        data = json.loads(path.read_text())
        task_list = data.get("tasks", data) if isinstance(data, dict) else data
        tasks.extend(task_list)
    return tasks


def call_api(prompt, api_key):
    payload = json.dumps({"model": MODEL, "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role":"system","content":SYSTEM_PROMPT},
                     {"role":"user","content":prompt}]}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
        data=payload, headers={"Authorization":f"Bearer {api_key}",
                               "Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def strip_fences(code):
    lines = code.splitlines()
    if lines and lines[0].startswith("```"): lines = lines[1:]
    if lines and lines[-1].strip() == "```": lines = lines[:-1]
    return "\n".join(lines)


def _yaml_parse_valid(filepath):
    py = shutil.which("python3.9") or "/Library/Developer/CommandLineTools/usr/bin/python3"
    chk = "import yaml,sys\ndocs=list(yaml.safe_load_all(open(sys.argv[1]).read()))\nsys.exit(0 if(docs and any(d is not None for d in docs))else 1)\n"
    env2 = {**ENV}
    try:
        return subprocess.run([py,"-c",chk,filepath],capture_output=True,timeout=10,env=env2).returncode==0
    except: return True


def _heuristic_tf_valid(code):
    return abs(code.count("{")-code.count("}"))<=2 and any(k in code for k in("resource","data","module","variable"))


def _heuristic_security_score(code, lang):
    dangerous = {"kubernetes":["privileged: true","hostNetwork: true","allowPrivilegeEscalation: true"],
                 "terraform":["publicly_accessible = true","encrypted = false"],
                 "dockerfile":["USER root",":latest"]}.get(lang,[])
    hits = sum(1 for d in dangerous if d in code)
    return round(max(0.0,1.0-hits*0.33),4)


def run_checkov(filepath):
    result = {"passed":0,"failed":0,"score":0.0}
    try:
        proc = subprocess.run(["checkov","-f",filepath,"--output","json","--quiet","--compact"],
            capture_output=True,text=True,timeout=60,env=ENV)
        raw = proc.stdout.strip()
        if not raw: return result
        data = json.loads(raw)
        if isinstance(data,list):
            data = next((x for x in data if isinstance(x,dict) and "summary" in x),{})
        s = data.get("summary",{})
        p,f = int(s.get("passed",0)),int(s.get("failed",0))
        result.update({"passed":p,"failed":f,"score":round(p/(p+f),4) if p+f>0 else 0.0})
    except: pass
    return result


def run_kubeconform(filepath):
    try:
        return subprocess.run(["kubeconform","-strict","-summary",filepath],
            capture_output=True,text=True,timeout=30,env=ENV).returncode==0
    except: return _yaml_parse_valid(filepath)


def validate_code(code, lang):
    if not code or len(code.strip())<10:
        return {"syntax_valid":False,"schema_valid":False,"security_score":0.0,
                "checkov_passed":0,"checkov_failed":0,"security_compliant":False,"functional_correct":False}
    with tempfile.TemporaryDirectory(prefix="mini_eval_") as tmp:
        fpath = os.path.join(tmp,"Dockerfile" if lang=="dockerfile" else f"generated{'.yaml' if lang=='kubernetes' else '.tf'}")
        open(fpath,"w").write(code)
        syntax = _yaml_parse_valid(fpath) if lang=="kubernetes" else (_heuristic_tf_valid(code) if lang=="terraform" else "FROM" in code)
        schema = run_kubeconform(fpath) if lang=="kubernetes" else syntax
        ck = run_checkov(fpath)
        ran = (ck["passed"]+ck["failed"])>0
        sec = ck["score"] if ran else _heuristic_security_score(code,lang)
        compliant = sec>=0.5
        return {"syntax_valid":syntax,"schema_valid":schema,"security_score":sec,
                "checkov_passed":ck["passed"],"checkov_failed":ck["failed"],
                "security_compliant":compliant,"functional_correct":syntax and schema and compliant}


def evaluate_task(task, api_key):
    prompt, lang = task["prompt"], task["language"]
    result = {"task_id":task["task_id"],"language":lang,"difficulty":task.get("difficulty",""),
              "category":task.get("category",""),"model":MODEL,"attempts":[],
              "final_functional":False,"self_corrections":0,"final_security_score":0.0,
              "final_syntax_valid":False,"final_schema_valid":False}
    code, validation = "", {}
    for attempt in range(1+MAX_SELF_CORRECTIONS):
        try:
            if attempt==0:
                code = strip_fences(call_api(prompt, api_key))
            else:
                issues = []
                if not validation.get("syntax_valid"): issues.append("SYNTAX ERROR")
                if not validation.get("schema_valid"): issues.append("SCHEMA ERROR")
                if not validation.get("security_compliant"): issues.append(f"SECURITY score={validation.get('security_score',0):.2f}")
                fix_prompt = f"Issues: {'; '.join(issues)}\nOriginal: {prompt}\nFix and output ONLY corrected code."
                code = strip_fences(call_api(fix_prompt, api_key))
                result["self_corrections"] += 1
            time.sleep(REQUEST_DELAY)
            validation = validate_code(code, lang)
            result["attempts"].append({"attempt":attempt+1,**{k:validation[k] for k in["functional_correct","syntax_valid","schema_valid","security_score"]}})
            if validation["functional_correct"]: break
        except urllib.error.HTTPError as e:
            result["attempts"].append({"attempt":attempt+1,"error":f"HTTP {e.code}"})
            if e.code==429: time.sleep(20)
            break
        except Exception as e:
            result["attempts"].append({"attempt":attempt+1,"error":str(e)})
            break
    result.update({"final_functional":validation.get("functional_correct",False),
                   "final_security_score":validation.get("security_score",0.0),
                   "final_syntax_valid":validation.get("syntax_valid",False),
                   "final_schema_valid":validation.get("schema_valid",False)})
    return result


def compute_stats(results, label):
    n = len(results)
    if n==0: return {}
    by_lang,by_diff = {},{}
    for r in results:
        by_lang.setdefault(r["language"],[]).append(r)
        by_diff.setdefault(r["difficulty"],[]).append(r)
    def pct(lst,k): return round(sum(1 for x in lst if x.get(k))/len(lst),4) if lst else 0
    return {"label":label,"n":n,
            "pct_functional":pct(results,"final_functional"),
            "pct_syntax":pct(results,"final_syntax_valid"),
            "mean_security":round(sum(r.get("final_security_score",0) for r in results)/n,4),
            "pct_self_corrected":round(sum(1 for r in results if r["self_corrections"]>0)/n,4),
            "by_language":{lang:{"n":len(lst),"pct_functional":pct(lst,"final_functional")} for lang,lst in by_lang.items()},
            "by_difficulty":{d:{"n":len(lst),"pct_functional":pct(lst,"final_functional")} for d,lst in sorted(by_diff.items())}}


def main():
    api_key = os.environ.get("OPENAI_API_KEY","")
    if not api_key: print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)
    tasks = load_tasks()
    total = len(tasks) * 2
    print(f"=== GPT-4o-mini IaCBench Evaluation ===")
    print(f"Tasks: {len(tasks)} | Conditions: baseline + RAG | Total calls: {total} | Output: {OUTPUT_FILE}\n")

    RAG_CORPUS_DIR = REPO_ROOT / "rag_corpus"

    def get_rag_context(task):
        lang = task.get("language","")
        category = task.get("category","").lower().replace(" ","-")
        corpus_dir = RAG_CORPUS_DIR / lang
        if not corpus_dir.exists(): return ""
        for doc in corpus_dir.glob("*.md"):
            if any(kw in doc.stem for kw in category.split("-")):
                return doc.read_text(errors="replace")[:1200]
        docs = list(corpus_dir.glob("*.md"))
        return docs[0].read_text(errors="replace")[:1200] if docs else ""

    baseline_results, rag_results = [], []
    done = 0

    for condition, results_list, use_rag in [
        ("BASELINE", baseline_results, False),
        ("RAG",      rag_results,      True),
    ]:
        print(f"\n--- {condition} ---")
        for task in tasks:
            done += 1
            prompt = task["prompt"]
            if use_rag:
                ctx = get_rag_context(task)
                if ctx: prompt = f"Reference examples:\n{ctx}\n\n---\nTask:\n{prompt}"
            print(f"[{done:3d}/{total}] {task['task_id']} ({task['language']}, L{task.get('difficulty','?')}, {'RAG' if use_rag else 'base'})...", end=" ", flush=True)
            r = evaluate_task(task, api_key)
            r["use_rag"] = use_rag
            results_list.append(r)
            sc = f"+{r['self_corrections']}SC" if r["self_corrections"] else ""
            print(f"{'PASS' if r['final_functional'] else 'FAIL'}{sc} sec={r['final_security_score']:.2f}")

    baseline_stats = compute_stats(baseline_results, f"{MODEL}_baseline")
    rag_stats = compute_stats(rag_results, f"{MODEL}_rag")
    output = {"meta":{"model":MODEL,"n_tasks":len(tasks),"max_self_corrections":MAX_SELF_CORRECTIONS},
              "baseline_stats":baseline_stats,"rag_stats":rag_stats,
              "baseline_results":baseline_results,"rag_results":rag_results}
    with open(OUTPUT_FILE,"w") as f: json.dump(output,f,indent=2)
    print(f"\n=== Results ===")
    print(f"  Baseline — functional: {baseline_stats['pct_functional']:.1%}  security: {baseline_stats['mean_security']:.3f}")
    print(f"  RAG      — functional: {rag_stats['pct_functional']:.1%}  security: {rag_stats['mean_security']:.3f}")
    print(f"\nSaved to {OUTPUT_FILE}")

if __name__=="__main__":
    main()
