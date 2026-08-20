#!/usr/bin/env python3
"""HumanEval pass@1 evaluator against any OpenAI-compatible endpoint.
Greedy, no-thinking. Executes generated code in a subprocess with timeout.
Usage: humaneval_eval.py --api-base http://127.0.0.1:8002/v1 --model X --api-key K --tag q4
Env: HE_N (default 164), HE_CONC (default 8)
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("QUALITY_DATA", os.path.join(HERE, "quality-data"))


def chat(base, model, key, prompt_text, max_tokens=1024):
    payload = {"model": model,
               "messages": [{"role": "user", "content":
                             prompt_text
                             + "\n\nComplete the function above. Output ONLY the completed Python code (full function including signature), no explanation, no markdown fences."}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"}, method="POST")
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    m = d["choices"][0]["message"]
    return m.get("content") or ""


def extract_code(resp, entry_point):
    resp = re.sub(r"```(?:python)?\n?", "", resp).replace("```", "")
    # find the def of entry_point; keep everything from there
    lines = resp.split("\n")
    start = None
    for i, l in enumerate(lines):
        if l.startswith(f"def {entry_point}") or l.startswith(f"def "):
            start = i
            break
    if start is None:
        return resp
    return "\n".join(lines[start:])


def run_test(code, test, entry_point, timeout=15):
    prog = code + "\n\n" + test + f"\n\ncheck({entry_point})\nprint('PASS')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout, text=True)
        return r.returncode == 0 and "PASS" in r.stdout
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    n = int(os.environ.get("HE_N", "164"))
    conc = int(os.environ.get("HE_CONC", "8"))

    t = pq.read_table(f"{DATA}/humaneval_test.parquet")
    rows = t.to_pylist()[:n]
    print(f"[{args.tag}] HumanEval n={len(rows)} conc={conc}", flush=True)

    def work(rec):
        try:
            resp = chat(args.api_base, args.model, args.api_key, rec["prompt"])
            code = extract_code(resp, rec["entry_point"])
            ok = run_test(code, rec["test"], rec["entry_point"])
            return {"task_id": rec["task_id"], "ok": bool(ok)}
        except Exception as e:
            return {"task_id": rec["task_id"], "ok": False, "err": str(e)[:120]}

    t0 = time.time()
    with ThreadPoolExecutor(conc) as ex:
        results = list(ex.map(work, rows))
    npass = sum(1 for r in results if r["ok"])
    out = os.path.join(HERE, f"humaneval_{args.tag}.jsonl")
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[{args.tag}] HumanEval pass@1 = {npass}/{len(results)} = {npass/len(results):.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
