#!/usr/bin/env python3
"""Quality battery against the running 164 server (adapted from syv-ai/qwen38-27b-rtx3090).
Run after every kernel/quant change. Catches "benchmarks great, outputs garbage".
  1. perplexity over ~300-token windows: wikitext-2 test (en), fineweb-2 dan test (da),
     local vllm python source (code)
  2. GSM8K exact-match, N test questions, thinking off, greedy
Env: VLLM_API, VLLM_API_KEY, VLLM_MODEL, QUALITY_DATA, GSM_N, GSM_CONC, CODE_SRC
Usage: python quality_battery_164.py <tag> [--ppl-only] [--gsm-only]
"""
import json, os, sys, glob, re, math, time, random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.environ["VLLM_API_KEY"]
API = os.environ.get("VLLM_API", "http://127.0.0.1:8002/v1")
MODEL = os.environ.get("VLLM_MODEL", "qwen3.8-27b-dspark")
Q = os.environ.get("QUALITY_DATA", os.path.join(HERE, "quality-data"))
CODE_SRC = os.environ.get("CODE_SRC", "/opt/dspark_dev_vllm/vllm/v1/core")
GSM_N = int(os.environ.get("GSM_N", "200"))
GSM_CONC = int(os.environ.get("GSM_CONC", "8"))

tag = sys.argv[1]
ppl_only = "--ppl-only" in sys.argv
gsm_only = "--gsm-only" in sys.argv


def post(path, payload, timeout=1200):
    req = urllib.request.Request(API + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def docs():
    out = []
    t = pq.read_table(f"{Q}/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet").column("text").to_pylist()
    txt = "".join(t)
    for i in range(0, min(len(txt), 40 * 1200), 1200):
        out.append(("en", txt[i:i + 1200]))
    tb = pq.read_table(f"{Q}/fineweb2/data/dan_Latn/test/000_00000.parquet", columns=["text"]).column("text").to_pylist()
    random.Random(0).shuffle(tb)
    n = 0
    for d in tb:
        if len(d) > 1500:
            out.append(("da", d[:1200])); n += 1
        if n >= 40:
            break
    files = sorted(glob.glob(os.path.join(CODE_SRC, "*.py")))
    for f in files:
        s = open(f).read()
        for i in range(0, min(len(s), 4 * 1200), 1200):
            if len(s[i:i + 1200]) > 800:
                out.append(("code", s[i:i + 1200]))
        if sum(1 for l, _ in out if l == "code") >= 40:
            break
    return out


def ppl_one(item):
    lang, text = item
    r = post("/completions", {"model": MODEL, "prompt": text, "max_tokens": 1,
                              "temperature": 0, "prompt_logprobs": 0, "echo": False})
    pl = r["choices"][0]["prompt_logprobs"]
    lps = []
    for e in pl[1:]:
        if e is None:
            continue
        v = list(e.values())[0]
        lps.append(v["logprob"] if isinstance(v, dict) else v)
    return lang, sum(lps), len(lps)


def run_ppl():
    items = docs()
    with ThreadPoolExecutor(2) as ex:
        res = list(ex.map(ppl_one, items))
    agg = {}
    for lang, s, n in res:
        a = agg.setdefault(lang, [0.0, 0]); a[0] += s; a[1] += n
    out = {lang: (math.exp(-s / n), n) for lang, (s, n) in agg.items()}
    tot_s = sum(s for s, n in agg.values()); tot_n = sum(n for s, n in agg.values())
    out["all"] = (math.exp(-tot_s / tot_n), tot_n)
    return out


def extract_num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return m[-1].replace(",", "") if m else None


def gsm_one(row):
    q, a = row
    gold = a.split("####")[-1].strip().replace(",", "")
    r = post("/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": q + "\n\nSolve step by step, then give the final answer as 'Final answer: <number>'."}],
        "max_tokens": 768, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    txt = r["choices"][0]["message"]["content"] or ""
    m = re.search(r"Final answer:\s*\**\s*\$?(-?[\d,]*\.?\d+)", txt)
    pred = (m.group(1).replace(",", "") if m else extract_num(txt))
    try:
        ok = abs(float(pred) - float(gold)) < 1e-6
    except Exception:
        ok = False
    return ok, r["usage"]["completion_tokens"]


def run_gsm():
    t = pq.read_table(f"{Q}/gsm8k/main/test-00000-of-00001.parquet")
    rows = list(zip(t.column("question").to_pylist(), t.column("answer").to_pylist()))[:GSM_N]
    with ThreadPoolExecutor(GSM_CONC) as ex:
        res = list(ex.map(gsm_one, rows))
    acc = sum(1 for ok, _ in res if ok) / len(res)
    toks = sum(n for _, n in res) / len(res)
    return acc, toks


t0 = time.time(); result = {"tag": tag, "model": MODEL, "api": API}
if not gsm_only:
    result["ppl"] = run_ppl()
    print(tag, "PPL", {k: (round(v[0], 4), v[1]) for k, v in result["ppl"].items()}, f"{time.time()-t0:.0f}s", flush=True)
if not ppl_only:
    t1 = time.time(); acc, toks = run_gsm()
    result["gsm8k"] = {"n": GSM_N, "acc": acc, "mean_tokens": toks}
    print(tag, f"GSM8K n={GSM_N} acc={acc:.3f} mean_tokens={toks:.0f} {time.time()-t1:.0f}s", flush=True)
os.makedirs(Q, exist_ok=True)
with open(os.path.join(Q, f"result_{tag}.json"), "w") as f:
    json.dump(result, f)
