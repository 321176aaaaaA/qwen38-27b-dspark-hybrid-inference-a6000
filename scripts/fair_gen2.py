#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fair IFBench response generation v2: incremental writes + resume.

- Appends each finished result to --output immediately (survives kill).
- On start, skips prompts already present in --output (resume).
- Otherwise same protocol as fair_gen_203.py.

Usage:
  python fair_gen2.py --api-base http://127.0.0.1:8002/v1 \
      --model qwen3.8-27b-dspark --api-key sk-qwen38-dspark-dev \
      --input /mnt/6/wangzihan/IFBench/data/IFBench_test.jsonl \
      --output out.jsonl --max-tokens 32768 --temperature 1.0 --thinking \
      --concurrency 6
"""
import argparse
import json
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

write_lock = threading.Lock()


def chat(base, model, api_key, prompt, max_tokens, temperature, thinking, timeout=1800):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"].get("content") or ""


def gen_one(args, idx, prompt, outf):
    last_err = None
    for attempt in range(args.retries):
        try:
            t0 = time.time()
            content = chat(args.api_base, args.model, args.api_key, prompt,
                           args.max_tokens, args.temperature, args.thinking).strip()
            if content.strip():
                rec = {"idx": idx, "prompt": prompt, "response": content,
                       "elapsed": round(time.time() - t0, 2), "error": None}
                with write_lock:
                    outf.write(json.dumps({"prompt": prompt, "response": content}) + "\n")
                    outf.flush()
                return rec
            last_err = "empty response"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(5 * (attempt + 1))
    # permanent error: record placeholder so resume does not retry blindly
    with write_lock:
        outf.write(json.dumps({"prompt": prompt, "response": ""}) + "\n")
        outf.flush()
    return {"idx": idx, "prompt": prompt, "response": "", "elapsed": -1, "error": last_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.set_defaults(thinking=False)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="0 = all prompts")
    args = ap.parse_args()

    prompts = []
    with open(args.input) as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
    if args.limit > 0:
        prompts = prompts[: args.limit]

    done_prompts = set()
    try:
        with open(args.output) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if (d.get("response") or "").strip():
                        done_prompts.add(d["prompt"])
                except Exception:
                    continue
    except FileNotFoundError:
        pass

    todo = [(i, p) for i, p in enumerate(prompts) if p not in done_prompts]
    print(f"[fair_gen2] {len(prompts)} prompts ({len(done_prompts)} already done, "
          f"{len(todo)} todo), model={args.model}, temp={args.temperature}, "
          f"max_tokens={args.max_tokens}, thinking={args.thinking}, conc={args.concurrency}",
          flush=True)

    outf = open(args.output, "a")
    t0 = time.time()
    n_err = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(gen_one, args, i, p, outf) for i, p in todo]
        done = 0
        for fut in futs:
            r = fut.result()
            done += 1
            if r["error"]:
                n_err += 1
                print(f"[fair_gen2] idx={r['idx']} PERMANENT_ERROR {r['error']}", flush=True)
            if done % 10 == 0:
                print(f"[fair_gen2] progress {done}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    outf.close()
    print(f"[fair_gen2] DONE todo={len(todo)} errors={n_err} wall={time.time()-t0:.0f}s -> {args.output}",
          flush=True)
    if n_err:
        sys.exit(2)


if __name__ == "__main__":
    main()
