#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate IFBench subset responses from the local DSpark server."""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:8002/v1"
MODEL = "qwen3.8-27b-dspark"
API_KEY = "sk-your-api-key"


def chat(prompt, max_tokens=512, timeout=600):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    content = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage", {})
    return {
        "prompt": prompt,
        "response": content,
        "completion_tokens": usage.get("completion_tokens", 0),
        "elapsed": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/opt/IFBench/data/IFBench_partial_120.jsonl")
    ap.add_argument("--output", default="/opt/dspark_dev_vllm/quality_ifbench_subset.jsonl")
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    prompts = []
    with open(args.input, "r") as f:
        for i, line in enumerate(f):
            if i >= args.num:
                break
            prompts.append(json.loads(line)["prompt"])

    print(f"generating {len(prompts)} responses", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(chat, p, args.max_tokens) for p in prompts]
        for fut in futs:
            results.append(fut.result())
    results.sort(key=lambda r: prompts.index(r["prompt"]))

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps({"prompt": r["prompt"], "response": r["response"]}) + "\n")
    total_tok = sum(r["completion_tokens"] for r in results)
    total_time = sum(r["elapsed"] for r in results)
    print(f"wrote {args.output} total_tokens={total_tok} total_time={total_time:.2f}s", flush=True)


if __name__ == "__main__":
    main()
