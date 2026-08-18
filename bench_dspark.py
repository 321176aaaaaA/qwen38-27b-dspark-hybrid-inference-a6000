#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark DSpark vLLM via OpenAI-compatible API on GPU2."""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

BASE = "http://127.0.0.1:8002/v1"
MODEL = "qwen3.8-27b-dspark"
API_KEY = "sk-qwen38-dspark-dev"


def chat_once(max_tokens=256, timeout=600):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Write a short paragraph about AI."}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "ignore_eos": True,
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
    usage = data.get("usage", {})
    return {
        "elapsed": elapsed,
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
    }


def run_round(concurrency, max_tokens):
    barrier = Barrier(concurrency)
    results = []

    def worker(_):
        barrier.wait()
        return chat_once(max_tokens=max_tokens)

    start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(worker, i) for i in range(concurrency)]
        results = [f.result() for f in futs]
    wall = time.time() - start
    total_tok = sum(r["completion_tokens"] for r in results)
    tokps = total_tok / wall if wall > 0 else 0.0
    per_req = [r["elapsed"] for r in results]
    print(f"concurrency={concurrency} round wall={wall:.3f}s total_tokens={total_tok} tok/s={tokps:.2f}")
    print(f"  per-request elapsed={per_req}")
    return tokps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 3, 8])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    # Warmup
    try:
        chat_once(max_tokens=16)
        print("warmup OK")
    except Exception as e:
        print("warmup FAIL", e)
        raise

    for c in args.concurrency:
        rates = []
        for r in range(args.rounds):
            rates.append(run_round(c, args.max_tokens))
        avg = sum(rates) / len(rates)
        print(f"SUMMARY concurrency={c}: avg {avg:.2f} tok/s rates={rates}")
    print("DONE")


if __name__ == "__main__":
    main()
