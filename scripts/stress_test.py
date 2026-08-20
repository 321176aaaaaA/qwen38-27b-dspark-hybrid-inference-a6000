#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Long-running stress test for DSpark vLLM on GPU2."""
import argparse
import json
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:8002/v1"
MODEL = "qwen3.8-27b-dspark"
API_KEY = "sk-your-api-key"

stop = threading.Event()
lock = threading.Lock()
stats = {"requests": 0, "tokens": 0, "errors": 0, "total_time": 0.0}


def chat_once(max_tokens):
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
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    usage = data.get("usage", {})
    with lock:
        stats["requests"] += 1
        stats["tokens"] += usage.get("completion_tokens", 0)
        stats["total_time"] += elapsed


def worker(max_tokens):
    while not stop.is_set():
        try:
            chat_once(max_tokens)
        except Exception as e:
            with lock:
                stats["errors"] += 1
            if stats["errors"] < 10:
                print(f"ERROR: {e}", flush=True)
            time.sleep(1.0)


def gpu_mem():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()
        return "; ".join(out)
    except Exception as e:
        return f"nvidia-smi error: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=3600.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--window", type=float, default=60.0)
    args = ap.parse_args()

    print(f"stress start duration={args.duration}s concurrency={args.concurrency} "
          f"max_tokens={args.max_tokens} window={args.window}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, args.max_tokens) for _ in range(args.concurrency)]

        start = time.time()
        last = start
        last_req = 0
        last_tok = 0
        last_err = 0
        while time.time() - start < args.duration:
            time.sleep(args.window)
            now = time.time()
            with lock:
                req = stats["requests"]
                tok = stats["tokens"]
                err = stats["errors"]
            dwall = now - last
            dtok = tok - last_tok
            dreq = req - last_req
            derr = err - last_err
            print(f"[{now-start:7.1f}s] req={dreq:4d} tok={dtok:6d} "
                  f"tok/s={dtok/dwall:7.2f} errors={derr:3d} | gpu: {gpu_mem()}", flush=True)
            last = now
            last_req = req
            last_tok = tok
            last_err = err

        stop.set()
        for f in futs:
            f.result(timeout=5)

    with lock:
        total_req = stats["requests"]
        total_tok = stats["tokens"]
        total_err = stats["errors"]
        total_time = stats["total_time"]
    wall = time.time() - start
    print(f"stress done wall={wall:.1f}s requests={total_req} tokens={total_tok} "
          f"errors={total_err} aggregate_tok/s={total_tok/wall:.2f} "
          f"avg_latency={total_time/max(total_req,1):.3f}s", flush=True)


if __name__ == "__main__":
    main()
