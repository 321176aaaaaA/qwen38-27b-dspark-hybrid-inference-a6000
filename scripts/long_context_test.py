#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Long-context stability test for DSpark vLLM (50K/100K/200K/220K)."""
import argparse
import json
import subprocess
import time
import urllib.request

BASE = "http://127.0.0.1:8002/v1"
MODEL = "qwen3.8-27b-dspark"
API_KEY = "sk-your-api-key"
SEGMENT = "The quick brown fox jumps over the lazy dog. "


def chat(prompt, max_tokens=32, timeout=1800):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
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
    content = data["choices"][0]["message"].get("content") or ""
    return {
        "elapsed": elapsed,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "output_chars": len(content),
        "ok": True,
    }


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
    ap.add_argument("--targets", type=int, nargs="+", default=[50000, 100000, 200000, 220000])
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    print("calibrating segment token count...", flush=True)
    cal_prompt = SEGMENT * 500
    cal = chat(cal_prompt, max_tokens=4)
    cal_tokens = cal["prompt_tokens"]
    repeats = 500
    tok_per_rep = cal_tokens / repeats
    print(f"calibration: prompt_tokens={cal_tokens} repeats={repeats} "
          f"tok_per_rep={tok_per_rep:.3f}", flush=True)

    for target in args.targets:
        n_rep = max(1, int(target / tok_per_rep))
        prompt = SEGMENT * n_rep
        print(f"testing target={target} repeats={n_rep} approx_chars={len(prompt)} "
              f"gpu_before=[{gpu_mem()}]", flush=True)
        t0 = time.time()
        try:
            r = chat(prompt, max_tokens=args.max_tokens, timeout=2400)
            wall = time.time() - t0
            print(f"RESULT target={target} prompt_tokens={r['prompt_tokens']} "
                  f"completion_tokens={r['completion_tokens']} output_chars={r['output_chars']} "
                  f"elapsed={r['elapsed']:.2f}s wall={wall:.2f}s ok={r['ok']} "
                  f"gpu_after=[{gpu_mem()}]", flush=True)
        except Exception as e:
            wall = time.time() - t0
            print(f"RESULT target={target} ERROR={e} wall={wall:.2f}s "
                  f"gpu_after=[{gpu_mem()}]", flush=True)
            raise

    print("LONG_CONTEXT_DONE", flush=True)


if __name__ == "__main__":
    main()
