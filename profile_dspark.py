#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a short DSpark decode profile with torch profiler via LLM API."""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_CACHE_ROOT", "/mnt/6/wangzihan/vllm_cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/mnt/6/wangzihan/triton_cache")
os.environ.setdefault("HOME", "/mnt/6/wangzihan")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "/mnt/6/wangzihan/qwen38_27b/model_autoread"
DRAFT = "/mnt/6/wangzihan/Doopeworld_Qwen3.8-27B-DSpark-vLLM"
PROF_DIR = "/mnt/6/wangzihan/dspark_dev_vllm/prof_trace"

def main():
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        kv_cache_dtype="int8_per_token_head",
        speculative_config={
            "method": "dspark",
            "model": DRAFT,
            "num_speculative_tokens": 7,
            "draft_sample_method": "probabilistic",
        },
        compilation_config={
            "cudagraph_capture_sizes": [1, 8, 16, 24, 32, 40, 48, 56],
            "max_cudagraph_capture_size": 56,
        },
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": PROF_DIR,
            "torch_profiler_with_stack": False,
            "torch_profiler_use_gzip": False,
            "ignore_frontend": True,
        },
    )

    prompt = "Write a short paragraph about AI."
    warm_params = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
    main_params = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)

    # Warmup / compile/cudagraph already done at init; run a tiny generation to
    # exercise steady-state before starting the profiler.
    llm.generate([prompt], warm_params, use_tqdm=False)
    print("warmup done", flush=True)

    llm.start_profile("decode8")
    try:
        outputs = llm.generate([prompt] * 8, main_params, use_tqdm=False)
        for o in outputs:
            print(f"prompt_tokens={len(o.prompt_token_ids)} output_tokens={len(o.outputs[0].token_ids)}", flush=True)
    finally:
        llm.stop_profile()
    print("profile saved under", PROF_DIR, flush=True)


if __name__ == "__main__":
    main()
