# Qwen3.8-27B DSpark — Hybrid-Precision Inference on A6000

**[中文文档](README.zh-CN.md)** | English

This repository publishes production hybrid-precision checkpoints (Q4 W4A16 release +
M1 down_proj-8bit hybrid variant) for Qwen3.8-27B DSpark, plus the companion vLLM
optimization patch set, tuned for NVIDIA A6000 (sm_86).

> Built on top of [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)
> (model + evaluation framework), extended with quantization / speculative-decoding / kernel-level
> optimizations for the A6000 (sm_86). Kudos to the upstream authors.

## Checkpoint: model_mtp_opt

Based on Qwen3.8-27B (hybrid architecture: 48× linear attention + 16× full attention,
interval=4, 64 layers):

- Backbone **W4A16** pack-quantized (int4 g128 symmetric; lm_head/embed int8 g128)
- MTP layer **int8 g128** (round-trip rel err 0.66–1.53%)
- MTP draft head with **40960-token vocabulary** (draft_vocab_ids from the model's own output statistics)
- `mtp_draft_vocab_ids.pt` ships with git (329KB)

Release `v1.0.0-q4-w4a16` assets contain 10 safetensors shards (≤2GB each, GitHub limit).
Download all shards and place them together with the small files from this repo
(config/tokenizer/index) in one directory to load.

## Benchmarks (test server, A6000 48GB sm_86, vLLM 0.26 + patches/)

### Q4 final stack (this release checkpoint + all patches)

| Metric | Value |
|---|---|
| Single-stream decode | 93.3 tok/s |
| 8-way concurrency | 517.5 tok/s |
| Long output (10496 tok) single / 8-way | 91.8 / 476.9 tok/s |
| PPL all (wikitext2-en/fineweb2-da/code 33k tok) | 8.0845 (historical BF16 ref 8.045) |
| GSM8K-200 | 94.0% |
| HumanEval-164 pass@1 | 79.9% |
| IFBench-300 thinking (60K budget, xhigh, temp1.0/top-p0.95/top-k20) | strict 72.7 / loose 79.3 |
| Weight memory | ~15 GB |

IFBench protocol note: no-think 512-token protocol scores only 0.40 (insufficient budget);
47% of xhigh thinking chains exceed 8192 tokens. 32K budget produced 7 empty responses
(strict 71.3); 60K budget has 0 empty responses (strict 72.7 / loose 79.3).

### M1 hybrid precision (down_proj→int8, experimental quality-first variant)

Backbone stays 4bit; all 64 `down_proj` layers (residual-stream injection points, the most
quantization-sensitive layer class) are upgraded to int8 g128. Bandwidth budget 17GB/step,
still above the 80 tok/s single-stream bar.

| Metric | M1 | Q4 | W8A16 |
|---|---|---|---|
| Single / 8-way tok/s | **85.0 / 475.0** | 93.3 / 517.5 | 52.4 / 337.9 |
| PPL all | 8.0242 | 8.0845 | 7.7681 |
| GSM8K-200 | **95.5%** | 94.0% | 95.5% |
| HumanEval-164 pass@1 | **81.1%** | 79.9% | 79.9% |
| MTP acceptance length | **3.07** | 2.90 | 2.82 |
| IFBench fail82 subset strict (the 82 prompts Q4 failed) | **24/82 = 29.3%** | 0/82 | FP8 tier 17/82 = 20.7% |

Build script: `scripts/build_m1_hybrid.py` (hardlink copy + requantize down_proj from BF16 +
prepend `group_d8` in config_groups).

### Quantization variants compared (same machine, same stack)

| Variant | Weight mem | Single tok/s | 8-way tok/s | PPL all | GSM8K | HumanEval |
|---|---|---|---|---|---|---|
| **Q4 W4A16 + MTP4 (this release)** | ~15 GB | **93.3** | **517.5** | 8.0845 | 94.0% | 79.9% |
| **M1 hybrid + MTP4 (experimental)** | ~17 GB | 85.0 | 475.0 | 8.0242 | **95.5%** | **81.1%** |
| W8A16 + MTP4 (experimental) | ~28 GB | 52.4 | 337.9 | 7.7681 | 95.5% | 79.9% |
| FP8 official dynamic + MTP4 | ~29 GB | 45.1 | 300.3 | 7.792 | 95.5% | — |
| Q8_0 GGUF (llama.cpp + MTP) | ~29 GB | 22.4 | 65.6 | 6.55* | 96.5% | — |

*llama.cpp sliding-window PPL; not directly comparable to the vLLM completion protocol.

Single-stream decode scales strictly with weight bytes per step (1430 tok·GB/s constant:
93.3×15.3 ≈ 52.4×27.3) — single-stream on sm_86 is fully bandwidth-bound; FP8 has no
tensor-core acceleration on sm_86.

### KV capacity & long context

- Max per-request context: 262144 tokens
- Full-attention KV per token: bf16 64 KB / int8 32 KB
- Total KV capacity = GPU 16.5 GB (bf16, 257k tok) + CPU offload 96 GB ≈ 1.75M tokens
- Beyond-GPU-memory concurrency (6×100k tok measured): requests serialize via queueing,
  0 preemptions 0 crashes, decode stays at 88–122 tok/s
- Offload revisit TTFT 3.2–4 s vs 81 s cold (25×)

## Launch command (final form)

```bash
vllm serve <checkpoint_dir> \
  --kv-cache-dtype auto \                 # bf16 KV (pairs with spec-decode-attn)
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}' \
  --mamba-ssm-cache-dtype float16 \
  --max-num-seqs 12 --max-num-batched-tokens 2048 \
  --kv-offloading-size 96 --kv-offloading-backend native \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "xhigh"}'
# env: VLLM_SPEC_DECODE_ATTN=1
```

Full launcher: `scripts/start_gpu2_exp.sh` (dirty-GPU cleanup, cudagraph tiers, all knobs).

## patches/ (dev vLLM 0.26 patch set, COMPILE_OK verified)

1. `PR50021` GDN spec-decode kernel out-of-bounds fix (4 kernel files; **without it, long
   thinking outputs degrade / the engine crashes**)
2. Sampler small-topk sort-free + multi-block row softmax + draft same-truncation sampling
   (`VLLM_DRAFT_TOPK_TOPP=1`, on by default)
3. MTP 40k draft vocab head (auto-enabled when checkpoint contains `mtp_draft_vocab_ids.pt`;
   `MTP_DRAFT_VOCAB=0` to disable)
4. spec-decode-attn split-KV verify attention (`VLLM_SPEC_DECODE_ATTN=1`, bf16 KV only,
   +31% on long outputs)
5. marlin int8 negative-scale sign fix + layer-selection regex
   (`VLLM_MARLIN_INT8_INCLUDE_RE/EXCLUDE_RE`)

## Ops lessons learned

- Benchmarks don't reveal garbage output (int8 Marlin sign-bug lesson) — always run the
  quality gate (PPL/GSM8K) after changes
- Restarting on a dirty GPU silently costs 25% throughput — the launcher guards against it
- Random-token benchmarks overestimate speculative gains — use real prompts
- Draft vocabulary must come from the model's own output statistics (+10% single largest speedup)
- thinking evaluation needs max_tokens ≥32768 (47% of xhigh chains exceed 8192)

## Conclusions (A6000)

Q4 W4A16 is the best speed/quality/memory balance:

- **FP8**: slightly better quality (GSM8K +1.5pp) but no FP8 tensor cores on sm_86 — half the speed (45 vs 93 tok/s)
- **W8A16**: FP8-level quality (PPL 7.77), faster than FP8 (+16%/+12%), but single-stream only 56% of Q4
  (bandwidth-bound, 2× weights); Marlin int8 tensor cores only pay off at 8+ concurrency
- **Q8_0 GGUF**: best GSM8K (96.5%) but llama.cpp engineering performance is far behind (22 tok/s single, 65.6 at 8-way)
- Q4→W8 quality delta: PPL +4%, GSM8K −1.5pp, HumanEval tied
- **M1 hybrid** recovers most of the quality delta at 91% of Q4's speed — the quality-first pick

## License

- **Code** (patches, scripts, benchmarks in this repo): [Apache License 2.0](LICENSE),
  consistent with upstream [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090).
- **Model weights**: the checkpoint derives from Qwen3.8-27B (DSpark variant); use of the
  weights is subject to the original model's license terms. This repository does not grant
  any additional rights over the weights.

## Acknowledgments

- [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) — base model and evaluation framework
- vLLM / compressed-tensors / Marlin — inference engine and quantization kernels
- IFBench, GSM8K, HumanEval — evaluation suites
