#!/usr/bin/env bash
# M1 production: int4_per_token_head KV (validated on GPU2 08-22:
# GPQA 93.8%==int8 baseline, NIAH-190K 5/5==int8, 4x193K soak 0 crash,
# prefill 1112 t/s >= int8 1004 t/s).  max-seqs 8 for 8-user concurrency.
set -euo pipefail
BASE=/opt/qwen38_27b
MODEL_DIR="$BASE/model_m1_hybrid"
CUDA_ENV=/opt/workspace/cuda_env
LOG="${LOG:-$BASE/vllm_official_200k.log}"
PID_FILE="${PID_FILE:-$BASE/vllm_official_200k.pid}"
API_KEY="${API_KEY:?set API_KEY env (production key, not committed)}"
PORT="${PORT:-8000}"
export PYTHONPATH="$BASE/site_vllm021"
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HOME=/opt/workspace
export HF_HOME="$BASE/hf_cache"
export VLLM_LOGGING_LEVEL=INFO
export TRITON_CACHE_DIR="$BASE/triton_cache"
export VLLM_CACHE_ROOT="$BASE/vllm_cache"
export CC=/opt/toolchains/bin/gcc
export CXX=/opt/toolchains/bin/g++
export CPATH=/opt/toolchains/py312/include/python3.12
export CUDA_HOME="$CUDA_ENV"
export CUDA_PATH="$CUDA_ENV"
export PATH="/opt/workspace/build_env/bin:$CUDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ENV/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_SPEC_DECODE_ATTN=1
export VLLM_SPEC_DECODE_INT4=1
export VLLM_TUA_LARGE_HEAD=1
export VLLM_TUA_BLOCK_M=64
export VLLM_TUA_TILE=64
export VLLM_TUA_NUM_WARPS=8
export VLLM_TUA_NUM_STAGES=2
export VLLM_SERVER_DEV_MODE=1

cd "$BASE"
setsid python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name qwen3.8-27b-vision \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 220000 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 3136 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --kv-cache-dtype int4_per_token_head \
  --kv-offloading-size 96 \
  --kv-offloading-backend native \
  --attention-backend triton_attn \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"piecewise","custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --api-key "$API_KEY" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "xhigh"}' \
  --trust-remote-code \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
echo "vLLM 200k starting PID $(cat "$PID_FILE"), log: $LOG"
