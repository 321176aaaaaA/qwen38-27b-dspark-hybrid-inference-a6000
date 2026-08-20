#!/usr/bin/env bash
set -euo pipefail
# Start Q4 DSpark on GPU2 from the isolated dev copy.
DEV=/opt/dspark_dev_vllm
BASE=/opt/qwen38_27b
MODEL_DIR="$BASE/model_autoread"
DRAFT_DIR=/opt/models/Doopeworld_Qwen3.8-27B-DSpark-vLLM
CUDA_ENV=/opt/workspace/cuda_env
LOG="$DEV/gpu2_dspark.log"
PID_FILE="$DEV/gpu2_dspark.pid"

export HOME=/opt/workspace
export PYTHONPATH="$DEV"
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=16
export HF_HOME="$BASE/hf_cache"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_CACHE_ROOT=/opt/workspace/vllm_cache
export TRITON_CACHE_DIR=/opt/workspace/triton_cache
export VLLM_USE_FLASHINFER_SAMPLER=0
export CC=/opt/toolchains/bin/gcc
export CXX=/opt/toolchains/bin/g++
export CPATH=/opt/toolchains/py312/include/python3.12
export CUDA_HOME="$CUDA_ENV"
export CUDA_PATH="$CUDA_ENV"
export PATH="/opt/workspace/build_env/bin:$CUDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ENV/lib:${LD_LIBRARY_PATH:-}"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "already running PID $(cat "$PID_FILE")"
  exit 0
fi

cd "$DEV"
setsid /opt/toolchains/py312/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name qwen3.8-27b-dspark \
  --host 0.0.0.0 \
  --port 8002 \
  --tensor-parallel-size 1 \
  --max-model-len 220000 \
  --max-num-seqs 12 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.80 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --kv-cache-dtype int8_per_token_head \
  --kv-offloading-size 96 \
  --kv-offloading-backend native \
  --compilation-config "{\"cudagraph_capture_sizes\":[1,8,16,24,32,40,48,56,64,72,80,84],\"max_cudagraph_capture_size\":84}" \
  --speculative-config "{\"method\":\"dspark\",\"model\":\"$DRAFT_DIR\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\"}" \
  --api-key sk-your-api-key \
  --trust-remote-code \
  > "$LOG" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "DSpark dev server starting PID $(cat "$PID_FILE"), log: $LOG"
