#!/usr/bin/env bash
# Experimental GPU2 server launcher for fair A/B: modes dspark|mtp|nospec
# Evidence + rollback: logs per mode in $DEV/logs_exp/, original start_gpu2_dspark.sh untouched.
set -euo pipefail
MODE="${1:?usage: start_gpu2_exp.sh dspark|mtp|nospec [max_num_seqs]}"
MAX_SEQS="${2:-4}"
NST="${NST:-7}"
DRAFT_SAMPLE="${DRAFT_SAMPLE:-probabilistic}"
FI_SAMPLER="${FI_SAMPLER:-0}"
DEV=/opt/dspark_dev_vllm
BASE=/opt/qwen38_27b
MODEL_DIR="${MODEL_DIR_OVERRIDE:-$BASE/model_autoread}"
DRAFT_DIR=/opt/models/Doopeworld_Qwen3.8-27B-DSpark-vLLM
CUDA_ENV=/opt/workspace/cuda_env
mkdir -p "$DEV/logs_exp"
LOG="$DEV/logs_exp/gpu2_${MODE}_$(date +%m%d_%H%M).log"
PID_FILE="$DEV/gpu2_exp.pid"

export HOME=/opt/workspace
export PYTHONPATH="$DEV"
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=16
export HF_HOME="$BASE/hf_cache"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_CACHE_ROOT=/opt/workspace/vllm_cache
export TRITON_CACHE_DIR=/opt/workspace/triton_cache
export VLLM_USE_FLASHINFER_SAMPLER="$FI_SAMPLER"
export VLLM_SPEC_DECODE_ATTN="${SPEC_ATTN:-0}"
export CC=/opt/toolchains/bin/gcc
export CXX=/opt/toolchains/bin/g++
export CPATH=/opt/toolchains/py312/include/python3.12
export CUDA_HOME="$CUDA_ENV"
export CUDA_PATH="$CUDA_ENV"
export PATH="/opt/workspace/build_env/bin:$CUDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ENV/lib:${LD_LIBRARY_PATH:-}"

# stop existing exp/dspark server (kill whole session groups incl. EngineCore)
for pf in "$PID_FILE" "$DEV/gpu2_dspark.pid"; do
  if [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; then
    OLDPID=$(cat "$pf")
    kill -- -"$OLDPID" 2>/dev/null || kill "$OLDPID" || true
    for i in $(seq 1 15); do kill -0 "$OLDPID" 2>/dev/null || break; sleep 2; done
    kill -9 -- -"$OLDPID" 2>/dev/null || kill -9 "$OLDPID" 2>/dev/null || true
  fi
done
# wait until GPU2 memory actually released (restart-race guard)
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
  [ "$used" -lt 3000 ] && break
  sleep 2
done
# still occupied -> orphaned EngineCore: kill all compute pids on GPU2 only
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
if [ "$used" -ge 3000 ]; then
  echo "WARN: GPU2 still ${used}MiB after kill; clearing orphaned compute pids on GPU2"
  for cpid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 2); do
    kill -9 "$cpid" 2>/dev/null || true
  done
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
    [ "$used" -lt 3000 ] && break
    sleep 2
  done
fi
sleep 3

SPEC_ARGS=()
MAMBA_FLAG=()
[ -n "${MAMBA_DTYPE:-}" ] && MAMBA_FLAG=(--mamba-ssm-cache-dtype "$MAMBA_DTYPE")
# cudagraph capture 上限需覆盖 max_num_seqs*(1+nst)：dspark nst=7 -> 8x; mtp nst=3 -> 4x
if [ "$MAX_SEQS" -le 4 ]; then
  CAPS='[1,8,16,24,28]'; CAPMAX=28
elif [ "$MAX_SEQS" -le 8 ]; then
  CAPS='[1,8,16,24,32,40,48,56,64]'; CAPMAX=64
else
  CAPS='[1,8,16,24,32,40,48,56,64,72,80,88,96,104,112]'; CAPMAX=112
fi
COMP_CFG="{\"cudagraph_capture_sizes\":${CAPS},\"max_cudagraph_capture_size\":${CAPMAX}}"
if [ "$MODE" = "dspark" ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"dspark\",\"model\":\"$DRAFT_DIR\",\"num_speculative_tokens\":${NST},\"draft_sample_method\":\"${DRAFT_SAMPLE}\"}")
elif [ "$MODE" = "mtp" ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_NST:-3},\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"}")
  # MTP draft 的 triton kernel 与 full cudagraph capture 冲突（stream capture 报错），生产用 piecewise
  COMP_CFG='{"cudagraph_mode":"piecewise","custom_ops":["+rms_norm","+silu_and_mul"]}'
fi

cd "$DEV"
setsid /opt/toolchains/py312/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name qwen3.8-27b-dspark \
  --host 0.0.0.0 \
  --port 8002 \
  --tensor-parallel-size 1 \
  --max-model-len ${MAX_LEN:-220000} \
  --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization ${MEM_UTIL:-0.80} \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --kv-cache-dtype ${KV_DTYPE:-int8_per_token_head} \
  --kv-offloading-size 96 \
  --kv-offloading-backend native \
  --compilation-config "$COMP_CFG" \
  "${SPEC_ARGS[@]}"   "${MAMBA_FLAG[@]}" \
  --api-key sk-your-api-key   --reasoning-parser qwen3   --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "xhigh"}' \
  --trust-remote-code \
  > "$LOG" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "mode=$MODE pid=$(cat $PID_FILE) log=$LOG"
