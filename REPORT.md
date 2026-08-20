# DSpark Q4 WNA16 GPU2 优化报告（第三轮：kernel 实验 + 长稳压力测试）

## 独立工作副本
- `/opt/dspark_dev_vllm`
- 未修改官方路径：`/opt/qwen38_27b/site_vllm021`、`/opt/qwen38_27b` 现有配置、`/opt/workspace/deepseek-harness` 均未触碰。

## 实际改动的文件（仅限副本内）
| 文件 | 说明 |
|---|---|
| `start_gpu2_dspark.sh` | 最终稳定配置：`max_model_len=4096`、`max_num_seqs=16`、`max_num_batched_tokens=2048`、`gpu_memory_utilization=0.80`、capture≤56、`num_speculative_tokens=7`、probabilistic。 |
| `bench_dspark.py` | 1/3/8 路吞吐测试。 |
| `profile_dspark.py` / `analyze_trace.py` | torch profiler 采集/分析。 |
| `gen_ifbench_subset.py` / `eval_ifbench_subset.py` | IFBench 质量评测脚本。 |
| `stress_test.py` | 1 小时 8 路并发压力测试脚本。 |
| `quality_ifbench_subset.jsonl` | 基线 20 条 IFBench 响应。 |
| `quality_ifbench_optimized.jsonl` | 优化后 20 条 IFBench 响应。 |
| `quality_ifbench_120_optimized.jsonl` | 优化后 120 条 IFBench 响应。 |
| `stress.log` / `quality_after_stress.log` | 压测和压测后质量评测日志。 |
| `prof_trace/decode8_rank0.*.pt.trace.json` | 8 路 decode profile trace。 |

本轮 kernel 实验文件在验证后均已**回滚为官方原版**：
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`（曾临时强制 Q4 使用 TritonW4A16，回滚）
- `vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py`（曾临时调整 GDN kernel warp/stages，回滚）

## 最终配置（当前 GPU2 服务）
```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/opt/dspark_dev_vllm \
VLLM_USE_FLASHINFER_SAMPLER=0 \
/opt/toolchains/py312/bin/python -m vllm.entrypoints.openai.api_server \
  --model /opt/qwen38_27b/model_autoread \
  --served-model-name qwen3.8-27b-dspark \
  --host 0.0.0.0 --port 8002 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.80 \
  --enable-chunked-prefill --enable-prefix-caching \
  --kv-cache-dtype int8_per_token_head \
  --compilation-config '{"cudagraph_capture_sizes":[1,8,16,24,32,40,48,56],"max_cudagraph_capture_size":56}' \
  --speculative-config '{"method":"dspark","model":"/opt/models/Doopeworld_Qwen3.8-27B-DSpark-vLLM","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

## Kernel 深度优化实验结果（均回滚）
| 实验 | 修改内容 | 1 路 | 3 路 | 8 路 | 结论 |
|---|---:|---:|---:|---:|---|
| 最终基线（Marlin） | 无 | 68.66 | 168.02 | 239.67 | 当前最优 |
| 轻量 W4A16：Q4 层强制 TritonW4A16 | `compressed_tensors_wNa16.py` | 15.68 | 44.68 | 104.97 | 严重变慢，回滚 |
| GDN recurrent：动态减少 warp/stages | `fused_sigmoid_gating.py` | 56.62 | 139.60 | 239.18 | 小 batch 反而变慢，回滚 |
| Marlin W4A8-INT8 (`VLLM_MARLIN_INPUT_DTYPE=int8`) | 环境变量 | 启动失败 | - | - | 8bit embedding 触发 `W8A8 not supported`，回滚 |
| Exllama + FP16 (`--dtype float16 --linear-backend exllama`) | 启动参数 | 46.24 | 55.93 | 73.67 | 明显更慢且 graph 内存升至 4.77 GiB，回滚 |

结论：现有 Marlin + 原生 GDN kernel 在该 A6000 小 batch 场景下仍优于 TritonW4A16、Exllama+FP16 和 Marlin W4A8-INT8 等现成替代；更轻量的 kernel 需要专门为 M≤16 的 W4A16 实现，不能简单换用现成路径。

## 最终吞吐与显存
| 项目 | 第一轮基线 | 最终优化 |
|---|---:|---:|
| 1 路 tok/s | 57.17 | **68.66** |
| 3 路 tok/s | 139.54 | **168.02** |
| 8 路 tok/s | 253.24 | **239.67** |
| GPU2 显存 | ~42.8 GiB | **~40.6 GiB** |
| CUDA graph | 1.19 GiB | **1.19 GiB** |
| KV cache | ~20.8 GiB | **~18.4 GiB** |

## 长时间压力测试稳定性（1 小时）
- 时长：**3616.7 秒（约 1 小时 0 分 17 秒）**
- 并发：8 路持续请求，`max_tokens=128`
- 总请求数：**6328**
- 总生成 tokens：**809,984**
- 聚合吞吐：**223.95 tok/s**
- 平均请求延迟：**4.572 s**
- 错误数：**0**
- GPU2 显存：稳定在 **40.6 GiB**
- GPU2 利用率：稳定 **99–100%**
- 无崩溃、无 OOM、无性能衰减；每 60s 窗口吞吐基本稳定在 221–239 tok/s。

压力测试日志：`stress.log`

## 质量校验
### 20 条子集（基线与最终优化对比）
| 版本 | strict prompt | strict instruction | loose prompt | loose instruction |
|---|---:|---:|---:|---:|
| 基线 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| 最终优化 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |

### 120 条子集（最终优化）
| 指标 | strict | loose |
|---|---:|---:|
| prompt-level | **0.5333** | **0.5333** |
| instruction-level | **0.5530** | **0.5682** |

数据文件：`quality_ifbench_120_optimized.jsonl`，评测日志：`quality_after_stress.log`

## 220K 长上下文对齐（最新）
按用户要求将 GPU2 实验实例 `max_model_len` 对齐到 **220000**，并启用 CPU KV offload：
```bash
--max-model-len 220000
--max-num-seqs 12
--max-num-batched-tokens 2048
--gpu-memory-utilization 0.80
--kv-cache-dtype int8_per_token_head
--kv-offloading-size 96
--kv-offloading-backend native
--compilation-config '{"cudagraph_capture_sizes":[1,8,16,24,32,40,48,56,64,72,80,84],"max_cudagraph_capture_size":84}'
```

### 220K 配置吞吐（1/3/5/8/12 路）
| 并发 | 1 | 3 | 5 | 8 | 12 |
|---|---:|---:|---:|---:|---:|
| tok/s | 56.89 | 139.39 | 194.63 | 240.75 | 230.45 |

日常实际并发 ≤5 时，吞吐约 **57 / 139 / 195** tok/s。

### 220K 配置显存
- GPU2 总显存：约 **40.5–41.2 GiB**
- KV cache：GPU 上约 18.4 GiB，另有 CPU offload 96（配置值）
- 最大并发按 220K/request 约 1.5x（受 KV cache 容量限制）

### 长上下文稳定性测试（220K 配置）
| 目标长度 | 实际 prompt_tokens | 输出 tokens | 耗时 | 结果 |
|---:|---:|---:|---:|---|
| 50K | 49,892 | 32 | 66.19s | OK |
| 100K | 99,772 | 32 | 155.69s | OK |
| 200K | 199,532 | 32 | 545.67s | OK |
| 220K | 219,482 | 32 | 169.31s | OK |

- 无崩溃、无 OOM、无超时。
- 测试期间 GPU2 显存稳定在约 **41.2 GiB**。
- 日志：`long_context.log`
- 测试脚本：`long_context_test.py`

### 220K 配置质量（IFBench 20 条）
| 版本 | strict prompt | strict instruction | loose prompt | loose instruction |
|---|---:|---:|---:|---:|
| 基线 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| 220K 配置 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |

数据：`quality_ifbench_220k.jsonl`

## 当前 GPU2 服务状态
- 端口：**8002**
- API server PID：**2885112**
- EngineCore PID：**2885296**
- 当前配置：**max_model_len=220000 + CPU KV offload**
- 日志：`/opt/dspark_dev_vllm/gpu2_dspark.log`
- 长上下文测试后服务仍正常运行，未重启。

## 后续建议
1. 轻量 W4A16 kernel 需要专门实现 M≤16 的 CUTLASS/Triton kernel（例如沿用 Marlin 的 repacked 权重格式、仅改 tile/warp 配置），而不是直接切换到现有 TritonW4A16。
2. GDN `fused_sigmoid_gating_delta_rule_update_kernel` 仍需针对 DSpark spec-decode 的多 token/request 形态做专门 kernel；简单的 warp/stages 调整无效。
3. 可继续用 120 条 IFBench 作为标准质量集；后续每次 kernel 改动后跑同一 120 条对比。
4. 长时间压测建议纳入 CI 门禁：1 小时 8 路无错误、显存平稳、吞吐无衰减。
