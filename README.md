# Qwen3.8-27B DSpark Q4 PTX A6000 Optimization

私有仓库，记录 Qwen3.8-27B Q4 DSpark 在 A6000（PTX/SM86）上的推理优化方案、数据与复现方法。

## 当前最优配置（220K 对齐）
```bash
--max-model-len 220000
--max-num-seqs 12
--max-num-batched-tokens 2048
--gpu-memory-utilization 0.80
--kv-cache-dtype int8_per_token_head
--kv-offloading-size 96
--kv-offloading-backend native
--compilation-config '{"cudagraph_capture_sizes":[1,8,16,24,32,40,48,56,64,72,80,84],"max_cudagraph_capture_size":84}'
--speculative-config '{"method":"dspark","model":"/mnt/6/wangzihan/Doopeworld_Qwen3.8-27B-DSpark-vLLM","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

## 吞吐（220K 配置，OpenAI API，短 prompt，max_tokens=256，关闭 thinking）
| 并发 | 1 | 3 | 5 | 8 | 12 |
|---|---:|---:|---:|---:|---:|
| tok/s | 56.89 | 139.39 | 194.63 | 240.75 | 230.45 |

日常实际并发 ≤5：约 57 / 139 / 195 tok/s。

## 显存
- GPU2 总显存：约 40.5–41.2 GiB（A6000 49GB 可稳定运行）
- GPU KV cache：约 18.4 GiB
- CPU KV offload：96G

## 长上下文稳定性
| 目标长度 | 实际 prompt_tokens | 输出 tokens | 耗时 | 结果 |
|---:|---:|---:|---:|---|
| 50K | 49,892 | 32 | 66.19s | OK |
| 100K | 99,772 | 32 | 155.69s | OK |
| 200K | 199,532 | 32 | 545.67s | OK |
| 220K | 219,482 | 32 | 169.31s | OK |

无崩溃、无 OOM、无超时。

## 质量校验（IFBench）
- 20 条子集：与基线完全一致，未劣化
- 120 条子集（最终优化）：strict 0.5333 / loose 0.5333（prompt-level）
- 数据文件见 `quality_ifbench_220k.jsonl`、`quality_ifbench_120_optimized.jsonl`

## 复现
```bash
# 启动 GPU2 DSpark（220K + offload）
bash start_gpu2_dspark.sh

# 吞吐 benchmark
python3 bench_dspark.py

# profile
python3 profile_dspark.py
python3 analyze_trace.py
```

## 关键结论
1. `num_speculative_tokens=7` 最优；14 更慢。
2. CUDA graph 捕获到 8 路（≤56 token）是吞吐/显存最佳点；16 路无收益。
3. 关闭 async scheduling 会显著降低 1/3 路吞吐。
4. Marlin W4A16 小 batch 仍是第一热点；GDN spec decode 是第二热点。
5. 下一步：实现 M≤16 轻量 W4A16 kernel、优化 GDN recurrent kernel、削减 CPU 小算子。

## 文件说明
- `REPORT.md`：完整实验报告
- `start_gpu2_dspark.sh`：GPU2 启动脚本
- `bench_dspark.py`：1/3/8 路吞吐测试
- `profile_dspark.py` / `analyze_trace.py`：profile 工具
- `patches/`：若有的 vLLM 源码修改文件（相对 site_vllm021 路径）
