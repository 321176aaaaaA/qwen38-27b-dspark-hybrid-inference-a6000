# Qwen3.8-27B DSpark — A6000 混合精度推理

**中文** | [English](README.md)

本仓库发布 Qwen3.8-27B DSpark 的生产级混合精度 checkpoint（Q4 W4A16 正式版 + M1 down_proj-8bit
混合变体）及配套 vLLM 优化补丁集，针对 NVIDIA A6000（sm_86）调优。

> 本项目基于 [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) 的模型与评测体系，
> 在其基础上针对 A6000（sm_86）做了量化/投机解码/kernel 级优化。感谢上游工作。

## Checkpoint: model_mtp_opt

基于 Qwen3.8-27B（混合架构：48× linear attention + 16× full attention, interval=4, 64 layers）：

- 主干 **W4A16** pack-quantized（int4 g128 symmetric，lm_head/embed int8 g128）
- MTP 层 **int8 g128**（round-trip rel err 0.66–1.53%）
- MTP draft head **40960 词表**（draft_vocab_ids 来自模型自身输出统计）
- `mtp_draft_vocab_ids.pt` 随仓库 git 分发（329KB）

Release `v1.0.0-q4-w4a16` assets 包含 10 个 safetensors 分片（≤2GB/片，GitHub 限制），
下载全部分片后与仓库内小文件（config/tokenizer/index 等）放同一目录即可加载。

## 基准测试（测试服务器，A6000 48GB sm_86，vLLM 0.26 + patches/）

### Q4 最终栈（本 release checkpoint + 全部补丁）

| 指标 | 数值 |
|---|---|
| 单路 decode | 93.3 tok/s |
| 8 路并发 | 517.5 tok/s |
| 长输出（10496 tok）单路 / 8路 | 91.8 / 476.9 tok/s |
| PPL all（wikitext2-en/fineweb2-da/code 33k tok） | 8.0845（历史 BF16 参考档 8.045） |
| GSM8K-200 | 94.0% |
| HumanEval-164 pass@1 | 79.9% |
| IFBench-300 thinking（60K budget，xhigh，temp1.0/top-p0.95/top-k20） | strict 72.7 / loose 79.3 |
| 显存占用（权重） | ~15 GB |

IFBench 口径说明：no-think 512 token 协议仅 0.40（预算不足）；xhigh 思维链 47% 超 8192，
32K 预算产生 7 条空响应（strict 71.3），60K 预算 0 空响应（strict 72.7 / loose 79.3）。

### M1 混合位宽（down_proj→int8，实验档，质量优先）

主干 4bit 不变，全部 64 层 `down_proj`（residual stream 注入点，量化最敏感层类）升为 int8 g128；
带宽预算 17GB/步，仍在单路 80 tok/s 门槛内。

| 指标 | M1 | Q4 | W8A16 |
|---|---|---|---|
| 单路 / 8路 tok/s | **85.0 / 475.0** | 93.3 / 517.5 | 52.4 / 337.9 |
| PPL all | 8.0242 | 8.0845 | 7.7681 |
| GSM8K-200 | **95.5%** | 94.0% | 95.5% |
| HumanEval-164 pass@1 | **81.1%** | 79.9% | 79.9% |
| MTP 接受长度 | **3.07** | 2.90 | 2.82 |
| IFBench fail82 子集 strict（Q4 全错的 82 题） | **24/82 = 29.3%** | 0/82 | FP8 档 17/82 = 20.7% |

构建脚本：`scripts/build_m1_hybrid.py`（硬链接复制 + 从 BF16 重量化 down_proj + config 前插 group_d8）。

### 量化方案对比（同机同栈实测）

| 方案 | 权重显存 | 单路 tok/s | 8路 tok/s | PPL all | GSM8K | HumanEval |
|---|---|---|---|---|---|---|
| **Q4 W4A16 + MTP4（本 release）** | ~15 GB | **93.3** | **517.5** | 8.0845 | 94.0% | 79.9% |
| **M1 混合位宽 + MTP4（实验档）** | ~17 GB | 85.0 | 475.0 | 8.0242 | **95.5%** | **81.1%** |
| W8A16 + MTP4（实验） | ~28 GB | 52.4 | 337.9 | 7.7681 | 95.5% | 79.9% |
| FP8 官方动态量化 + MTP4 | ~29 GB | 45.1 | 300.3 | 7.792 | 95.5% | — |
| Q8_0 GGUF（llama.cpp + MTP） | ~29 GB | 22.4 | 65.6 | 6.55* | 96.5% | — |

*llama.cpp 滑窗 PPL 口径，与 vLLM completion 口径不可直接比较。

单路 decode 与权重字节数严格成正比（1430 tok·GB/s 常数：93.3×15.3 ≈ 52.4×27.3），
sm_86 单路场景完全带宽 bound；FP8 在 sm_86 无 tensor core 加速。

### KV 容量与长上下文

- 单请求上下文上限 262144 tokens
- 全注意力层 KV：bf16 64 KB / int8 32 KB 每 token
- KV 总容量 = GPU 16.5 GB（bf16，257k tok）+ CPU offload 96 GB ≈ 1.75M tokens
- 超显存并发（6×100k tok 实测）：请求排队串行化，0 抢占 0 崩溃，decode 88–122 tok/s 不降速
- offload 重访 TTFT 3.2–4 s vs 冷启动 81 s（25×）

## 启动命令（最终形态）

```bash
vllm serve <checkpoint_dir> \
  --kv-cache-dtype auto \                 # bf16 KV（配合 spec-decode-attn）
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}' \
  --mamba-ssm-cache-dtype float16 \
  --max-num-seqs 12 --max-num-batched-tokens 2048 \
  --kv-offloading-size 96 --kv-offloading-backend native \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "xhigh"}'
# 环境变量: VLLM_SPEC_DECODE_ATTN=1
```

完整启动器见 `scripts/start_gpu2_exp.sh`（含脏 GPU 清理、cudagraph 分档、全部旋钮）。

## patches/（dev vLLM 0.26 补丁集，COMPILE_OK 验证过）

1. `PR50021` GDN spec-decode kernel 越界修复（4 kernel 文件；**不修会导致长 thinking 输出质量崩塌/引擎崩溃**）
2. sampler small-topk 免排序 + 多块行 softmax + draft 同截断支撑采样（`VLLM_DRAFT_TOPK_TOPP=1` 默认开）
3. MTP 40k draft vocab head（checkpoint 含 `mtp_draft_vocab_ids.pt` 时自动启用；`MTP_DRAFT_VOCAB=0` 关闭）
4. spec-decode-attn split-KV verify attention（`VLLM_SPEC_DECODE_ATTN=1`，bf16 KV only，长输出 +31%）
5. marlin int8 负 scale 符号修复 + 层选 regex（`VLLM_MARLIN_INT8_INCLUDE_RE/EXCLUDE_RE`）

## 关键运维教训

- benchmark 看不出输出垃圾（int8 Marlin 符号 bug 教训）——改动后必跑质量门（PPL/GSM8K）
- 脏 GPU 重启静默损失 25% 性能——启动脚本已防护
- 随机 token benchmark 高估投机收益——用真实 prompt
- draft 词表必须用模型自身输出统计（单项最大提速 +10%）
- thinking 评测 max_tokens ≥32768（xhigh 思维链 47% 超 8192 预算）

## 对比结论（A6000）

Q4 W4A16 是速度/质量/显存最优平衡：

- **FP8**：质量略好（GSM8K +1.5pp）但 sm_86 无 FP8 tensor core，速度腰斩（45 vs 93 tok/s）
- **W8A16**：质量打平 FP8（PPL 7.77），速度优于 FP8（+16%/+12%），但单路仍只有 Q4 的 56%
  （带宽 bound，权重 2×）；Marlin int8 tensor core 收益在 8 路以上并发才显现
- **Q8_0 GGUF**：GSM8K 最高（96.5%）但 llama.cpp 工程性能远逊（22 tok/s 单路，8 路 65.6）
- Q4→W8 质量差距：PPL +4%，GSM8K −1.5pp，HumanEval 持平

## 许可证

- **代码**（本仓库的补丁、脚本、评测工具）：[Apache License 2.0](LICENSE)，
  与上游 [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) 保持一致。
- **模型权重**：checkpoint 派生自 Qwen3.8-27B（DSpark 变体），权重的使用须遵循原模型的
  许可条款。本仓库不就权重授予任何额外权利。

## 致谢

- [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) —— 基础模型与评测体系
- vLLM / compressed-tensors / Marlin —— 推理引擎与量化 kernel
- IFBench / GSM8K / HumanEval —— 评测套件
