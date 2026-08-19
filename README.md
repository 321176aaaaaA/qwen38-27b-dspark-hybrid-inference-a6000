# Qwen3.8-27B DSpark Q4 (W4A16) — A6000 Optimized Checkpoint

**Private release.** 本仓库发布 Q4(W4A16) 正式版 checkpoint 及配套优化补丁。

## Checkpoint: model_mtp_opt

基于 Qwen3.8-27B（混合架构：48× linear attention + 16× full attention, interval=4, 64 layers）：

- 主干 **W4A16** pack-quantized（int4 g128 symmetric，lm_head/embed int8 g128）
- MTP 层 **int8 g128**（round-trip rel err 0.66–1.53%）
- MTP draft head **40960 词表**（draft_vocab_ids 来自模型自身输出统计）
- `mtp_draft_vocab_ids.pt` 随仓库 git 分发（329KB）

Release `v1.0.0-q4-w4a16` assets 包含 10 个 safetensors 分片（≤2GB/片，GitHub 限制），
下载全部分片后与仓库内小文件（config/tokenizer/index 等）放同一目录即可加载。

## 性能（164 服务器，A6000 48GB sm_86，vLLM 0.26 + patches/）

| 指标 | 数值 |
|---|---|
| 单路 decode | 93.3 tok/s |
| 8 路并发 | 517.5 tok/s |
| 长输出（10496 tok）单路 / 8路 | 91.8 / 476.9 tok/s |
| PPL all（wikitext2-en/fineweb2-da/code 33k tok） | 8.087（参考 BF16 档 8.045） |
| GSM8K-200 | 93.5% |
| HumanEval-164 | 见 EVIDENCE_LOG（评测中） |

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
- 单请求上下文上限 262144 tokens；KV 总容量 = GPU(257k tok bf16) + CPU offload（96GB → ~1.75M tok）

## 对比结论（A6000）

Q4 W4A16 是速度/质量/显存最优平衡：FP8 质量略好（GSM8K +2pp）但 sm_86 无 FP8 tensor core
速度腰斩（45 vs 93 tok/s）；Q8_0 GGUF GSM8K 最高（96.5%）但 llama.cpp 工程性能远逊（22 tok/s）。
Q4→Q8 质量差距仅 ~1% PPL。
