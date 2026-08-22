"""Split-KV paged attention for speculative-decode batches (a few query tokens per
request, GQA), Triton.

Neither vLLM's FlashAttention-2 path nor its Triton unified attention split the KV
sequence across SMs when a request has more than one query token: with MTP k=4
(5 queries) on a 24-head model that leaves 24 (FA) or ~8 (Triton) thread blocks on an
82-SM RTX 3090 and the attention layer takes ~57 us for a 1.5k-token context.
This kernel gives every (request, kv-head) NUM_SEGMENTS blocks, each computing an
online-softmax partial over its slice of the KV cache for all q_len x G query rows at
once (G = query heads per kv head), followed by a tiny combine kernel.

Layout matches vLLM's FLASH_ATTN backend: q [T, Hq, D], key/value cache
[num_blocks, block_size, Hkv, D] (block_size any multiple of 16), block_table
[num_reqs, max_blocks], seqused_k [num_reqs] = kv length including the new tokens,
cu_seqlens_q [num_reqs + 1]. Query token i of a request sits at kv position
seqused_k - q_len + i and attends causally.

Restrictions: q_len * G <= BLOCK_M (64) per request, D a power of two <= 256,
no sliding window / softcap / alibi.

Quantized KV: when k_scale_cache / v_scale_cache are passed (per-token-head
int8/fp8 layout used by the TRITON_ATTN backend), K/V stay quantized in the
cache and the per-(token, head) scales are folded into the score columns and
the P matrix respectively, mirroring the core unified-attention kernel.

Two-pass mode (twopass=True, default via VLLM_SPEC_DECODE_TWOPASS=1):
the single-pass kernel holds acc[BLOCK_M, D] fp32 in registers across the KV
loop; at D=256 that is 32KB per program and spills badly on sm86 (REG:255,
STACK:632), capping deep-context bandwidth at ~150-230 GB/s.  Two-pass mode
splits the work so no program ever holds acc over the full head dim:

  pass 1 (_spec_p1_score): reads K once, computes S = QK^T per tile, and
      writes P = exp(S - m_tile) (bf16) plus the per-tile max m_tile to a
      workspace.  No accumulator at all -> tiny register footprint.
  pass 2 (_spec_p2_pv): online-softmax accumulation over tiles, but the
      "scores" are reloaded from the P workspace instead of recomputed from
      Q,K, and the head dim is split (DS = D // DSPLIT) so acc is
      [BLOCK_M, DS].  v_scale and the per-tile rescale weight w are folded
      into the P tile before the dot; the softmax denominator l uses the
      unscaled P (mirrors single-pass semantics).

The combine kernel is shared between both modes.  Extra DRAM traffic is one
write+read of P (~2-3% of the KV bytes), far cheaper than a second K read.
"""
import os

import torch
import triton
import triton.language as tl

NUM_SEGMENTS = 16
BLOCK_M = 64      # max query rows (q_len * G) per request handled by one program
_P_BUF_CAP = 2 * 1024**3  # bytes; above this fall back to single-pass


def _rht(q, inverse=False):
    """Forward/inverse Randomized Hadamard Transform from the int4 KV mode.
    Imported lazily so this module has no hard dependency on hadacore."""
    from vllm.v1.attention.ops.int4_per_token_head import single_rht
    return single_rht(q, inverse=inverse)


def _twopass_default():
    # measured slower than single-pass both in microbench (351 vs 450 GB/s)
    # and in-situ (3.4 vs 2.6 ms/layer); kept for reference, opt-in only
    return os.environ.get("VLLM_SPEC_DECODE_TWOPASS", "0") == "1"


@triton.jit
def _spec_attn_partial(
    q_ptr, k_ptr, v_ptr, bt_ptr, seqused_ptr, cu_q_ptr,
    ks_ptr, vs_ptr,
    part_o_ptr, part_m_ptr, part_l_ptr,
    scale,
    stride_qt, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_bt,
    stride_ksb, stride_kss, stride_ksh,
    stride_vsb, stride_vss, stride_vsh,
    G: tl.constexpr, Hq: tl.constexpr, QMAX: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, TILE: tl.constexpr, NSEG: tl.constexpr,
    USE_PTH_SCALES: tl.constexpr, GSPLIT: tl.constexpr,
):
    req = tl.program_id(0)
    kvh2 = tl.program_id(1)
    seg = tl.program_id(2)
    kvh = kvh2 // GSPLIT
    gsub_off = (kvh2 % GSPLIT) * G

    q_start = tl.load(cu_q_ptr + req)
    q_len = tl.load(cu_q_ptr + req + 1) - q_start
    kv_len = tl.load(seqused_ptr + req)

    # rows: r = i * G + g  -> query token i (0..q_len-1),
    # head (kvh * G_ORIG + gsub_off + g) with G rows in this program
    r = tl.arange(0, BLOCK_M)
    ri = r // G
    rg = r % G
    row_ok = ri < q_len
    q_pos = kv_len - q_len + ri                      # kv position of each query row
    d = tl.arange(0, D)
    q_ptrs = q_ptr + (q_start + ri)[:, None] * stride_qt + (kvh * G * GSPLIT + gsub_off + rg)[:, None] * stride_qh + d[None, :]
    q = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0)

    # this segment's key range
    tiles_total = (kv_len + TILE - 1) // TILE
    tiles_per_seg = (tiles_total + NSEG - 1) // NSEG
    t0 = seg * tiles_per_seg
    t1 = tl.minimum(t0 + tiles_per_seg, tiles_total)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    qs = (q * scale).to(tl.bfloat16)

    for t in range(t0, t1):
        pos = t * TILE + tl.arange(0, TILE)
        k_ok = pos < kv_len
        blk = tl.load(bt_ptr + req * stride_bt + pos // BLOCK_SIZE, mask=k_ok, other=0)
        slot = pos % BLOCK_SIZE
        k_ptrs = k_ptr + blk[:, None] * stride_kb + slot[:, None] * stride_ks + kvh * stride_kh + d[None, :]
        v_ptrs = v_ptr + blk[:, None] * stride_vb + slot[:, None] * stride_vs + kvh * stride_vh + d[None, :]
        k = tl.load(k_ptrs, mask=k_ok[:, None], other=0.0).to(qs.dtype)
        v = tl.load(v_ptrs, mask=k_ok[:, None], other=0.0).to(qs.dtype)
        s = tl.dot(qs, tl.trans(k)).to(tl.float32)            # [BLOCK_M, TILE]
        if USE_PTH_SCALES:
            # per-(token, head) int8/fp8 quant: K stays quantized in cache;
            # fold the per-token scale into the score columns (mirrors the
            # core unified kernel: S = dot(Q, K) * score_scale * k_scale).
            ks = tl.load(ks_ptr + blk * stride_ksb + slot * stride_kss + kvh * stride_ksh,
                         mask=k_ok, other=1.0)
            s = s * ks[None, :]
        allowed = k_ok[None, :] & (pos[None, :] <= q_pos[:, None]) & row_ok[:, None]
        s = tl.where(allowed, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(tl.where(m_i == float("-inf"), float("-inf"), m_i - m_safe))
        l_i = l_i * alpha + tl.sum(p, 1)
        if USE_PTH_SCALES:
            # per-(token, head) quant: apply v_scale to P columns (mirrors
            # the core unified kernel: P_v = P * v_scale; dot(P_v, V)).
            vs = tl.load(vs_ptr + blk * stride_vsb + slot * stride_vss + kvh * stride_vsh,
                         mask=k_ok, other=1.0)
            pv = (p * vs[None, :]).to(v.dtype)
        else:
            pv = p.to(v.dtype)
        acc = acc * alpha[:, None] + tl.dot(pv, v).to(tl.float32)
        m_i = m_new

    # store partials at flat index ((req*Hq + head)*QMAX + i)*NSEG + seg
    hrow = kvh * G * GSPLIT + gsub_off + rg
    pidx = ((req * Hq + hrow) * QMAX + ri) * NSEG + seg
    tl.store(part_o_ptr + pidx[:, None] * D + d[None, :], acc, mask=row_ok[:, None])
    tl.store(part_m_ptr + pidx, m_i, mask=row_ok)
    tl.store(part_l_ptr + pidx, l_i, mask=row_ok)


@triton.jit
def _spec_attn_partial_int4(
    q_ptr, k_ptr, v_ptr, bt_ptr, seqused_ptr, cu_q_ptr,
    ks_ptr, vs_ptr,
    part_o_ptr, part_m_ptr, part_l_ptr,
    stride_qt, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_bt,
    stride_ksb, stride_kss, stride_ksh,
    stride_vsb, stride_vss, stride_vsh,
    G: tl.constexpr, Hq: tl.constexpr, QMAX: tl.constexpr, D: tl.constexpr, DP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, TILE: tl.constexpr, NSEG: tl.constexpr,
    GSPLIT: tl.constexpr,
):
    """Split-KV verify attention over the INT4 packed per-token-head cache.

    Mirrors the core int4 kernel math (int4_per_token_head.py):
      - K/V stored as packed nibbles, even/odd dims -> two streams of DP=D//2
      - asymmetric quant: value = (q_nibble - zp) * scale, with zp hidden in
        the low 4 mantissa bits of the fp32 scale
      - S = (Q_s0.K_s0 + Q_s1.K_s1 - sum(Q)*k_zp) * (softmax_scale * k_scale)
        (q here is pre-rotated and pre-scaled, so the fused factor is k_scale)
      - acc streams subtract sum(P * v_scale * v_zp) once per tile
    q must already be RHT-rotated and multiplied by softmax_scale/head_size;
    the output must be inverse-RHT'd (and divided by head_size) by the caller.
    """
    req = tl.program_id(0)
    kvh2 = tl.program_id(1)
    seg = tl.program_id(2)
    kvh = kvh2 // GSPLIT
    gsub_off = (kvh2 % GSPLIT) * G

    q_start = tl.load(cu_q_ptr + req)
    q_len = tl.load(cu_q_ptr + req + 1) - q_start
    kv_len = tl.load(seqused_ptr + req)

    r = tl.arange(0, BLOCK_M)
    ri = r // G
    rg = r % G
    row_ok = ri < q_len
    q_pos = kv_len - q_len + ri
    dh = tl.arange(0, DP)
    hrow = kvh * G * GSPLIT + gsub_off + rg
    qrow = q_start + ri
    q_base = q_ptr + qrow[:, None] * stride_qt + hrow[:, None] * stride_qh
    # even/odd (stream) split of the rotated, pre-scaled q
    q_s0 = tl.load(q_base + (2 * dh)[None, :], mask=row_ok[:, None], other=0.0)
    q_s1 = tl.load(q_base + (2 * dh + 1)[None, :], mask=row_ok[:, None], other=0.0)
    qs_s0 = q_s0.to(tl.bfloat16)
    qs_s1 = q_s1.to(tl.bfloat16)
    q_sum = tl.sum(q_s0.to(tl.float32), 1) + tl.sum(q_s1.to(tl.float32), 1)

    tiles_total = (kv_len + TILE - 1) // TILE
    tiles_per_seg = (tiles_total + NSEG - 1) // NSEG
    t0 = seg * tiles_per_seg
    t1 = tl.minimum(t0 + tiles_per_seg, tiles_total)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc_s0 = tl.zeros([BLOCK_M, DP], tl.float32)
    acc_s1 = tl.zeros([BLOCK_M, DP], tl.float32)

    for t in range(t0, t1):
        pos = t * TILE + tl.arange(0, TILE)
        k_ok = pos < kv_len
        blk = tl.load(bt_ptr + req * stride_bt + pos // BLOCK_SIZE, mask=k_ok, other=0)
        slot = pos % BLOCK_SIZE
        kp_ptrs = (k_ptr + blk[:, None] * stride_kb + slot[:, None] * stride_ks
                   + kvh * stride_kh + dh[None, :])
        vp_ptrs = (v_ptr + blk[:, None] * stride_vb + slot[:, None] * stride_vs
                   + kvh * stride_vh + dh[None, :])
        k_packed = tl.load(kp_ptrs, mask=k_ok[:, None], other=0).to(tl.uint8, bitcast=True)
        v_packed = tl.load(vp_ptrs, mask=k_ok[:, None], other=0).to(tl.uint8, bitcast=True)
        k_s0 = (k_packed & 0xF).to(tl.bfloat16)
        k_s1 = ((k_packed >> 4) & 0xF).to(tl.bfloat16)
        v_s0 = (v_packed & 0xF).to(tl.bfloat16)
        v_s1 = ((v_packed >> 4) & 0xF).to(tl.bfloat16)

        ks_raw = tl.load(ks_ptr + blk * stride_ksb + slot * stride_kss + kvh * stride_ksh,
                         mask=k_ok, other=0.0)
        vs_raw = tl.load(vs_ptr + blk * stride_vsb + slot * stride_vss + kvh * stride_vsh,
                         mask=k_ok, other=0.0)
        ks_bits = ks_raw.to(tl.int32, bitcast=True)
        k_zp = (ks_bits & 0xF).to(tl.float32)
        k_scales = (ks_bits & -16).to(tl.float32, bitcast=True)
        vs_bits = vs_raw.to(tl.int32, bitcast=True)
        v_zp = (vs_bits & 0xF).to(tl.float32)
        v_scales = (vs_bits & -16).to(tl.float32, bitcast=True)

        raw = tl.dot(qs_s0, tl.trans(k_s0)).to(tl.float32) + tl.dot(qs_s1, tl.trans(k_s1)).to(tl.float32)
        s = (raw - q_sum[:, None] * k_zp[None, :]) * k_scales[None, :]
        allowed = k_ok[None, :] & (pos[None, :] <= q_pos[:, None]) & row_ok[:, None]
        s = tl.where(allowed, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(tl.where(m_i == float("-inf"), float("-inf"), m_i - m_safe))
        l_i = l_i * alpha + tl.sum(p, 1)
        pv = p * v_scales[None, :]                               # fp32
        pv_zp = tl.sum(pv * v_zp[None, :], 1)                    # [BLOCK_M]
        pv_b = pv.to(tl.bfloat16)
        acc_s0 = acc_s0 * alpha[:, None] + tl.dot(pv_b, v_s0).to(tl.float32) - pv_zp[:, None]
        acc_s1 = acc_s1 * alpha[:, None] + tl.dot(pv_b, v_s1).to(tl.float32) - pv_zp[:, None]
        m_i = m_new

    pidx = ((req * Hq + hrow) * QMAX + ri) * NSEG + seg
    tl.store(part_o_ptr + pidx[:, None] * D + (2 * dh)[None, :], acc_s0, mask=row_ok[:, None])
    tl.store(part_o_ptr + pidx[:, None] * D + (2 * dh + 1)[None, :], acc_s1, mask=row_ok[:, None])
    tl.store(part_m_ptr + pidx, m_i, mask=row_ok)
    tl.store(part_l_ptr + pidx, l_i, mask=row_ok)


@triton.jit
def _spec_p1_score(
    q_ptr, k_ptr, bt_ptr, seqused_ptr, cu_q_ptr, ks_ptr,
    p_ptr, mt_ptr,
    scale,
    stride_qt, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_bt,
    stride_ksb, stride_kss, stride_ksh,
    stride_pw, stride_mt,
    G: tl.constexpr, Hq: tl.constexpr, QMAX: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, TILE: tl.constexpr, NSEG: tl.constexpr,
    USE_PTH_SCALES: tl.constexpr, GSPLIT: tl.constexpr,
):
    """Pass 1: S = QK^T per tile -> P = exp(S - m_tile) to workspace + m_tile.

    No accumulator: registers hold only q, one K tile and one S/P tile.
    """
    req = tl.program_id(0)
    kvh2 = tl.program_id(1)
    seg = tl.program_id(2)
    kvh = kvh2 // GSPLIT
    gsub_off = (kvh2 % GSPLIT) * G

    q_start = tl.load(cu_q_ptr + req)
    q_len = tl.load(cu_q_ptr + req + 1) - q_start
    kv_len = tl.load(seqused_ptr + req)

    r = tl.arange(0, BLOCK_M)
    ri = r // G
    rg = r % G
    row_ok = ri < q_len
    q_pos = kv_len - q_len + ri
    d = tl.arange(0, D)
    q_ptrs = q_ptr + (q_start + ri)[:, None] * stride_qt + (kvh * G * GSPLIT + gsub_off + rg)[:, None] * stride_qh + d[None, :]
    q = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0)
    qs = (q * scale).to(tl.bfloat16)

    hrow = kvh * G * GSPLIT + gsub_off + rg
    pidx = ((req * Hq + hrow) * QMAX + ri)          # [BLOCK_M] workspace row

    tiles_total = (kv_len + TILE - 1) // TILE
    tiles_per_seg = (tiles_total + NSEG - 1) // NSEG
    t0 = seg * tiles_per_seg
    t1 = tl.minimum(t0 + tiles_per_seg, tiles_total)

    for t in range(t0, t1):
        pos = t * TILE + tl.arange(0, TILE)
        k_ok = pos < kv_len
        blk = tl.load(bt_ptr + req * stride_bt + pos // BLOCK_SIZE, mask=k_ok, other=0)
        slot = pos % BLOCK_SIZE
        k_ptrs = k_ptr + blk[:, None] * stride_kb + slot[:, None] * stride_ks + kvh * stride_kh + d[None, :]
        k = tl.load(k_ptrs, mask=k_ok[:, None], other=0.0).to(qs.dtype)
        s = tl.dot(qs, tl.trans(k)).to(tl.float32)            # [BLOCK_M, TILE]
        if USE_PTH_SCALES:
            ks = tl.load(ks_ptr + blk * stride_ksb + slot * stride_kss + kvh * stride_ksh,
                         mask=k_ok, other=1.0)
            s = s * ks[None, :]
        allowed = k_ok[None, :] & (pos[None, :] <= q_pos[:, None]) & row_ok[:, None]
        s = tl.where(allowed, s, float("-inf"))
        m_t = tl.max(s, 1)                                    # [BLOCK_M], -inf if row fully masked
        m_safe = tl.where(m_t == float("-inf"), 0.0, m_t)
        p = tl.exp(s - m_safe[:, None])                       # masked cols -> 0
        tl.store(p_ptr + pidx[:, None] * stride_pw + pos[None, :], p.to(tl.bfloat16),
                 mask=row_ok[:, None] & k_ok[None, :])
        tl.store(mt_ptr + pidx * stride_mt + t, m_t, mask=row_ok)


@triton.jit
def _spec_p2_pv(
    p_ptr, mt_ptr, v_ptr, bt_ptr, seqused_ptr, cu_q_ptr, vs_ptr,
    part_o_ptr, part_m_ptr, part_l_ptr,
    stride_vb, stride_vs, stride_vh,
    stride_bt,
    stride_vsb, stride_vss, stride_vsh,
    stride_pw, stride_mt,
    G: tl.constexpr, Hq: tl.constexpr, QMAX: tl.constexpr, D: tl.constexpr, DS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_M: tl.constexpr, TILE: tl.constexpr, NSEG: tl.constexpr,
    USE_PTH_SCALES: tl.constexpr, GSPLIT: tl.constexpr, DSPLIT: tl.constexpr,
):
    """Pass 2: online-softmax accumulation with P reloaded from the workspace.

    acc is [BLOCK_M, DS] with DS = D // DSPLIT, so no register spill at D=256.
    The true tile weight is P * exp(m_tile - m_run); the softmax denominator
    uses unscaled P, the PV dot uses P * v_scale (mirrors single-pass).
    """
    req = tl.program_id(0)
    kvh2 = tl.program_id(1)
    sd = tl.program_id(2)
    seg = sd // DSPLIT
    dsi = sd % DSPLIT
    kvh = kvh2 // GSPLIT
    gsub_off = (kvh2 % GSPLIT) * G

    q_start = tl.load(cu_q_ptr + req)
    q_len = tl.load(cu_q_ptr + req + 1) - q_start
    kv_len = tl.load(seqused_ptr + req)

    r = tl.arange(0, BLOCK_M)
    ri = r // G
    rg = r % G
    row_ok = ri < q_len

    hrow = kvh * G * GSPLIT + gsub_off + rg
    pidx = ((req * Hq + hrow) * QMAX + ri)

    d = dsi * DS + tl.arange(0, DS)

    tiles_total = (kv_len + TILE - 1) // TILE
    tiles_per_seg = (tiles_total + NSEG - 1) // NSEG
    t0 = seg * tiles_per_seg
    t1 = tl.minimum(t0 + tiles_per_seg, tiles_total)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, DS], tl.float32)

    for t in range(t0, t1):
        pos = t * TILE + tl.arange(0, TILE)
        k_ok = pos < kv_len
        blk = tl.load(bt_ptr + req * stride_bt + pos // BLOCK_SIZE, mask=k_ok, other=0)
        slot = pos % BLOCK_SIZE
        m_t = tl.load(mt_ptr + pidx * stride_mt + t, mask=row_ok, other=float("-inf"))
        m_new = tl.maximum(m_i, m_t)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        w = tl.exp(tl.where(m_t == float("-inf"), float("-inf"), m_t - m_safe))
        alpha = tl.exp(tl.where(m_i == float("-inf"), float("-inf"), m_i - m_safe))
        p = tl.load(p_ptr + pidx[:, None] * stride_pw + pos[None, :],
                    mask=row_ok[:, None] & k_ok[None, :], other=0.0)     # bf16
        l_i = l_i * alpha + tl.sum(p.to(tl.float32), 1) * w
        v_ptrs = v_ptr + blk[:, None] * stride_vb + slot[:, None] * stride_vs + kvh * stride_vh + d[None, :]
        v = tl.load(v_ptrs, mask=k_ok[:, None], other=0.0).to(tl.bfloat16)
        if USE_PTH_SCALES:
            vs = tl.load(vs_ptr + blk * stride_vsb + slot * stride_vss + kvh * stride_vsh,
                         mask=k_ok, other=1.0)
            pv = (p.to(tl.float32) * (w[:, None] * vs[None, :])).to(tl.bfloat16)
        else:
            pv = (p.to(tl.float32) * w[:, None]).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(pv, v).to(tl.float32)
        m_i = m_new

    pidx_seg = pidx * NSEG + seg
    tl.store(part_o_ptr + pidx_seg[:, None] * D + d[None, :], acc, mask=row_ok[:, None])
    if dsi == 0:
        tl.store(part_m_ptr + pidx_seg, m_i, mask=row_ok)
        tl.store(part_l_ptr + pidx_seg, l_i, mask=row_ok)


@triton.jit
def _spec_attn_combine(
    part_o_ptr, part_m_ptr, part_l_ptr, out_ptr, cu_q_ptr,
    stride_ot, stride_oh,
    Hq: tl.constexpr, QMAX: tl.constexpr, D: tl.constexpr, NSEG: tl.constexpr,
):
    req = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + req)
    q_len = tl.load(cu_q_ptr + req + 1) - q_start
    if i < q_len:
        base = ((req * Hq + h) * QMAX + i) * NSEG
        segs = tl.arange(0, NSEG)
        m = tl.load(part_m_ptr + base + segs)
        l = tl.load(part_l_ptr + base + segs)
        m_max = tl.max(m, 0)
        m_max = tl.where(m_max == float("-inf"), 0.0, m_max)
        w = tl.exp(m - m_max)                       # segments with -inf give 0
        l_tot = tl.sum(l * w, 0)
        d = tl.arange(0, D)
        o = tl.load(part_o_ptr + (base + segs)[:, None] * D + d[None, :])   # [NSEG, D]
        o = tl.sum(o * w[:, None], 0) / tl.maximum(l_tot, 1e-30)
        tl.store(out_ptr + (q_start + i) * stride_ot + h * stride_oh + d, o.to(out_ptr.dtype.element_ty))


class SpecDecodeAttention:
    """Holds the partial buffers; call .run(...) per layer."""

    def __init__(self, max_num_reqs, num_heads, head_dim, device, num_segments=NUM_SEGMENTS):
        self.nseg = num_segments
        self.qmax = None
        self.max_num_reqs = max_num_reqs
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.part_o = None
        self.p_ws = None
        self.mt_ws = None
        self.ws_kv_stride = 0

    def _ensure(self, G):
        qmax = BLOCK_M // G
        if self.part_o is None or self.qmax != qmax:
            self.qmax = qmax
            n = self.max_num_reqs * self.num_heads * qmax * self.nseg
            self.part_o = torch.empty(n, self.head_dim, dtype=torch.float32, device=self.device)
            self.part_m = torch.empty(n, dtype=torch.float32, device=self.device)
            self.part_l = torch.empty(n, dtype=torch.float32, device=self.device)

    def _ensure_ws(self, num_reqs, kv_stride, tile):
        """Grow-only P (bf16) + per-tile-max (fp32) workspaces, sized by the
        actual batch (num_reqs), not the max_num_reqs upper bound."""
        n_rows = num_reqs * self.num_heads * self.qmax
        need_rows = n_rows
        if self.p_ws is None or kv_stride > self.ws_kv_stride or need_rows > self.p_ws.shape[0]:
            rows = max(need_rows, self.p_ws.shape[0] if self.p_ws is not None else 0)
            self.ws_kv_stride = max(kv_stride, self.ws_kv_stride)
            self.p_ws = torch.empty(rows, self.ws_kv_stride, dtype=torch.bfloat16, device=self.device)
            self.mt_ws = torch.empty(rows, self.ws_kv_stride // tile, dtype=torch.float32, device=self.device)

    def twopass_fits(self, num_reqs, kv_stride):
        n_rows = num_reqs * self.num_heads * self.qmax
        return n_rows * kv_stride * 2 <= _P_BUF_CAP

    def run(self, q, key_cache, value_cache, out, cu_seqlens_q, seqused_k, block_table, scale, num_reqs, max_query_len,
            k_scale_cache=None, v_scale_cache=None, gsplit=1, twopass=None, int4=False,
            tile=None, warps=4, stages=1, dsplit=2, warps2=4, stages2=2):
        use_pth = k_scale_cache is not None
        Hq, D = q.shape[1], q.shape[2]
        if int4:
            # RHT is linear: rotate in fp32, fold softmax_scale/head_size into
            # q afterwards (kernel expects pre-scaled rotated q).
            q_rot = _rht(q.float())
            return self._run_int4(q_rot, key_cache, value_cache, out, cu_seqlens_q,
                                  seqused_k, block_table, scale / D, num_reqs,
                                  max_query_len, k_scale_cache, v_scale_cache,
                                  gsplit=gsplit, tile=tile, warps=warps, stages=stages,
                                  out_dtype=q.dtype)
        if use_pth:
            ks_arg, vs_arg = k_scale_cache, v_scale_cache
            kss = (k_scale_cache.stride(0), k_scale_cache.stride(1), k_scale_cache.stride(2))
            vss = (v_scale_cache.stride(0), v_scale_cache.stride(1), v_scale_cache.stride(2))
        else:
            ks_arg, vs_arg = key_cache, value_cache  # dummy pointers (dead code)
            kss = vss = (0, 0, 0)
        Hq, D = q.shape[1], q.shape[2]
        Hkv = key_cache.shape[2]
        block_size = key_cache.shape[1]
        G_full = Hq // Hkv
        G = G_full // gsplit
        assert gsplit * G == G_full, "gsplit must divide G"
        assert max_query_len * G <= BLOCK_M, "too many query rows per request for this kernel"
        self._ensure(G)
        assert num_reqs <= self.max_num_reqs
        # shared memory on sm86 is 99 KB: q tile + one K and one V tile + scores must fit
        rows = max_query_len * G
        block_m = 16 if rows <= 16 else (32 if rows <= 32 else 64)
        if tile is None:
            tile = 64 if (block_m <= 32 or D <= 128) else 32
        if twopass is None:
            twopass = _twopass_default()
        kv_cap = block_table.shape[1] * block_size
        if twopass:
            kv_stride = (kv_cap + tile - 1) // tile * tile
            twopass = self.twopass_fits(num_reqs, kv_stride)
        if twopass:
            self._ensure_ws(num_reqs, kv_stride, tile)
            DS = D // dsplit
            grid1 = (num_reqs, Hkv * gsplit, self.nseg)
            _spec_p1_score[grid1](
                q, key_cache, block_table, seqused_k, cu_seqlens_q, ks_arg,
                self.p_ws, self.mt_ws,
                scale,
                q.stride(0), q.stride(1),
                key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
                block_table.stride(0),
                kss[0], kss[1], kss[2],
                kv_stride, kv_stride // tile,
                G=G, Hq=Hq, QMAX=self.qmax, D=D, BLOCK_SIZE=block_size,
                BLOCK_M=block_m, TILE=tile, NSEG=self.nseg,
                USE_PTH_SCALES=use_pth, GSPLIT=gsplit,
                num_warps=warps, num_stages=stages,
            )
            grid2 = (num_reqs, Hkv * gsplit, self.nseg * dsplit)
            _spec_p2_pv[grid2](
                self.p_ws, self.mt_ws, value_cache, block_table, seqused_k, cu_seqlens_q, vs_arg,
                self.part_o, self.part_m, self.part_l,
                value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                block_table.stride(0),
                vss[0], vss[1], vss[2],
                kv_stride, kv_stride // tile,
                G=G, Hq=Hq, QMAX=self.qmax, D=D, DS=DS, BLOCK_SIZE=block_size,
                BLOCK_M=block_m, TILE=tile, NSEG=self.nseg,
                USE_PTH_SCALES=use_pth, GSPLIT=gsplit, DSPLIT=dsplit,
                num_warps=warps2, num_stages=stages2,
            )
        else:
            grid = (num_reqs, Hkv * gsplit, self.nseg)
            _spec_attn_partial[grid](
                q, key_cache, value_cache, block_table, seqused_k, cu_seqlens_q,
                ks_arg, vs_arg,
                self.part_o, self.part_m, self.part_l,
                scale,
                q.stride(0), q.stride(1),
                key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
                value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                block_table.stride(0),
                kss[0], kss[1], kss[2],
                vss[0], vss[1], vss[2],
                G=G, Hq=Hq, QMAX=self.qmax, D=D, BLOCK_SIZE=block_size, BLOCK_M=block_m, TILE=tile, NSEG=self.nseg,
                USE_PTH_SCALES=use_pth, GSPLIT=gsplit,
                num_warps=warps, num_stages=stages,
            )
        _spec_attn_combine[(num_reqs, Hq, max_query_len)](
            self.part_o, self.part_m, self.part_l, out, cu_seqlens_q,
            out.stride(0), out.stride(1),
            Hq=Hq, QMAX=self.qmax, D=D, NSEG=self.nseg, num_warps=4,
        )
        return out

    def _run_int4(self, q, key_cache, value_cache, out, cu_seqlens_q, seqused_k,
                  block_table, scale, num_reqs, max_query_len,
                  k_scale_cache, v_scale_cache, gsplit=1, tile=None, warps=4, stages=1,
                  out_dtype=torch.bfloat16):
        """q is already RHT-rotated (fp32); scale is softmax_scale / head_size.
        Applies the inverse RHT to *out* after the combine."""
        Hq, D = q.shape[1], q.shape[2]
        # Packed int4 data occupies D//2 bytes per (token, head) row.  In-situ
        # the cache row additionally carries a trailing 4-byte fp32 scale
        # (reached via the separate scale views, never through the data view),
        # so accept both D//2 (standalone tests) and D//2+4 (live cache).
        DP = D // 2
        assert key_cache.shape[3] in (DP, DP + 4, DP + 16) and value_cache.shape[3] in (DP, DP + 4, DP + 16), \
            f"int4 packed cache last dim {key_cache.shape[3]} must be head_dim//2 (+4B scale pad)"
        q = (q * scale).to(out_dtype)
        Hkv = key_cache.shape[2]
        block_size = key_cache.shape[1]
        G_full = Hq // Hkv
        G = G_full // gsplit
        assert gsplit * G == G_full, "gsplit must divide G"
        assert max_query_len * G <= BLOCK_M
        self._ensure(G)
        assert num_reqs <= self.max_num_reqs
        rows = max_query_len * G
        block_m = 16 if rows <= 16 else (32 if rows <= 32 else 64)
        if tile is None:
            # int4 live-layout sweep (8 reqs, 15K/40K/100K KV, q_len=4):
            # TILE=32 beats 64 by 1.5-1.9x (strided NHD rows make wide tiles
            # inefficient); see bench_spec_int4_live.py.
            tile = 32
        kss = (k_scale_cache.stride(0), k_scale_cache.stride(1), k_scale_cache.stride(2))
        vss = (v_scale_cache.stride(0), v_scale_cache.stride(1), v_scale_cache.stride(2))
        grid = (num_reqs, Hkv * gsplit, self.nseg)
        _spec_attn_partial_int4[grid](
            q, key_cache, value_cache, block_table, seqused_k, cu_seqlens_q,
            k_scale_cache, v_scale_cache,
            self.part_o, self.part_m, self.part_l,
            q.stride(0), q.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            block_table.stride(0),
            kss[0], kss[1], kss[2],
            vss[0], vss[1], vss[2],
            G=G, Hq=Hq, QMAX=self.qmax, D=D, DP=DP, BLOCK_SIZE=block_size,
            BLOCK_M=block_m, TILE=tile, NSEG=self.nseg, GSPLIT=gsplit,
            num_warps=warps, num_stages=stages,
        )
        _spec_attn_combine[(num_reqs, Hq, max_query_len)](
            self.part_o, self.part_m, self.part_l, out, cu_seqlens_q,
            out.stride(0), out.stride(1),
            Hq=Hq, QMAX=self.qmax, D=D, NSEG=self.nseg, num_warps=4,
        )
        out_f = _rht(out.float(), inverse=True) / D
        out.copy_(out_f.to(out.dtype))
        return out
