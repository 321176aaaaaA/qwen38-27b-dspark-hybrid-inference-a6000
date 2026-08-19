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
"""
import torch
import triton
import triton.language as tl

NUM_SEGMENTS = 16
BLOCK_M = 64      # max query rows (q_len * G) per request handled by one program


@triton.jit
def _spec_attn_partial(
    q_ptr, k_ptr, v_ptr, bt_ptr, seqused_ptr, cu_q_ptr,
    part_o_ptr, part_m_ptr, part_l_ptr,
    scale,
    stride_qt, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_bt,
    G: tl.constexpr, Hq: tl.constexpr, QMAX: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, TILE: tl.constexpr, NSEG: tl.constexpr,
):
    req = tl.program_id(0)
    kvh = tl.program_id(1)
    seg = tl.program_id(2)

    q_start = tl.load(cu_q_ptr + req)
    q_len = tl.load(cu_q_ptr + req + 1) - q_start
    kv_len = tl.load(seqused_ptr + req)

    # rows: r = i * G + g  -> query token i (0..q_len-1), head kvh*G + g
    r = tl.arange(0, BLOCK_M)
    ri = r // G
    rg = r % G
    row_ok = ri < q_len
    q_pos = kv_len - q_len + ri                      # kv position of each query row
    d = tl.arange(0, D)
    q_ptrs = q_ptr + (q_start + ri)[:, None] * stride_qt + (kvh * G + rg)[:, None] * stride_qh + d[None, :]
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
        k = tl.load(k_ptrs, mask=k_ok[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=k_ok[:, None], other=0.0)
        s = tl.dot(qs, tl.trans(k)).to(tl.float32)            # [BLOCK_M, TILE]
        allowed = k_ok[None, :] & (pos[None, :] <= q_pos[:, None]) & row_ok[:, None]
        s = tl.where(allowed, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(tl.where(m_i == float("-inf"), float("-inf"), m_i - m_safe))
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        m_i = m_new

    # store partials at flat index ((req*Hq + head)*QMAX + i)*NSEG + seg
    hrow = kvh * G + rg
    pidx = ((req * Hq + hrow) * QMAX + ri) * NSEG + seg
    tl.store(part_o_ptr + pidx[:, None] * D + d[None, :], acc, mask=row_ok[:, None])
    tl.store(part_m_ptr + pidx, m_i, mask=row_ok)
    tl.store(part_l_ptr + pidx, l_i, mask=row_ok)


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

    def _ensure(self, G):
        qmax = BLOCK_M // G
        if self.part_o is None or self.qmax != qmax:
            self.qmax = qmax
            n = self.max_num_reqs * self.num_heads * qmax * self.nseg
            self.part_o = torch.empty(n, self.head_dim, dtype=torch.float32, device=self.device)
            self.part_m = torch.empty(n, dtype=torch.float32, device=self.device)
            self.part_l = torch.empty(n, dtype=torch.float32, device=self.device)

    def run(self, q, key_cache, value_cache, out, cu_seqlens_q, seqused_k, block_table, scale, num_reqs, max_query_len):
        Hq, D = q.shape[1], q.shape[2]
        Hkv = key_cache.shape[2]
        G = Hq // Hkv
        assert max_query_len * G <= BLOCK_M, "too many query rows per request for this kernel"
        self._ensure(G)
        assert num_reqs <= self.max_num_reqs
        # shared memory on sm86 is 99 KB: q tile + one K and one V tile + scores must fit
        rows = max_query_len * G
        block_m = 32 if rows <= 32 else 64
        tile = 64 if (block_m == 32 or D <= 128) else 32
        grid = (num_reqs, Hkv, self.nseg)
        _spec_attn_partial[grid](
            q, key_cache, value_cache, block_table, seqused_k, cu_seqlens_q,
            self.part_o, self.part_m, self.part_l,
            scale,
            q.stride(0), q.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            block_table.stride(0),
            G=G, Hq=Hq, QMAX=self.qmax, D=D, BLOCK_SIZE=key_cache.shape[1], BLOCK_M=block_m, TILE=tile, NSEG=self.nseg,
            num_warps=4, num_stages=1,
        )
        _spec_attn_combine[(num_reqs, Hq, max_query_len)](
            self.part_o, self.part_m, self.part_l, out, cu_seqlens_q,
            out.stride(0), out.stride(1),
            Hq=Hq, QMAX=self.qmax, D=D, NSEG=self.nseg, num_warps=4,
        )
        return out
