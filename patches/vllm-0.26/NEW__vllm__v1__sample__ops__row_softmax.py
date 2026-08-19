"""syv patch: multi-block row softmax for a handful of very wide rows.

torch.softmax launches one thread block per row; for a single request's
248k-entry logits row that is one SM doing ~140 us of work per call, and the
spec-decode sampler calls it several times per step. This splits every row over
NCHUNK blocks (two launches, ~10 us).
"""
import torch
import triton
import triton.language as tl

_NCHUNK = 64
_BLOCK = 4096


@triton.jit
def _partial_kernel(x_ptr, pm_ptr, ps_ptr, V, stride_x, NCHUNK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1)
    per = (V + NCHUNK - 1) // NCHUNK
    start = c * per
    end = tl.minimum(start + per, V)
    m = float("-inf")
    s = 0.0
    for off in range(start, end, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row * stride_x + idx, mask=idx < end, other=float("-inf")).to(tl.float32)
        bm = tl.max(x, 0)
        m_new = tl.maximum(m, bm)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        s = s * tl.exp(tl.where(m == float("-inf"), float("-inf"), m - m_safe)) + tl.sum(tl.exp(x - m_safe), 0)
        m = m_new
    tl.store(pm_ptr + row * NCHUNK + c, m)
    tl.store(ps_ptr + row * NCHUNK + c, s)


@triton.jit
def _final_kernel(x_ptr, out_ptr, pm_ptr, ps_ptr, V, stride_x, stride_o, NCHUNK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1)
    cs = tl.arange(0, NCHUNK)
    pm = tl.load(pm_ptr + row * NCHUNK + cs)
    ps = tl.load(ps_ptr + row * NCHUNK + cs)
    M = tl.max(pm, 0)
    M_safe = tl.where(M == float("-inf"), 0.0, M)
    S = tl.sum(ps * tl.exp(tl.where(pm == float("-inf"), float("-inf"), pm - M_safe)), 0)
    inv = 1.0 / tl.maximum(S, 1e-38)
    per = (V + NCHUNK - 1) // NCHUNK
    start = c * per
    end = tl.minimum(start + per, V)
    for off in range(start, end, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row * stride_x + idx, mask=idx < end, other=float("-inf")).to(tl.float32)
        y = tl.exp(x - M_safe) * inv
        tl.store(out_ptr + row * stride_o + idx, y, mask=idx < end)


def row_softmax_fp32(x: torch.Tensor) -> torch.Tensor:
    """softmax(x, dim=-1) in float32 for a 2-D tensor with few rows and a wide last dim."""
    assert x.ndim == 2 and x.stride(1) == 1
    B, V = x.shape
    out = torch.empty(B, V, dtype=torch.float32, device=x.device)
    pm = torch.empty(B, _NCHUNK, dtype=torch.float32, device=x.device)
    ps = torch.empty(B, _NCHUNK, dtype=torch.float32, device=x.device)
    _partial_kernel[(B, _NCHUNK)](x, pm, ps, V, x.stride(0), NCHUNK=_NCHUNK, BLOCK=_BLOCK, num_warps=4)
    _final_kernel[(B, _NCHUNK)](x, out, pm, ps, V, x.stride(0), out.stride(0), NCHUNK=_NCHUNK, BLOCK=_BLOCK, num_warps=4)
    return out


def softmax_fp32(x: torch.Tensor) -> torch.Tensor:
    """Drop-in for x.softmax(dim=-1, dtype=torch.float32); fast path for <= 16 wide rows."""
    if x.is_cuda and x.ndim == 2 and x.shape[0] <= 16 and x.shape[1] >= 16384 and x.stride(1) == 1:
        return row_softmax_fp32(x)
    return x.softmax(dim=-1, dtype=torch.float32)
