import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from . import kernels

try:
    from flash_attn import flash_attn_func
except Exception:
    flash_attn_func = None

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


def _next_pow2_cap(v: int, cap: int) -> int:
    v = int(max(1, v))
    out = 1
    while out < v and out < cap:
        out <<= 1
    return int(min(out, cap))


if triton is not None:
    @triton.jit
    def _echo_qk_page_argmax_kernel(
        q_ptr,
        mid_ptr,
        out_ptr,
        stride_q_b,
        stride_q_h,
        stride_q_d,
        stride_m_b,
        stride_m_t,
        stride_m_h,
        stride_m_d,
        stride_o_b,
        stride_o_h,
        stride_o_p,
        n_heads,
        mid_pages,
        head_dim,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pages_per_batch = n_heads * mid_pages
        b = pid // pages_per_batch
        rem = pid - b * pages_per_batch
        h = rem // mid_pages
        p = rem - h * mid_pages

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        q_ptrs = q_ptr + b * stride_q_b + h * stride_q_h + offs_d * stride_q_d
        q_vec = tl.load(q_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        offs_t = tl.arange(0, BLOCK_T)
        best_score = -float("inf")
        best_idx = 0
        page_base = p * PAGE_SIZE

        for t0 in tl.static_range(0, PAGE_SIZE, BLOCK_T):
            tok = t0 + offs_t
            mask_t = tok < PAGE_SIZE
            mid_ptrs = (
                mid_ptr
                + b * stride_m_b
                + (page_base + tok)[:, None] * stride_m_t
                + h * stride_m_h
                + offs_d[None, :] * stride_m_d
            )
            k_mat = tl.load(
                mid_ptrs,
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(k_mat * q_vec[None, :], axis=1)
            scores = tl.where(mask_t, scores, -float("inf"))
            blk_score = tl.max(scores, axis=0)
            blk_idx = tl.argmax(scores, axis=0)
            blk_idx = (t0 + blk_idx).to(tl.int32)
            better = blk_score > best_score
            best_score = tl.where(better, blk_score, best_score)
            best_idx = tl.where(better, blk_idx, best_idx)

        out_off = b * stride_o_b + h * stride_o_h + p * stride_o_p
        tl.store(out_ptr + out_off, best_idx)


if triton is not None:
    @triton.jit
    def _echo_flash_decode_fused_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        best_ptr,
        stride_q_b,
        stride_q_h,
        stride_q_d,
        stride_k_b,
        stride_k_h,
        stride_k_t,
        stride_k_d,
        stride_v_b,
        stride_v_h,
        stride_v_t,
        stride_v_d,
        stride_o_b,
        stride_o_h,
        stride_o_d,
        stride_best_b,
        stride_best_h,
        stride_best_p,
        n_q_heads,
        n_q_per_kv,
        seq_len,
        head_dim,
        scale,
        BLOCK_D: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        MAX_TOKENS: tl.constexpr,
        MID_START: tl.constexpr,
        MID_END: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // n_q_heads
        qh = pid - b * n_q_heads
        kvh = qh // n_q_per_kv

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        q_ptrs = q_ptr + b * stride_q_b + qh * stride_q_h + offs_d * stride_q_d
        q_vec = tl.load(q_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for n0 in tl.static_range(0, MAX_TOKENS, PAGE_SIZE):
            offs_t = n0 + tl.arange(0, PAGE_SIZE)
            mask_t = offs_t < seq_len

            k_ptrs = (
                k_ptr
                + b * stride_k_b
                + kvh * stride_k_h
                + offs_t[:, None] * stride_k_t
                + offs_d[None, :] * stride_k_d
            )
            v_ptrs = (
                v_ptr
                + b * stride_v_b
                + kvh * stride_v_h
                + offs_t[:, None] * stride_v_t
                + offs_d[None, :] * stride_v_d
            )
            k_mat = tl.load(k_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(
                tl.float32
            )
            v_mat = tl.load(v_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(
                tl.float32
            )

            scores = tl.sum(k_mat * q_vec[None, :], axis=1) * scale
            scores = tl.where(mask_t, scores, -float("inf"))

            if n0 >= MID_START and n0 < MID_END:
                page_idx = (n0 - MID_START) // PAGE_SIZE
                best_local = tl.argmax(scores, axis=0).to(tl.int32)
                b_ptr = (
                    best_ptr
                    + b * stride_best_b
                    + qh * stride_best_h
                    + page_idx * stride_best_p
                )
                tl.store(b_ptr, best_local)

            m_ij = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v_mat, axis=0)
            m_i = m_new

        out = acc / l_i
        o_ptrs = o_ptr + b * stride_o_b + qh * stride_o_h + offs_d * stride_o_d
        tl.store(o_ptrs, out, mask=mask_d)


if triton is not None:
    @triton.jit
    def _echo_flash_decode_plain_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        stride_q_b,
        stride_q_h,
        stride_q_d,
        stride_k_b,
        stride_k_h,
        stride_k_t,
        stride_k_d,
        stride_v_b,
        stride_v_h,
        stride_v_t,
        stride_v_d,
        stride_o_b,
        stride_o_h,
        stride_o_d,
        n_q_heads,
        n_q_per_kv,
        seq_len,
        head_dim,
        scale,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        MAX_TOKENS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // n_q_heads
        qh = pid - b * n_q_heads
        kvh = qh // n_q_per_kv

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        q_ptrs = q_ptr + b * stride_q_b + qh * stride_q_h + offs_d * stride_q_d
        q_vec = tl.load(q_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for n0 in tl.static_range(0, MAX_TOKENS, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_len
            k_ptrs = (
                k_ptr
                + b * stride_k_b
                + kvh * stride_k_h
                + offs_n[:, None] * stride_k_t
                + offs_d[None, :] * stride_k_d
            )
            v_ptrs = (
                v_ptr
                + b * stride_v_b
                + kvh * stride_v_h
                + offs_n[:, None] * stride_v_t
                + offs_d[None, :] * stride_v_d
            )
            k_mat = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(
                tl.float32
            )
            v_mat = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(
                tl.float32
            )

            scores = tl.sum(k_mat * q_vec[None, :], axis=1) * scale
            scores = tl.where(mask_n, scores, -float("inf"))

            m_ij = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v_mat, axis=0)
            m_i = m_new

        out = acc / l_i
        o_ptrs = o_ptr + b * stride_o_b + qh * stride_o_h + offs_d * stride_o_d
        tl.store(o_ptrs, out, mask=mask_d)


if triton is not None:
    @triton.jit
    def _echo_flash_decode_qk_select_kernel(
        q_ptr,
        k_ptr,
        scores_ptr,
        best_ptr,
        stride_q_b,
        stride_q_h,
        stride_q_d,
        stride_k_b,
        stride_k_h,
        stride_k_t,
        stride_k_d,
        stride_s_b,
        stride_s_h,
        stride_s_t,
        stride_best_b,
        stride_best_h,
        stride_best_p,
        n_q_heads,
        n_q_per_kv,
        seq_len,
        head_dim,
        scale,
        BLOCK_D: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        MAX_TOKENS: tl.constexpr,
        MID_START: tl.constexpr,
        MID_END: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // n_q_heads
        qh = pid - b * n_q_heads
        kvh = qh // n_q_per_kv

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        q_ptrs = q_ptr + b * stride_q_b + qh * stride_q_h + offs_d * stride_q_d
        q_vec = tl.load(q_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        for n0 in tl.static_range(0, MAX_TOKENS, PAGE_SIZE):
            offs_t = n0 + tl.arange(0, PAGE_SIZE)
            mask_t = offs_t < seq_len
            k_ptrs = (
                k_ptr
                + b * stride_k_b
                + kvh * stride_k_h
                + offs_t[:, None] * stride_k_t
                + offs_d[None, :] * stride_k_d
            )
            k_mat = tl.load(k_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(
                tl.float32
            )
            scores = tl.sum(k_mat * q_vec[None, :], axis=1) * scale

            s_ptrs = scores_ptr + b * stride_s_b + qh * stride_s_h + offs_t * stride_s_t
            tl.store(s_ptrs, scores, mask=mask_t)

            if n0 >= MID_START and n0 < MID_END:
                page_idx = (n0 - MID_START) // PAGE_SIZE
                best_local = tl.argmax(tl.where(mask_t, scores, -float("inf")), axis=0).to(
                    tl.int32
                )
                b_ptr = (
                    best_ptr
                    + b * stride_best_b
                    + qh * stride_best_h
                    + page_idx * stride_best_p
                )
                tl.store(b_ptr, best_local)


if triton is not None:
    @triton.jit
    def _echo_flash_decode_pv_kernel(
        scores_ptr,
        v_ptr,
        o_ptr,
        stride_s_b,
        stride_s_h,
        stride_s_t,
        stride_v_b,
        stride_v_h,
        stride_v_t,
        stride_v_d,
        stride_o_b,
        stride_o_h,
        stride_o_d,
        n_q_heads,
        n_q_per_kv,
        seq_len,
        head_dim,
        BLOCK_D: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        MAX_TOKENS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // n_q_heads
        qh = pid - b * n_q_heads
        kvh = qh // n_q_per_kv

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim

        m_i = -float("inf")
        for n0 in tl.static_range(0, MAX_TOKENS, PAGE_SIZE):
            offs_t = n0 + tl.arange(0, PAGE_SIZE)
            mask_t = offs_t < seq_len
            s_ptrs = scores_ptr + b * stride_s_b + qh * stride_s_h + offs_t * stride_s_t
            s = tl.load(s_ptrs, mask=mask_t, other=-float("inf"))
            m_i = tl.maximum(m_i, tl.max(s, axis=0))

        l_i = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for n0 in tl.static_range(0, MAX_TOKENS, PAGE_SIZE):
            offs_t = n0 + tl.arange(0, PAGE_SIZE)
            mask_t = offs_t < seq_len
            s_ptrs = scores_ptr + b * stride_s_b + qh * stride_s_h + offs_t * stride_s_t
            s = tl.load(s_ptrs, mask=mask_t, other=-float("inf"))
            p = tl.exp(s - m_i)
            p = tl.where(mask_t, p, 0.0)
            l_i += tl.sum(p, axis=0)

            v_ptrs = (
                v_ptr
                + b * stride_v_b
                + kvh * stride_v_h
                + offs_t[:, None] * stride_v_t
                + offs_d[None, :] * stride_v_d
            )
            v_mat = tl.load(v_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(
                tl.float32
            )
            acc += tl.sum(p[:, None] * v_mat, axis=0)

        out = acc / l_i
        o_ptrs = o_ptr + b * stride_o_b + qh * stride_o_h + offs_d * stride_o_d
        tl.store(o_ptrs, out, mask=mask_d)


def _triton_qk_page_argmax(
    grouped_q: torch.Tensor,
    mid: torch.Tensor,
    mid_pages: int,
    page_size: int,
) -> Optional[torch.Tensor]:
    if triton is None:
        return None
    if grouped_q.device.type != "cuda" or mid.device.type != "cuda":
        return None
    if grouped_q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if mid.dtype != grouped_q.dtype:
        return None
    if page_size <= 0 or mid_pages <= 0:
        return None
    if mid.shape[1] != int(mid_pages * page_size):
        return None

    eff_bsz, sampled_heads, head_dim = grouped_q.shape
    if head_dim > 256:
        return None
    out = torch.empty(
        (eff_bsz, sampled_heads, mid_pages),
        dtype=torch.int32,
        device=grouped_q.device,
    )
    block_t = _next_pow2_cap(page_size, 128)
    block_d = _next_pow2_cap(head_dim, 256)
    num_warps = 4 if block_t <= 64 and block_d <= 128 else 8
    grid = (eff_bsz * sampled_heads * mid_pages,)

    _echo_qk_page_argmax_kernel[grid](
        grouped_q,
        mid,
        out,
        grouped_q.stride(0),
        grouped_q.stride(1),
        grouped_q.stride(2),
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        mid.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        sampled_heads,
        mid_pages,
        head_dim,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        PAGE_SIZE=page_size,
        num_warps=num_warps,
    )
    return out


def _triton_flash_decode_attn_with_page_max(
    q: torch.Tensor,            # [b, n_q_heads, d]
    k: torch.Tensor,            # [b, n_kv_heads, s, d]
    v: torch.Tensor,            # [b, n_kv_heads, s, d]
    n_q_per_kv: int,
    page_size: int,
    mid_start: int,
    mid_pages: int,
    max_tokens: int,
):
    if triton is None:
        return None, None
    if q.device.type != "cuda" or k.device.type != "cuda" or v.device.type != "cuda":
        return None, None
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None, None
    if not (q.dtype == k.dtype == v.dtype):
        return None, None
    if q.dim() != 3 or k.dim() != 4 or v.dim() != 4:
        return None, None

    bsz, n_q_heads, head_dim = q.shape
    if head_dim <= 0 or head_dim > 256:
        return None, None
    if n_q_per_kv <= 0 or (n_q_heads % n_q_per_kv) != 0:
        return None, None
    n_kv_heads = n_q_heads // n_q_per_kv
    if k.shape[0] != bsz or v.shape[0] != bsz:
        return None, None
    if k.shape[1] != n_kv_heads or v.shape[1] != n_kv_heads:
        return None, None
    if k.shape[3] != head_dim or v.shape[3] != head_dim or k.shape[2] != v.shape[2]:
        return None, None

    seq_len = int(k.shape[2])
    if seq_len <= 0:
        return None, None
    if max_tokens < seq_len:
        return None, None
    if page_size <= 0 or (max_tokens % page_size) != 0:
        return None, None
    mid_tokens = int(mid_pages * page_size)
    mid_end = int(mid_start + mid_tokens)
    if mid_pages <= 0 or mid_start < 0 or mid_end > seq_len:
        return None, None
    if (mid_start % page_size) != 0:
        return None, None

    q_c = q if q.is_contiguous() else q.contiguous()
    k_c = k if k.is_contiguous() else k.contiguous()
    v_c = v if v.is_contiguous() else v.contiguous()

    out = torch.empty_like(q_c)
    best = torch.empty(
        (bsz, n_q_heads, mid_pages),
        dtype=torch.int32,
        device=q.device,
    )

    block_d = _next_pow2_cap(head_dim, 256)
    num_warps = 4 if block_d <= 128 else 8
    grid = (bsz * n_q_heads,)
    _echo_flash_decode_fused_kernel[grid](
        q_c,
        k_c,
        v_c,
        out,
        best,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        k_c.stride(0),
        k_c.stride(1),
        k_c.stride(2),
        k_c.stride(3),
        v_c.stride(0),
        v_c.stride(1),
        v_c.stride(2),
        v_c.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        best.stride(0),
        best.stride(1),
        best.stride(2),
        n_q_heads,
        n_q_per_kv,
        seq_len,
        head_dim,
        1.0 / math.sqrt(float(head_dim)),
        BLOCK_D=block_d,
        PAGE_SIZE=page_size,
        MAX_TOKENS=max_tokens,
        MID_START=mid_start,
        MID_END=mid_end,
        num_warps=num_warps,
    )
    return out, best


def _triton_flash_decode_qk_select(
    q: torch.Tensor,            # [b, n_q_heads, d]
    k: torch.Tensor,            # [b, n_kv_heads, s, d]
    n_q_per_kv: int,
    page_size: int,
    mid_start: int,
    mid_pages: int,
    max_tokens: int,
):
    if triton is None:
        return None, None
    if q.device.type != "cuda" or k.device.type != "cuda":
        return None, None
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None, None
    if q.dtype != k.dtype:
        return None, None
    if q.dim() != 3 or k.dim() != 4:
        return None, None

    bsz, n_q_heads, head_dim = q.shape
    if head_dim <= 0 or head_dim > 256:
        return None, None
    if n_q_per_kv <= 0 or (n_q_heads % n_q_per_kv) != 0:
        return None, None
    n_kv_heads = n_q_heads // n_q_per_kv
    if k.shape[0] != bsz or k.shape[1] != n_kv_heads or k.shape[3] != head_dim:
        return None, None

    seq_len = int(k.shape[2])
    if seq_len <= 0 or max_tokens < seq_len:
        return None, None
    if page_size <= 0 or (max_tokens % page_size) != 0:
        return None, None
    mid_tokens = int(mid_pages * page_size)
    mid_end = int(mid_start + mid_tokens)
    if mid_pages <= 0 or mid_start < 0 or mid_end > seq_len:
        return None, None
    if (mid_start % page_size) != 0:
        return None, None

    q_c = q if q.is_contiguous() else q.contiguous()
    k_c = k if k.is_contiguous() else k.contiguous()
    scores = torch.empty(
        (bsz, n_q_heads, max_tokens),
        dtype=torch.float32,
        device=q.device,
    )
    best = torch.empty(
        (bsz, n_q_heads, mid_pages),
        dtype=torch.int32,
        device=q.device,
    )

    block_d = _next_pow2_cap(head_dim, 256)
    num_warps = 4 if block_d <= 128 else 8
    grid = (bsz * n_q_heads,)
    _echo_flash_decode_qk_select_kernel[grid](
        q_c,
        k_c,
        scores,
        best,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        k_c.stride(0),
        k_c.stride(1),
        k_c.stride(2),
        k_c.stride(3),
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        best.stride(0),
        best.stride(1),
        best.stride(2),
        n_q_heads,
        n_q_per_kv,
        seq_len,
        head_dim,
        1.0 / math.sqrt(float(head_dim)),
        BLOCK_D=block_d,
        PAGE_SIZE=page_size,
        MAX_TOKENS=max_tokens,
        MID_START=mid_start,
        MID_END=mid_end,
        num_warps=num_warps,
    )
    return scores, best


def _triton_flash_decode_pv_from_scores(
    scores: torch.Tensor,       # [b, n_q_heads, max_tokens], float32
    v: torch.Tensor,            # [b, n_kv_heads, s, d]
    n_q_per_kv: int,
    seq_len: int,
):
    if triton is None:
        return None
    if scores.device.type != "cuda" or v.device.type != "cuda":
        return None
    if scores.dtype != torch.float32:
        return None
    if v.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if scores.dim() != 3 or v.dim() != 4:
        return None

    bsz, n_q_heads, max_tokens = scores.shape
    n_kv_heads = v.shape[1]
    head_dim = v.shape[3]
    if n_q_per_kv <= 0 or n_q_heads != n_kv_heads * n_q_per_kv:
        return None
    if seq_len <= 0 or seq_len > max_tokens or seq_len > int(v.shape[2]):
        return None
    if head_dim <= 0 or head_dim > 256:
        return None

    scores_c = scores if scores.is_contiguous() else scores.contiguous()
    v_c = v if v.is_contiguous() else v.contiguous()
    out = torch.empty(
        (bsz, n_q_heads, head_dim),
        dtype=torch.float32,
        device=scores.device,
    )

    page_size = 32
    if max_tokens % 128 == 0:
        page_size = 128
    elif max_tokens % 64 == 0:
        page_size = 64
    if max_tokens % page_size != 0:
        return None

    block_d = _next_pow2_cap(head_dim, 256)
    num_warps = 4 if block_d <= 128 else 8
    grid = (bsz * n_q_heads,)
    _echo_flash_decode_pv_kernel[grid](
        scores_c,
        v_c,
        out,
        scores_c.stride(0),
        scores_c.stride(1),
        scores_c.stride(2),
        v_c.stride(0),
        v_c.stride(1),
        v_c.stride(2),
        v_c.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        n_q_heads,
        n_q_per_kv,
        seq_len,
        head_dim,
        BLOCK_D=block_d,
        PAGE_SIZE=page_size,
        MAX_TOKENS=max_tokens,
        num_warps=num_warps,
    )
    return out.to(dtype=v.dtype)


def _triton_flash_decode_attn_plain(
    q: torch.Tensor,            # [b, n_q_heads, d]
    k: torch.Tensor,            # [b, n_kv_heads, s, d]
    v: torch.Tensor,            # [b, n_kv_heads, s, d]
    n_q_per_kv: int,
    max_tokens: int,
):
    if triton is None:
        return None
    if q.device.type != "cuda" or k.device.type != "cuda" or v.device.type != "cuda":
        return None
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if not (q.dtype == k.dtype == v.dtype):
        return None
    if q.dim() != 3 or k.dim() != 4 or v.dim() != 4:
        return None

    bsz, n_q_heads, head_dim = q.shape
    if head_dim <= 0 or head_dim > 256:
        return None
    if n_q_per_kv <= 0 or (n_q_heads % n_q_per_kv) != 0:
        return None
    n_kv_heads = n_q_heads // n_q_per_kv
    if k.shape[0] != bsz or v.shape[0] != bsz:
        return None
    if k.shape[1] != n_kv_heads or v.shape[1] != n_kv_heads:
        return None
    if k.shape[3] != head_dim or v.shape[3] != head_dim:
        return None
    if k.shape[2] != v.shape[2]:
        return None

    seq_len = int(k.shape[2])
    if seq_len <= 0 or max_tokens < seq_len:
        return None

    q_c = q if q.is_contiguous() else q.contiguous()
    k_c = k if k.is_contiguous() else k.contiguous()
    v_c = v if v.is_contiguous() else v.contiguous()
    out = torch.empty_like(q_c)

    block_d = _next_pow2_cap(head_dim, 256)
    block_n = 128 if max_tokens >= 128 else 64
    if block_n > max_tokens:
        block_n = _next_pow2_cap(max_tokens, 128)
    num_warps = 4 if block_d <= 128 else 8
    grid = (bsz * n_q_heads,)

    _echo_flash_decode_plain_kernel[grid](
        q_c,
        k_c,
        v_c,
        out,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        k_c.stride(0),
        k_c.stride(1),
        k_c.stride(2),
        k_c.stride(3),
        v_c.stride(0),
        v_c.stride(1),
        v_c.stride(2),
        v_c.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        n_q_heads,
        n_q_per_kv,
        seq_len,
        head_dim,
        1.0 / math.sqrt(float(head_dim)),
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        MAX_TOKENS=max_tokens,
        num_warps=num_warps,
    )
    return out


class EchoTokenPrefetchRuntime:
    """
    Token-wise EchoKV runtime:
    - CPU linear KV storage (pinned)
    - fixed-size GPU middle buffer
    - async prefetch on dedicated CUDA stream
    - page-aligned tiled starts to fill middle budget
    """

    def __init__(
        self,
        n_qo_heads: int,
        n_kv_heads: int,
        head_dim: int,
        page_size: int,
        n_sink_pages: int,
        n_win_pages: int,
        mid_pages: int,
        seed_anchors: int = 64,
        shared_batch: bool = True,
        use_cuda_token_recall: bool = True,
        anchor_head_sample: int = 0,
        use_triton_qk_select: bool = True,
        use_triton_flash_attn: bool = True,
        allow_anchor_overlap: bool = True,
        prefer_flash_attn_package: bool = True,
        enable_local_kv: bool = True,
        enable_page_kv: bool = True,
        stream_chunk_pages: int = 4,
        stream_prefetch_only: bool = True,
    ) -> None:
        self.n_qo_heads = n_qo_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.n_sink_pages = n_sink_pages
        self.n_win_pages = n_win_pages
        self.mid_pages = int(max(0, mid_pages))
        self.mid_tokens = self.mid_pages * self.page_size
        self.seed_anchors = int(max(1, seed_anchors))
        self.shared_batch = bool(shared_batch)
        self.use_cuda_token_recall = bool(use_cuda_token_recall)
        self.anchor_head_sample = int(anchor_head_sample)
        self.use_triton_qk_select = bool(use_triton_qk_select and (triton is not None))
        self.use_triton_flash_attn = bool(use_triton_flash_attn and (triton is not None))
        self.allow_anchor_overlap = bool(allow_anchor_overlap)
        self.prefer_flash_attn_package = bool(prefer_flash_attn_package)
        self.enable_local_kv = bool(enable_local_kv)
        self.enable_page_kv = bool(enable_page_kv)
        self.stream_chunk_pages = int(max(1, stream_chunk_pages))
        self.stream_prefetch_only = bool(stream_prefetch_only)
        self.n_q_per_kv = self.n_qo_heads // self.n_kv_heads
        self._inv_sqrt_head_dim = 1.0 / math.sqrt(self.head_dim)
        self.delta_max_pages = 8

        self.sink_tokens = self.n_sink_pages * self.page_size
        self.win_tokens = self.n_win_pages * self.page_size
        self.local_pages = self.n_sink_pages + self.mid_pages + self.n_win_pages

        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None
        self.recall_stream: Optional[torch.cuda.Stream] = None
        self.prefetch_event = None
        self.append_event = None

        self.batch_size = 0
        self.eff_batch = 0
        self.cpu_capacity = 0
        self.cpu_len = 0
        self.cpu_kv = None  # [bsz, cap, 2, n_kv_heads, head_dim], pinned

        self.sink_k = None  # [bsz, n_kv_heads, sink_tokens, head_dim]
        self.sink_v = None
        self.win_k = None   # [bsz, n_kv_heads, win_tokens, head_dim]
        self.win_v = None
        self.sink_len = 0
        self.win_len = 0
        self.win_ptr = 0

        self.local_k_buf = [None, None]  # [eff, n_kv_heads, total, head_dim]
        self.local_v_buf = [None, None]
        self.local_mid_ready = [False, False]
        self.local_total_cap = self.sink_tokens + self.mid_tokens + self.win_tokens
        self.local_sink_len_cached = [0, 0]
        self.page_kv_buf = [None, None]  # [eff, local_pages, 2, page_size, n_kv_heads, head_dim]
        self.page_mid_ready = [False, False]
        self.page_sink_ready = [False, False]
        self.page_win_ready = [False, False]
        self.page_ids = None  # [bsz, local_pages], int32
        self.page_ids_bsz = 0

        self.gpu_mid = [None, None]  # [eff_bsz, mid_tokens, 2, n_kv_heads, head_dim]
        self.pending_idx = 0
        self.active_idx = 0
        self.pending_seq_len = -1
        self.active_seq_len = -1
        self.pending_starts = None
        self.active_starts = None
        self.prefetch_ready = False

        self.anchors = None  # [eff_bsz, mid_pages], token indices
        self.prefetch_hint_starts = None  # [eff_bsz, mid_pages], int32 cpu
        self.prefetch_hint_seq_len = -1
        self.slide_half_window = self.page_size // 2
        self._head_idx = None

    def _effective_batch(self, bsz: int) -> int:
        if self.shared_batch and bsz > 1:
            return 1
        return bsz

    def ensure(self, bsz: int, device: torch.device, dtype: torch.dtype):
        if self.mid_pages <= 0:
            self.batch_size = bsz
            self.eff_batch = self._effective_batch(bsz)
            self.device = device
            self.dtype = dtype
            if self.recall_stream is None:
                self.recall_stream = torch.cuda.Stream(device)
                self.prefetch_event = torch.cuda.Event()
                self.append_event = torch.cuda.Event()
            elif self.append_event is None:
                self.append_event = torch.cuda.Event()
            return

        eff_bsz = self._effective_batch(bsz)
        need_alloc = (
            self.device != device
            or self.dtype != dtype
            or self.batch_size != bsz
            or self.eff_batch != eff_bsz
            or self.cpu_kv is None
            or self.gpu_mid[0] is None
        )
        if not need_alloc:
            return

        self.device = device
        self.dtype = dtype
        self.batch_size = bsz
        self.eff_batch = eff_bsz

        if self.recall_stream is None:
            self.recall_stream = torch.cuda.Stream(device)
            self.prefetch_event = torch.cuda.Event()
            self.append_event = torch.cuda.Event()
        elif self.append_event is None:
            self.append_event = torch.cuda.Event()

        min_cap = self.sink_tokens + self.win_tokens + self.mid_tokens + 16
        self.cpu_capacity = max(2048, min_cap)
        self.cpu_kv = torch.empty(
            (bsz, self.cpu_capacity, 2, self.n_kv_heads, self.head_dim),
            dtype=dtype,
            device=torch.device("cpu"),
            pin_memory=True,
        )
        self.gpu_mid[0] = torch.empty(
            (eff_bsz, self.mid_tokens, 2, self.n_kv_heads, self.head_dim),
            dtype=dtype,
            device=device,
        )
        self.gpu_mid[1] = torch.empty_like(self.gpu_mid[0])
        if self.enable_local_kv and self.local_total_cap > 0:
            self.local_k_buf[0] = torch.empty(
                (eff_bsz, self.n_kv_heads, self.local_total_cap, self.head_dim),
                dtype=dtype,
                device=device,
            )
            self.local_v_buf[0] = torch.empty_like(self.local_k_buf[0])
            self.local_k_buf[1] = torch.empty_like(self.local_k_buf[0])
            self.local_v_buf[1] = torch.empty_like(self.local_v_buf[0])
        else:
            self.local_k_buf = [None, None]
            self.local_v_buf = [None, None]
        if self.enable_page_kv and self.local_pages > 0:
            self.page_kv_buf[0] = torch.empty(
                (
                    eff_bsz,
                    self.local_pages,
                    2,
                    self.page_size,
                    self.n_kv_heads,
                    self.head_dim,
                ),
                dtype=dtype,
                device=device,
            )
            self.page_kv_buf[1] = torch.empty_like(self.page_kv_buf[0])
        else:
            self.page_kv_buf = [None, None]
        self.reset_state()

    def reset_state(self):
        self.cpu_len = 0
        self.sink_k = None
        self.sink_v = None
        self.win_k = None
        self.win_v = None
        self.sink_len = 0
        self.win_len = 0
        self.win_ptr = 0
        self.anchors = None
        self.pending_idx = 0
        self.active_idx = 0
        self.pending_seq_len = -1
        self.active_seq_len = -1
        self.pending_starts = None
        self.active_starts = None
        self.prefetch_ready = False
        self.prefetch_hint_starts = None
        self.prefetch_hint_seq_len = -1
        self.local_mid_ready = [False, False]
        self.local_sink_len_cached = [0, 0]
        self.page_mid_ready = [False, False]
        self.page_sink_ready = [False, False]
        self.page_win_ready = [False, False]
        self.page_ids = None
        self.page_ids_bsz = 0
        self._head_idx = None

    def _maybe_expand_cpu(self, needed_tokens: int):
        if needed_tokens <= self.cpu_capacity:
            return
        if self.recall_stream is not None:
            self.recall_stream.synchronize()
        old_cap = self.cpu_capacity
        new_cap = old_cap
        while new_cap < needed_tokens:
            new_cap *= 2
        new_buf = torch.empty(
            (self.batch_size, new_cap, 2, self.n_kv_heads, self.head_dim),
            dtype=self.cpu_kv.dtype,
            device=torch.device("cpu"),
            pin_memory=True,
        )
        if self.cpu_len > 0:
            new_buf[:, : self.cpu_len].copy_(self.cpu_kv[:, : self.cpu_len], non_blocking=False)
        self.cpu_kv = new_buf
        self.cpu_capacity = new_cap

    def copy_prefill_to_cpu(self, k: torch.Tensor, v: torch.Tensor):
        # k/v: [bsz, q_len, n_kv_heads, head_dim]
        q_len = k.shape[1]
        self._maybe_expand_cpu(q_len)
        self.cpu_kv[:, :q_len, 0].copy_(k, non_blocking=True)
        self.cpu_kv[:, :q_len, 1].copy_(v, non_blocking=True)
        self.cpu_len = q_len

        self.sink_len = min(self.sink_tokens, q_len)
        self.win_len = min(self.win_tokens, q_len)
        self.win_ptr = 0

        if self.sink_tokens > 0:
            self.sink_k = torch.empty(
                (self.batch_size, self.n_kv_heads, self.sink_tokens, self.head_dim),
                dtype=k.dtype,
                device=k.device,
            )
            self.sink_v = torch.empty_like(self.sink_k)
            if self.sink_len > 0:
                self.sink_k[:, :, : self.sink_len].copy_(
                    k[:, : self.sink_len].transpose(1, 2), non_blocking=True
                )
                self.sink_v[:, :, : self.sink_len].copy_(
                    v[:, : self.sink_len].transpose(1, 2), non_blocking=True
                )
        else:
            self.sink_k = None
            self.sink_v = None

        if self.win_tokens > 0:
            self.win_k = torch.empty(
                (self.batch_size, self.n_kv_heads, self.win_tokens, self.head_dim),
                dtype=k.dtype,
                device=k.device,
            )
            self.win_v = torch.empty_like(self.win_k)
            if self.win_len > 0:
                self.win_k[:, :, : self.win_len].copy_(
                    k[:, q_len - self.win_len : q_len].transpose(1, 2), non_blocking=True
                )
                self.win_v[:, :, : self.win_len].copy_(
                    v[:, q_len - self.win_len : q_len].transpose(1, 2), non_blocking=True
                )
        else:
            self.win_k = None
            self.win_v = None

        self._sync_sink_to_local_buffers()
        self._sync_sink_to_page_buffers()
        self._sync_window_to_local_buffers_full()
        self._sync_window_to_page_buffers_full()

    def append_decode_token_to_cpu(
        self,
        k_tok: torch.Tensor,
        v_tok: torch.Tensor,
        src_stream: Optional[torch.cuda.Stream] = None,
    ):
        # k_tok/v_tok: [bsz, 1, n_kv_heads, head_dim]
        needed = self.cpu_len + 1
        self._maybe_expand_cpu(needed)
        if src_stream is None:
            src_stream = torch.cuda.current_stream(self.device)
        self.append_event.record(src_stream)
        with torch.cuda.stream(self.recall_stream):
            self.recall_stream.wait_event(self.append_event)
            self.cpu_kv[:, self.cpu_len : self.cpu_len + 1, 0].copy_(k_tok, non_blocking=True)
            self.cpu_kv[:, self.cpu_len : self.cpu_len + 1, 1].copy_(v_tok, non_blocking=True)
        self.cpu_len = needed
        k_tok_h = k_tok.transpose(1, 2)
        v_tok_h = v_tok.transpose(1, 2)

        if self.sink_tokens > 0:
            if self.sink_len < self.sink_tokens:
                self.sink_k[:, :, self.sink_len : self.sink_len + 1].copy_(k_tok_h, non_blocking=True)
                self.sink_v[:, :, self.sink_len : self.sink_len + 1].copy_(v_tok_h, non_blocking=True)
                self.sink_len += 1
                self._sync_sink_to_local_buffers()
                self._sync_sink_to_page_buffers()

        if self.win_tokens > 0:
            if self.win_len < self.win_tokens:
                self.win_k[:, :, self.win_len : self.win_len + 1].copy_(k_tok_h, non_blocking=True)
                self.win_v[:, :, self.win_len : self.win_len + 1].copy_(v_tok_h, non_blocking=True)
                self.win_len += 1
                self._update_window_token_in_local_buffers(self.win_len - 1, k_tok_h, v_tok_h)
                if self.win_len == self.win_tokens:
                    self._sync_window_to_local_buffers_full()
                self._sync_window_to_page_buffers_full()
            else:
                write_slot = self.win_ptr
                self.win_k[:, :, write_slot : write_slot + 1].copy_(k_tok_h, non_blocking=True)
                self.win_v[:, :, write_slot : write_slot + 1].copy_(v_tok_h, non_blocking=True)
                self._update_window_token_in_local_buffers(write_slot, k_tok_h, v_tok_h)
                self._update_window_token_in_page_buffers(write_slot, k_tok_h, v_tok_h)
                self.win_ptr = (self.win_ptr + 1) % self.win_tokens

    def middle_bounds(self, seq_len: int) -> Optional[Tuple[int, int]]:
        lower = self.sink_tokens
        upper = seq_len - self.win_tokens
        if upper <= lower:
            return None
        if (upper - lower) < self.mid_tokens:
            return None
        return lower, upper

    def _sync_sink_to_local_buffers(self):
        if not self.enable_local_kv:
            self.local_sink_len_cached = [0, 0]
            return
        if self.sink_len <= 0:
            self.local_sink_len_cached = [0, 0]
            return
        for i in (0, 1):
            lk = self.local_k_buf[i]
            lv = self.local_v_buf[i]
            if lk is None or lv is None:
                continue
            if self.local_sink_len_cached[i] == self.sink_len:
                continue
            lk[:, :, : self.sink_len].copy_(
                self.sink_k[: self.eff_batch, :, : self.sink_len], non_blocking=True
            )
            lv[:, :, : self.sink_len].copy_(
                self.sink_v[: self.eff_batch, :, : self.sink_len], non_blocking=True
            )
            self.local_sink_len_cached[i] = self.sink_len

    def _sync_sink_to_page_buffers(self):
        if not self.enable_page_kv:
            return
        if self.sink_len < self.sink_tokens:
            return
        if self.n_sink_pages <= 0:
            return
        sink_k_pages = self.sink_k[: self.eff_batch].transpose(1, 2).reshape(
            self.eff_batch, self.n_sink_pages, self.page_size, self.n_kv_heads, self.head_dim
        )
        sink_v_pages = self.sink_v[: self.eff_batch].transpose(1, 2).reshape(
            self.eff_batch, self.n_sink_pages, self.page_size, self.n_kv_heads, self.head_dim
        )
        for i in (0, 1):
            if self.page_sink_ready[i]:
                continue
            pbuf = self.page_kv_buf[i]
            if pbuf is None:
                continue
            pbuf[:, : self.n_sink_pages, 0].copy_(sink_k_pages, non_blocking=True)
            pbuf[:, : self.n_sink_pages, 1].copy_(sink_v_pages, non_blocking=True)
            self.page_sink_ready[i] = True

    def _sync_window_to_page_buffers_full(self):
        if not self.enable_page_kv:
            return
        if self.win_len < self.win_tokens or self.n_win_pages <= 0:
            return
        off = self.n_sink_pages + self.mid_pages
        win_pages_k = self.win_k[: self.eff_batch].transpose(1, 2).reshape(
            self.eff_batch, self.n_win_pages, self.page_size, self.n_kv_heads, self.head_dim
        )
        win_pages_v = self.win_v[: self.eff_batch].transpose(1, 2).reshape(
            self.eff_batch, self.n_win_pages, self.page_size, self.n_kv_heads, self.head_dim
        )
        for i in (0, 1):
            if self.page_win_ready[i]:
                continue
            pbuf = self.page_kv_buf[i]
            if pbuf is None:
                continue
            pbuf[:, off : off + self.n_win_pages, 0].copy_(win_pages_k, non_blocking=True)
            pbuf[:, off : off + self.n_win_pages, 1].copy_(win_pages_v, non_blocking=True)
            self.page_win_ready[i] = True

    def _update_window_token_in_page_buffers(
        self, slot: int, k_tok_h: torch.Tensor, v_tok_h: torch.Tensor
    ):
        if not self.enable_page_kv:
            return
        if self.n_win_pages <= 0:
            return
        page_idx = int(slot // self.page_size)
        tok_off = int(slot % self.page_size)
        off = self.n_sink_pages + self.mid_pages + page_idx
        k_src = k_tok_h[: self.eff_batch, :, 0, :]
        v_src = v_tok_h[: self.eff_batch, :, 0, :]
        for i in (0, 1):
            if not self.page_win_ready[i]:
                continue
            pbuf = self.page_kv_buf[i]
            if pbuf is None:
                continue
            pbuf[:, off, 0, tok_off].copy_(k_src, non_blocking=True)
            pbuf[:, off, 1, tok_off].copy_(v_src, non_blocking=True)

    def _sync_window_to_local_buffers_full(self):
        if not self.enable_local_kv:
            return
        if self.win_len < self.win_tokens:
            return
        if self.local_total_cap <= 0:
            return
        off = self.sink_len + self.mid_tokens
        for i in (0, 1):
            lk = self.local_k_buf[i]
            lv = self.local_v_buf[i]
            if lk is None or lv is None:
                continue
            lk[:, :, off : off + self.win_len].copy_(
                self.win_k[: self.eff_batch, :, : self.win_len], non_blocking=True
            )
            lv[:, :, off : off + self.win_len].copy_(
                self.win_v[: self.eff_batch, :, : self.win_len], non_blocking=True
            )

    def _update_window_token_in_local_buffers(
        self, slot: int, k_tok_h: torch.Tensor, v_tok_h: torch.Tensor
    ):
        if not self.enable_local_kv:
            return
        if self.local_total_cap <= 0:
            return
        off = self.sink_len + self.mid_tokens + int(slot)
        k_src = k_tok_h[: self.eff_batch, :, 0, :]
        v_src = v_tok_h[: self.eff_batch, :, 0, :]
        for i in (0, 1):
            lk = self.local_k_buf[i]
            lv = self.local_v_buf[i]
            if lk is None or lv is None:
                continue
            lk[:, :, off : off + 1].copy_(k_src.unsqueeze(2), non_blocking=True)
            lv[:, :, off : off + 1].copy_(v_src.unsqueeze(2), non_blocking=True)

    def _ensure_page_ids(self, bsz: int):
        if (not self.enable_page_kv) or self.local_pages <= 0:
            self.page_ids = None
            self.page_ids_bsz = bsz
            return
        if self.page_ids is not None and self.page_ids_bsz == bsz:
            return
        base = torch.arange(self.local_pages, dtype=torch.int32, device=self.device)
        if self.eff_batch == 1 and bsz > 1:
            self.page_ids = base.view(1, -1).expand(bsz, -1).contiguous()
        else:
            offs = (
                torch.arange(bsz, dtype=torch.int32, device=self.device).view(-1, 1)
                * int(self.local_pages)
            )
            self.page_ids = offs + base.view(1, -1)
        self.page_ids_bsz = bsz

    def _copy_window_pages_to_page_buffer(self, dst_pages: torch.Tensor):
        if not self.enable_page_kv:
            return
        win_len = self.win_len
        if win_len < self.win_tokens or self.n_win_pages <= 0:
            return
        # For q_len=1 with non-causal attention, key/value permutation is output-invariant.
        # Keep ring order and avoid per-step reordering/copying.
        self._sync_window_to_page_buffers_full()

    def _get_head_index(self, device: torch.device):
        if self.anchor_head_sample <= 0 or self.anchor_head_sample >= self.n_kv_heads:
            return None
        if self._head_idx is not None and self._head_idx.device == device:
            return self._head_idx
        k = int(max(1, min(self.anchor_head_sample, self.n_kv_heads)))
        if k == 1:
            idx = torch.zeros((1,), dtype=torch.long, device=device)
        else:
            idx = torch.floor(
                torch.arange(k, device=device, dtype=torch.float32)
                * (self.n_kv_heads / float(k))
            ).to(torch.long)
            idx = torch.clamp(idx, max=self.n_kv_heads - 1)
        self._head_idx = idx.contiguous()
        return self._head_idx

    def _expand_seed_anchors(self, anchors: torch.Tensor) -> torch.Tensor:
        # anchors: [eff_bsz, seed_k]
        if anchors.shape[1] == self.mid_pages:
            return anchors
        if anchors.shape[1] == 0:
            return torch.zeros(
                (anchors.shape[0], self.mid_pages),
                dtype=anchors.dtype,
                device=anchors.device,
            )
        idx = torch.linspace(
            0,
            anchors.shape[1] - 1,
            steps=self.mid_pages,
            device=anchors.device,
        ).round().to(torch.long)
        idx = idx.view(1, -1).expand(anchors.shape[0], -1)
        return anchors.gather(dim=1, index=idx)

    def _maybe_sort_anchors(self, anchors: torch.Tensor) -> torch.Tensor:
        if self.allow_anchor_overlap:
            return anchors
        return torch.sort(anchors, dim=-1).values

    def init_anchors_from_prefill(self, q_last: torch.Tensor, k_full: torch.Tensor):
        # q_last: [bsz, 1, n_qo_heads, head_dim]
        # k_full: [bsz, seq_len, n_kv_heads, head_dim]
        self._set_prefetch_hint(None, -1)
        if self.mid_pages <= 0:
            self.anchors = None
            return
        bounds = self.middle_bounds(k_full.shape[1])
        if bounds is None:
            self.anchors = None
            return
        lower, upper = bounds
        eff_q = q_last[: self.eff_batch]
        eff_k = k_full[: self.eff_batch]

        grouped_q = eff_q.view(
            self.eff_batch, self.n_kv_heads, self.n_q_per_kv, self.head_dim
        ).mean(dim=2)  # [eff, n_kv_heads, head_dim]
        k_hsd = eff_k.transpose(1, 2).contiguous()  # [eff, n_kv_heads, seq_len, head_dim]
        head_idx = self._get_head_index(grouped_q.device)
        if head_idx is not None:
            grouped_q = grouped_q.index_select(1, head_idx)
            k_hsd = k_hsd.index_select(1, head_idx)
        scores = torch.matmul(
            grouped_q.unsqueeze(2), k_hsd.transpose(2, 3)
        ).squeeze(2) * self._inv_sqrt_head_dim  # [eff, sampled_heads, seq_len]
        mid_scores = scores[..., lower:upper]  # [eff, sampled_heads, mid_len]
        k = min(self.seed_anchors, mid_scores.shape[-1])
        seed = torch.topk(mid_scores, k=k, dim=-1).indices + lower
        seed = seed.to(mid_scores.dtype).mean(dim=1).round().to(torch.int32)  # [eff, k], shared over kv heads
        seed = self._maybe_sort_anchors(seed)
        anchors = self._expand_seed_anchors(seed)
        anchors = self._maybe_sort_anchors(anchors)
        self.anchors = anchors.to(device=torch.device("cpu"), dtype=torch.int32).contiguous()

    def _tile_starts(self, anchors: torch.Tensor, lower: int, upper: int) -> torch.Tensor:
        # anchors: [eff_bsz, mid_pages]
        max_start = upper - self.page_size
        if max_start < lower:
            return torch.full_like(anchors, lower)

        ideal = torch.clamp(anchors - self.page_size // 2, min=lower, max=max_start)
        if self.allow_anchor_overlap:
            return ideal.to(dtype=torch.int32).contiguous()
        ideal, _ = torch.sort(ideal, dim=-1)

        penalty = torch.arange(self.mid_pages, device=anchors.device, dtype=ideal.dtype)
        penalty = penalty * self.page_size
        y = ideal - penalty.view(1, -1)
        starts = torch.cummax(y, dim=-1).values + penalty.view(1, -1)

        upper_bounds = max_start - torch.arange(
            self.mid_pages - 1, -1, -1, device=anchors.device, dtype=ideal.dtype
        ) * self.page_size
        starts = torch.minimum(starts, upper_bounds.view(1, -1))

        y2 = starts - penalty.view(1, -1)
        starts = torch.cummax(y2, dim=-1).values + penalty.view(1, -1)
        starts = torch.clamp(starts, min=lower, max=max_start)
        return starts.to(dtype=torch.int32).contiguous()

    def _set_prefetch_hint(self, starts: Optional[torch.Tensor], target_seq_len: int):
        if starts is None:
            self.prefetch_hint_starts = None
            self.prefetch_hint_seq_len = -1
            return
        if starts.dtype != torch.int32 or (not starts.is_contiguous()):
            starts = starts.to(dtype=torch.int32).contiguous()
        if starts.device.type != "cpu":
            starts = starts.to(device=torch.device("cpu"), dtype=torch.int32).contiguous()
        self.prefetch_hint_starts = starts
        self.prefetch_hint_seq_len = int(target_seq_len)

    def _starts_from_local_best(
        self,
        local_best_cpu_i32: torch.Tensor,
        target_seq_len: int,
    ) -> Optional[torch.Tensor]:
        if self.active_starts is None:
            return None
        bounds = self.middle_bounds(int(target_seq_len))
        if bounds is None:
            return None
        lower, upper = bounds
        max_start = upper - self.page_size
        if max_start < lower:
            return None

        local_best_cpu_i32 = local_best_cpu_i32.to(
            device=torch.device("cpu"), dtype=torch.int32
        ).contiguous()
        starts = self.active_starts + local_best_cpu_i32 - int(self.slide_half_window)
        starts = torch.clamp(starts, min=lower, max=max_start)
        return starts.to(dtype=torch.int32).contiguous()

    def build_starts(self, seq_len: int) -> Optional[torch.Tensor]:
        if self.mid_pages <= 0:
            return None
        if (
            self.prefetch_hint_starts is not None
            and self.prefetch_hint_seq_len == int(seq_len)
        ):
            return self.prefetch_hint_starts
        if self.anchors is None:
            return None
        anchors = self.anchors
        if anchors.dim() == 3:
            # Backward-compatible fix: collapse accidental [eff, heads, pages]
            # anchors to the shared [eff, pages] layout used by recall.
            anchors = anchors.to(torch.float32).mean(dim=1).round().to(torch.int32)
            self.anchors = anchors.to(device=torch.device("cpu"), dtype=torch.int32).contiguous()
        if anchors.dim() != 2:
            raise RuntimeError(
                f"EchoKV anchors must be 2D [eff_bsz, mid_pages], got shape={tuple(anchors.shape)}"
            )
        bounds = self.middle_bounds(seq_len)
        if bounds is None:
            return None
        lower, upper = bounds
        return self._tile_starts(anchors, lower, upper)

    def _fallback_recall(self, starts: torch.Tensor, out_buf: torch.Tensor):
        # starts: [eff_bsz, mid_pages], out_buf: [eff_bsz, mid_tokens, 2, n_kv_heads, head_dim]
        starts_cpu = starts.to(device=torch.device("cpu"), dtype=torch.long)
        for b in range(starts_cpu.shape[0]):
            for p in range(self.mid_pages):
                s = int(starts_cpu[b, p].item())
                e = s + self.page_size
                out_buf[b, p * self.page_size : (p + 1) * self.page_size].copy_(
                    self.cpu_kv[b, s:e],
                    non_blocking=True,
                )

    def _fallback_recall_partial(
        self,
        starts: torch.Tensor,
        out_buf: torch.Tensor,
        page_begin: int,
        page_count: int,
    ):
        if page_count <= 0:
            return
        starts_cpu = starts.to(device=torch.device("cpu"), dtype=torch.long)
        p_end = int(page_begin + page_count)
        for b in range(starts_cpu.shape[0]):
            for p in range(int(page_begin), p_end):
                s = int(starts_cpu[b, p].item())
                e = s + self.page_size
                off = p * self.page_size
                out_buf[b, off : off + self.page_size].copy_(
                    self.cpu_kv[b, s:e],
                    non_blocking=True,
                )

    def _fill_qk_scores_segment(
        self,
        scores: torch.Tensor,       # [eff, n_q_heads, max_tokens], float32
        q_compact: torch.Tensor,    # [eff, n_q_heads, head_dim]
        local_k: torch.Tensor,      # [eff, n_kv_heads, seq_len, head_dim]
        tok_begin: int,
        tok_end: int,
    ):
        tok_begin = int(max(0, tok_begin))
        tok_end = int(min(int(local_k.shape[2]), tok_end))
        if tok_end <= tok_begin:
            return
        tok_count = int(tok_end - tok_begin)
        if hasattr(kernels, "echo_decode_qk_scores_chunk"):
            try:
                kernels.echo_decode_qk_scores_chunk(
                    q_compact,
                    local_k,
                    scores,
                    self.n_q_per_kv,
                    tok_begin,
                    tok_count,
                )
                return
            except Exception:
                pass

        q_group_f32 = q_compact.view(
            q_compact.shape[0], self.n_kv_heads, self.n_q_per_kv, self.head_dim
        ).to(dtype=torch.float32)
        k_seg = local_k[:, :, tok_begin:tok_end].to(dtype=torch.float32)
        seg = torch.einsum("bgqd,bgtd->bgqt", q_group_f32, k_seg)
        seg = (seg * self._inv_sqrt_head_dim).reshape(
            q_group_f32.shape[0], self.n_qo_heads, tok_count
        )
        scores[:, :, tok_begin:tok_end].copy_(seg, non_blocking=True)

    def _delta_recall_from_active(
        self, starts_cpu_i32: torch.Tensor, out_buf: torch.Tensor, in_place: bool = False
    ) -> bool:
        # Reuse previous active middle buffer and patch only changed pages.
        if self.active_starts is None:
            return False
        prev = self.active_starts
        if prev.device.type != "cpu":
            return False
        if prev.shape != starts_cpu_i32.shape:
            return False
        changed = starts_cpu_i32.ne(prev)
        changed_cnt = int(changed.sum().item())
        if changed_cnt <= 0:
            if not in_place:
                out_buf.copy_(self.gpu_mid[self.active_idx], non_blocking=True)
            return True
        if changed_cnt > int(self.delta_max_pages):
            return False

        if not in_place:
            out_buf.copy_(self.gpu_mid[self.active_idx], non_blocking=True)
        for b in range(changed.shape[0]):
            idxs = torch.nonzero(changed[b], as_tuple=False).flatten()
            for p_t in idxs:
                p = int(p_t.item())
                s = int(starts_cpu_i32[b, p].item())
                e = s + self.page_size
                out_buf[b, p * self.page_size : (p + 1) * self.page_size].copy_(
                    self.cpu_kv[b, s:e],
                    non_blocking=True,
                )
        return True

    def launch_prefetch(
        self,
        starts: torch.Tensor,
        target_seq_len: int,
        allow_inplace_delta: bool = True,
    ):
        # starts: [eff_bsz, mid_pages]
        if self.mid_pages <= 0 or starts is None:
            self.prefetch_ready = False
            return
        if starts.dtype != torch.int32 or (not starts.is_contiguous()):
            starts_i32 = starts.to(dtype=torch.int32).contiguous()
        else:
            starts_i32 = starts
        starts_cpu = starts_i32 if starts_i32.device.type == "cpu" else starts_i32.to("cpu")
        pending_idx = 1 - self.active_idx
        out_buf = self.gpu_mid[pending_idx]

        with torch.cuda.stream(self.recall_stream):
            used_delta = False
            if (
                allow_inplace_delta
                and self.active_starts is not None
                and self.use_cuda_token_recall
                and hasattr(kernels, "recall_tokens_delta_linear")
            ):
                try:
                    # Fast path: patch active middle buffer in place to avoid full d2d copies.
                    kernels.recall_tokens_delta_linear(
                        starts_i32,
                        self.active_starts,
                        self.cpu_kv[: self.eff_batch],
                        self.gpu_mid[self.active_idx],
                        self.gpu_mid[self.active_idx],
                        self.cpu_len,
                    )
                    pending_idx = self.active_idx
                    out_buf = self.gpu_mid[pending_idx]
                    used_delta = True
                except Exception:
                    used_delta = False
            if (
                (not used_delta)
                and allow_inplace_delta
                and self.active_starts is not None
                and self.use_cuda_token_recall
                and hasattr(kernels, "recall_tokens_delta_linear")
            ):
                try:
                    kernels.recall_tokens_delta_linear(
                        starts_i32,
                        self.active_starts,
                        self.cpu_kv[: self.eff_batch],
                        self.gpu_mid[self.active_idx],
                        out_buf,
                        self.cpu_len,
                    )
                    used_delta = True
                except Exception:
                    used_delta = False

            if (not used_delta) and allow_inplace_delta and self.active_starts is not None:
                # Fallback path when delta kernel is unavailable.
                used_delta = self._delta_recall_from_active(
                    starts_cpu, self.gpu_mid[self.active_idx], in_place=True
                )
                if used_delta:
                    pending_idx = self.active_idx
                    out_buf = self.gpu_mid[pending_idx]

            used_cuda = False
            if (not used_delta) and self.use_cuda_token_recall and hasattr(kernels, "recall_tokens_linear"):
                try:
                    kernels.recall_tokens_linear(
                        starts_i32,
                        self.cpu_kv[: self.eff_batch],
                        out_buf,
                        self.cpu_len,
                    )
                    used_cuda = True
                except Exception:
                    used_cuda = False
            if (not used_delta) and (not used_cuda):
                self._fallback_recall(starts_cpu, out_buf)
            self.pending_idx = pending_idx
            self.local_mid_ready[pending_idx] = False
            self.page_mid_ready[pending_idx] = False
            if self.local_k_buf[self.pending_idx] is not None:
                local_k = self.local_k_buf[self.pending_idx]
                local_v = self.local_v_buf[self.pending_idx]
                off = self.sink_len
                mid_k_hsd = out_buf[:, :, 0].permute(0, 2, 1, 3)
                mid_v_hsd = out_buf[:, :, 1].permute(0, 2, 1, 3)
                local_k[:, :, off : off + self.mid_tokens].copy_(mid_k_hsd, non_blocking=True)
                local_v[:, :, off : off + self.mid_tokens].copy_(mid_v_hsd, non_blocking=True)
                self.local_mid_ready[self.pending_idx] = True
            if self.page_kv_buf[self.pending_idx] is not None:
                self._sync_sink_to_page_buffers()
                pbuf = self.page_kv_buf[self.pending_idx]
                offp = self.n_sink_pages
                mid_pages_k = out_buf[:, :, 0].view(
                    self.eff_batch,
                    self.mid_pages,
                    self.page_size,
                    self.n_kv_heads,
                    self.head_dim,
                )
                mid_pages_v = out_buf[:, :, 1].view(
                    self.eff_batch,
                    self.mid_pages,
                    self.page_size,
                    self.n_kv_heads,
                    self.head_dim,
                )
                pbuf[:, offp : offp + self.mid_pages, 0].copy_(mid_pages_k, non_blocking=True)
                pbuf[:, offp : offp + self.mid_pages, 1].copy_(mid_pages_v, non_blocking=True)
                self.page_mid_ready[self.pending_idx] = True
            self.prefetch_event.record(self.recall_stream)

        self.pending_seq_len = int(target_seq_len)
        self.pending_starts = starts
        self.prefetch_ready = True

    def activate_prefetch(self, seq_len: int, compute_stream: torch.cuda.Stream):
        if (not self.prefetch_ready) or self.pending_seq_len != int(seq_len):
            return False
        compute_stream.wait_event(self.prefetch_event)
        self.active_idx = self.pending_idx
        self.active_seq_len = self.pending_seq_len
        self.active_starts = self.pending_starts
        return True

    def paged_kv(self, bsz: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.enable_page_kv:
            raise RuntimeError("Echo page KV buffer is disabled for this attention backend")
        self._ensure_page_ids(bsz)
        pbuf = self.page_kv_buf[self.active_idx]
        if pbuf is None:
            raise RuntimeError("Echo page buffer is not initialized")
        if not self.page_mid_ready[self.active_idx]:
            mid = self.gpu_mid[self.active_idx]
            mid_pages_k = mid[:, :, 0].view(
                self.eff_batch, self.mid_pages, self.page_size, self.n_kv_heads, self.head_dim
            )
            mid_pages_v = mid[:, :, 1].view(
                self.eff_batch, self.mid_pages, self.page_size, self.n_kv_heads, self.head_dim
            )
            offp = self.n_sink_pages
            pbuf[:, offp : offp + self.mid_pages, 0].copy_(mid_pages_k, non_blocking=True)
            pbuf[:, offp : offp + self.mid_pages, 1].copy_(mid_pages_v, non_blocking=True)
            self.page_mid_ready[self.active_idx] = True
        self._sync_sink_to_page_buffers()
        self._copy_window_pages_to_page_buffer(pbuf)

        if self.eff_batch == 1 and bsz > 1:
            kv_data = pbuf[:1].reshape(
                self.local_pages, 2, self.page_size, self.n_kv_heads, self.head_dim
            )
        else:
            kv_data = pbuf.reshape(
                self.eff_batch * self.local_pages,
                2,
                self.page_size,
                self.n_kv_heads,
                self.head_dim,
            )
        return kv_data, self.page_ids

    def current_middle_kv(self, bsz: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mid = self.gpu_mid[self.active_idx]  # [eff_bsz, mid_tokens, 2, n_kv_heads, head_dim]
        if self.eff_batch == 1 and bsz > 1:
            mid = mid[:1]
        k_mid = mid[:, :, 0].permute(0, 2, 1, 3).contiguous()  # [eff, n_kv_heads, mid_tokens, head_dim]
        v_mid = mid[:, :, 1].permute(0, 2, 1, 3).contiguous()
        return k_mid, v_mid

    def _copy_window_ordered(self, dst_k: torch.Tensor, dst_v: torch.Tensor, offset: int):
        win_len = self.win_len
        if win_len <= 0:
            return
        # Permutation of keys/values does not change attention output for single-query non-causal decode.
        dst_k[:, :, offset : offset + win_len].copy_(
            self.win_k[: self.eff_batch, :, :win_len], non_blocking=True
        )
        dst_v[:, :, offset : offset + win_len].copy_(
            self.win_v[: self.eff_batch, :, :win_len], non_blocking=True
        )

    def local_kv(self, bsz: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enable_local_kv:
            raise RuntimeError("Echo local KV buffer is disabled for this attention backend")
        sink_len = self.sink_len
        win_len = self.win_len
        total = sink_len + self.mid_tokens + win_len
        local_k = self.local_k_buf[self.active_idx][:, :, :total]
        local_v = self.local_v_buf[self.active_idx][:, :, :total]
        if not self.local_mid_ready[self.active_idx]:
            k_mid, v_mid = self.current_middle_kv(bsz)
            local_k[:, :, sink_len : sink_len + self.mid_tokens].copy_(k_mid, non_blocking=True)
            local_v[:, :, sink_len : sink_len + self.mid_tokens].copy_(v_mid, non_blocking=True)
            self.local_mid_ready[self.active_idx] = True
        # When sink region is still growing, window base offset shifts and we refresh it.
        # Otherwise window slots are maintained incrementally during append_decode_token_to_cpu.
        if win_len > 0 and (self.sink_len < self.sink_tokens or self.win_len < self.win_tokens):
            self._copy_window_ordered(local_k, local_v, sink_len + self.mid_tokens)
        k_mid = local_k[:, :, sink_len : sink_len + self.mid_tokens]
        if self.eff_batch == 1 and bsz > 1:
            return local_k[:1], local_v[:1], k_mid
        return local_k, local_v, k_mid

    def update_anchors_from_middle(self, q: torch.Tensor, k_mid: torch.Tensor):
        # q: [bsz, 1, n_qo_heads, head_dim]
        if self.mid_pages <= 0:
            return
        if self.active_starts is None:
            return
        eff_q = q[: self.eff_batch]
        eff_k_mid = k_mid[: self.eff_batch]
        starts = self.active_starts

        grouped_q = eff_q.view(
            self.eff_batch, self.n_kv_heads, self.n_q_per_kv, self.head_dim
        ).mean(dim=2)  # [eff, n_kv_heads, head_dim]
        head_idx = self._get_head_index(grouped_q.device)
        if head_idx is not None:
            grouped_q = grouped_q.index_select(1, head_idx)
            eff_k_mid = eff_k_mid.index_select(1, head_idx)
        local_best = None
        if self.use_triton_qk_select:
            tri_out = _triton_qk_page_argmax(
                grouped_q,
                eff_k_mid.transpose(1, 2).contiguous(),
                self.mid_pages,
                self.page_size,
            )
            if tri_out is not None:
                local_best = tri_out.float().mean(dim=1).round().to(torch.int32)
        if local_best is None:
            scores = torch.matmul(
                grouped_q.unsqueeze(2), eff_k_mid.transpose(2, 3)
            ).squeeze(2) * self._inv_sqrt_head_dim  # [eff, sampled_heads, mid_tokens]
            page_probs = scores.view(
                self.eff_batch, scores.shape[1], self.mid_pages, self.page_size
            )
            local_best = page_probs.argmax(dim=-1).float().mean(dim=1).round().to(torch.int32)
        local_best_cpu = local_best.to(device=torch.device("cpu"), dtype=torch.int32)
        anchors = starts + local_best_cpu
        anchors = self._maybe_sort_anchors(anchors)
        self.anchors = anchors.to(device=torch.device("cpu"), dtype=torch.int32).contiguous()
        self._set_prefetch_hint(
            self._starts_from_local_best(local_best_cpu, self.cpu_len + 1),
            self.cpu_len + 1,
        )

    def update_anchors_from_active_mid(self, q: torch.Tensor):
        # Use active gpu_mid directly to avoid extra permute/contiguous on flashinfer path.
        if self.mid_pages <= 0:
            return
        if self.active_starts is None:
            return
        eff_q = q[: self.eff_batch]
        mid = self.gpu_mid[self.active_idx][: self.eff_batch, :, 0]  # [eff, mid_tokens, n_kv_heads, head_dim]
        grouped_q = eff_q.view(
            self.eff_batch, self.n_kv_heads, self.n_q_per_kv, self.head_dim
        ).mean(dim=2)  # [eff, n_kv_heads, head_dim]
        head_idx = self._get_head_index(grouped_q.device)
        if head_idx is not None:
            grouped_q = grouped_q.index_select(1, head_idx)
            mid = mid.index_select(2, head_idx)
        local_best = None
        if self.use_triton_qk_select:
            tri_out = _triton_qk_page_argmax(
                grouped_q,
                mid,
                self.mid_pages,
                self.page_size,
            )
            if tri_out is not None:
                local_best = tri_out.float().mean(dim=1).round().to(torch.int32)
        if local_best is None:
            # [eff, sampled_heads, mid_tokens]
            scores = torch.einsum("ehd,ethd->eht", grouped_q, mid) * self._inv_sqrt_head_dim
            page_probs = scores.view(
                self.eff_batch, scores.shape[1], self.mid_pages, self.page_size
            )
            local_best = page_probs.argmax(dim=-1).float().mean(dim=1).round().to(torch.int32)
        local_best_cpu = local_best.to(device=torch.device("cpu"), dtype=torch.int32)
        anchors = self.active_starts + local_best_cpu
        anchors = self._maybe_sort_anchors(anchors)
        self.anchors = anchors.to(dtype=torch.int32).contiguous()
        self._set_prefetch_hint(
            self._starts_from_local_best(local_best_cpu, self.cpu_len + 1),
            self.cpu_len + 1,
        )

    def qk_select_and_cache_scores(
        self,
        q: torch.Tensor,         # [bsz, 1, n_qo_heads, head_dim]
        local_k: torch.Tensor,   # [eff/bsz, n_kv_heads, s, head_dim]
    ) -> Optional[torch.Tensor]:
        # Split-flash stage-1: QK + page argmax (anchor update), returns score cache.
        if self.mid_pages <= 0:
            return None
        has_cuda_qk_chunk = hasattr(kernels, "echo_decode_qk_scores_chunk")
        has_cuda_qk_pagemax = hasattr(kernels, "echo_decode_qk_scores_pagemax_chunk")
        if (not self.use_triton_flash_attn) and (not (has_cuda_qk_chunk and has_cuda_qk_pagemax)):
            return None
        if self.active_starts is None:
            return None
        bsz = q.shape[0]
        if local_k.shape[0] != bsz:
            q_use = q[:1]
        else:
            q_use = q
        q_compact = q_use[:, 0].contiguous()  # [eff, n_qo_heads, head_dim]
        scores = None
        qh_best = None
        if has_cuda_qk_chunk and has_cuda_qk_pagemax:
            try:
                scores = torch.full(
                    (q_compact.shape[0], self.n_qo_heads, self.local_total_cap),
                    -float("inf"),
                    dtype=torch.float32,
                    device=q_compact.device,
                )
                seq_len = int(local_k.shape[2])
                kernels.echo_decode_qk_scores_chunk(
                    q_compact,
                    local_k,
                    scores,
                    self.n_q_per_kv,
                    0,
                    seq_len,
                )
                qh_best = kernels.echo_decode_qk_scores_pagemax_chunk(
                    q_compact,
                    local_k,
                    scores,
                    self.n_q_per_kv,
                    self.sink_len,
                    self.page_size,
                    self.mid_pages,
                )
            except Exception:
                scores = None
                qh_best = None

        if scores is None or qh_best is None:
            scores, qh_best = _triton_flash_decode_qk_select(
                q_compact,
                local_k,
                n_q_per_kv=self.n_q_per_kv,
                page_size=self.page_size,
                mid_start=self.sink_len,
                mid_pages=self.mid_pages,
                max_tokens=self.local_total_cap,
            )
        if scores is None or qh_best is None:
            return None

        local_best = qh_best.view(
            qh_best.shape[0], self.n_kv_heads, self.n_q_per_kv, self.mid_pages
        ).float().mean(dim=(1, 2)).round().to(torch.int32)
        local_best_cpu = local_best.to(device=torch.device("cpu"), dtype=torch.int32)
        anchors = self.active_starts + local_best_cpu
        anchors = self._maybe_sort_anchors(anchors)
        self.anchors = anchors.to(dtype=torch.int32).contiguous()
        self._set_prefetch_hint(
            self._starts_from_local_best(local_best_cpu, self.cpu_len + 1),
            self.cpu_len + 1,
        )
        return scores

    def qk_select_and_cache_scores_stream_prefetch(
        self,
        q: torch.Tensor,         # [bsz, 1, n_qo_heads, head_dim]
        local_k: torch.Tensor,   # [eff/bsz, n_kv_heads, s, head_dim]
        target_seq_len: int,
    ) -> Optional[torch.Tensor]:
        # Stream-overlap path:
        # - chunked middle QK
        # - update anchors per chunk
        # - immediately launch partial recall for next step on recall stream
        if self.mid_pages <= 0:
            return None
        has_cuda_qk_chunk = hasattr(kernels, "echo_decode_qk_scores_chunk")
        has_cuda_qk_pagemax = hasattr(kernels, "echo_decode_qk_scores_pagemax_chunk")
        if (not self.use_triton_flash_attn) and (not (has_cuda_qk_chunk and has_cuda_qk_pagemax)):
            return None
        if self.active_starts is None:
            return None
        if not self.allow_anchor_overlap:
            # Non-overlap starts require global ordering across pages, which breaks chunk-level recall.
            return None
        if not self.use_cuda_token_recall:
            return None

        has_delta_partial = hasattr(kernels, "recall_tokens_delta_linear_partial")
        has_linear_partial = hasattr(kernels, "recall_tokens_linear_partial")
        if (not has_delta_partial) and (not has_linear_partial):
            return None

        bsz = q.shape[0]
        if local_k.shape[0] != bsz:
            q_use = q[:1]
        else:
            q_use = q
        q_compact = q_use[:, 0].contiguous()  # [eff, n_qo_heads, head_dim]
        eff = q_compact.shape[0]
        seq_len = int(local_k.shape[2])
        if seq_len <= 0:
            return None

        bounds = self.middle_bounds(int(target_seq_len))
        if bounds is None:
            return None
        lower, upper = bounds
        max_start = upper - self.page_size
        if max_start < lower:
            return None

        scores = torch.full(
            (eff, self.n_qo_heads, self.local_total_cap),
            -float("inf"),
            dtype=torch.float32,
            device=q_compact.device,
        )

        # Fill sink segment scores first (no recall dependency).
        self._fill_qk_scores_segment(scores, q_compact, local_k, 0, self.sink_len)

        starts_full = self.active_starts.to(
            device=torch.device("cpu"), dtype=torch.int32
        ).clone()
        local_best_full = torch.empty(
            (eff, self.mid_pages), dtype=torch.int32, device=torch.device("cpu")
        )

        pending_idx = 1 - self.active_idx
        out_buf = self.gpu_mid[pending_idx]
        starts_dev = starts_full
        prev_starts_dev = self.active_starts

        chunk_pages = int(max(1, min(self.stream_chunk_pages, self.mid_pages)))
        for p0 in range(0, self.mid_pages, chunk_pages):
            p_count = int(min(chunk_pages, self.mid_pages - p0))
            tok_begin = int(self.sink_len + p0 * self.page_size)
            tok_end = int(tok_begin + p_count * self.page_size)

            qh_best = None
            if has_cuda_qk_pagemax:
                try:
                    qh_best = kernels.echo_decode_qk_scores_pagemax_chunk(
                        q_compact,
                        local_k,
                        scores,
                        self.n_q_per_kv,
                        tok_begin,
                        self.page_size,
                        p_count,
                    )
                except Exception:
                    qh_best = None

            if qh_best is None:
                chunk_scores, qh_best = _triton_flash_decode_qk_select(
                    q_compact,
                    local_k[:, :, tok_begin:tok_end],
                    n_q_per_kv=self.n_q_per_kv,
                    page_size=self.page_size,
                    mid_start=0,
                    mid_pages=p_count,
                    max_tokens=p_count * self.page_size,
                )
                if chunk_scores is None or qh_best is None:
                    return None
                scores[:, :, tok_begin:tok_end].copy_(chunk_scores, non_blocking=True)

            local_best_chunk = qh_best.view(
                eff, self.n_kv_heads, self.n_q_per_kv, p_count
            ).float().mean(dim=(1, 2)).round().to(torch.int32)
            local_best_chunk_cpu = local_best_chunk.to(
                device=torch.device("cpu"), dtype=torch.int32
            )
            local_best_full[:, p0 : p0 + p_count].copy_(local_best_chunk_cpu)

            starts_chunk = self.active_starts[:, p0 : p0 + p_count] + local_best_chunk_cpu
            starts_chunk = torch.clamp(
                starts_chunk - int(self.slide_half_window),
                min=lower,
                max=max_start,
            ).to(dtype=torch.int32)
            starts_full[:, p0 : p0 + p_count].copy_(starts_chunk)

            with torch.cuda.stream(self.recall_stream):
                if has_delta_partial and self.active_starts is not None:
                    kernels.recall_tokens_delta_linear_partial(
                        starts_dev,
                        prev_starts_dev,
                        self.cpu_kv[: self.eff_batch],
                        self.gpu_mid[self.active_idx],
                        out_buf,
                        self.cpu_len,
                        p0,
                        p_count,
                    )
                elif has_linear_partial:
                    kernels.recall_tokens_linear_partial(
                        starts_dev,
                        self.cpu_kv[: self.eff_batch],
                        out_buf,
                        self.cpu_len,
                        p0,
                        p_count,
                    )
                else:
                    self._fallback_recall_partial(starts_dev, out_buf, p0, p_count)

        # Fill window segment scores after middle chunk loop.
        self._fill_qk_scores_segment(
            scores,
            q_compact,
            local_k,
            self.sink_len + self.mid_tokens,
            seq_len,
        )

        anchors = (self.active_starts + local_best_full).to(
            device=torch.device("cpu"), dtype=torch.int32
        ).contiguous()
        self.anchors = anchors
        self._set_prefetch_hint(starts_full, int(target_seq_len))

        with torch.cuda.stream(self.recall_stream):
            self.pending_idx = pending_idx
            self.local_mid_ready[pending_idx] = False
            self.page_mid_ready[pending_idx] = False
            if self.local_k_buf[pending_idx] is not None:
                local_k_dst = self.local_k_buf[pending_idx]
                local_v_dst = self.local_v_buf[pending_idx]
                off = self.sink_len
                mid_k_hsd = out_buf[:, :, 0].permute(0, 2, 1, 3)
                mid_v_hsd = out_buf[:, :, 1].permute(0, 2, 1, 3)
                local_k_dst[:, :, off : off + self.mid_tokens].copy_(
                    mid_k_hsd, non_blocking=True
                )
                local_v_dst[:, :, off : off + self.mid_tokens].copy_(
                    mid_v_hsd, non_blocking=True
                )
                self.local_mid_ready[pending_idx] = True
            if self.page_kv_buf[pending_idx] is not None:
                self._sync_sink_to_page_buffers()
                pbuf = self.page_kv_buf[pending_idx]
                offp = self.n_sink_pages
                mid_pages_k = out_buf[:, :, 0].view(
                    self.eff_batch,
                    self.mid_pages,
                    self.page_size,
                    self.n_kv_heads,
                    self.head_dim,
                )
                mid_pages_v = out_buf[:, :, 1].view(
                    self.eff_batch,
                    self.mid_pages,
                    self.page_size,
                    self.n_kv_heads,
                    self.head_dim,
                )
                pbuf[:, offp : offp + self.mid_pages, 0].copy_(
                    mid_pages_k, non_blocking=True
                )
                pbuf[:, offp : offp + self.mid_pages, 1].copy_(
                    mid_pages_v, non_blocking=True
                )
                self.page_mid_ready[pending_idx] = True
            self.prefetch_event.record(self.recall_stream)

        self.pending_seq_len = int(target_seq_len)
        self.pending_starts = starts_full
        self.prefetch_ready = True
        return scores

    def select_and_prefetch_stream_only(
        self,
        q: torch.Tensor,         # [bsz, 1, n_qo_heads, head_dim]
        local_k: torch.Tensor,   # [eff/bsz, n_kv_heads, s, head_dim]
        target_seq_len: int,
    ) -> bool:
        # Fast stream mode:
        # - run chunk-level page-max selection only (no full score cache)
        # - launch chunk partial recall immediately
        # - current-step output should use flash_attn attention path
        if self.mid_pages <= 0:
            return False
        if self.active_starts is None:
            return False
        if not self.allow_anchor_overlap:
            return False
        if not self.use_cuda_token_recall:
            return False

        has_delta_partial = hasattr(kernels, "recall_tokens_delta_linear_partial")
        has_linear_partial = hasattr(kernels, "recall_tokens_linear_partial")
        if (not has_delta_partial) and (not has_linear_partial):
            return False

        has_cuda_pagemax_only = hasattr(kernels, "echo_decode_qk_pagemax_chunk_only")
        if (not has_cuda_pagemax_only) and (not self.use_triton_flash_attn):
            return False

        bsz = q.shape[0]
        q_use = q[:1] if local_k.shape[0] != bsz else q
        q_compact = q_use[:, 0].contiguous()  # [eff, n_qo_heads, head_dim]
        eff = q_compact.shape[0]

        bounds = self.middle_bounds(int(target_seq_len))
        if bounds is None:
            return False
        lower, upper = bounds
        max_start = upper - self.page_size
        if max_start < lower:
            return False

        starts_full = self.active_starts.to(
            device=torch.device("cpu"), dtype=torch.int32
        ).clone()
        local_best_full = torch.empty(
            (eff, self.mid_pages), dtype=torch.int32, device=torch.device("cpu")
        )

        pending_idx = 1 - self.active_idx
        out_buf = self.gpu_mid[pending_idx]
        starts_dev = starts_full
        prev_starts_dev = self.active_starts

        chunk_pages = int(max(1, min(self.stream_chunk_pages, self.mid_pages)))
        for p0 in range(0, self.mid_pages, chunk_pages):
            p_count = int(min(chunk_pages, self.mid_pages - p0))
            tok_begin = int(self.sink_len + p0 * self.page_size)
            tok_end = int(tok_begin + p_count * self.page_size)

            qh_best = None
            if has_cuda_pagemax_only:
                try:
                    qh_best = kernels.echo_decode_qk_pagemax_chunk_only(
                        q_compact,
                        local_k,
                        self.n_q_per_kv,
                        tok_begin,
                        self.page_size,
                        p_count,
                    )
                except Exception:
                    qh_best = None

            if qh_best is None:
                chunk_scores, qh_best = _triton_flash_decode_qk_select(
                    q_compact,
                    local_k[:, :, tok_begin:tok_end],
                    n_q_per_kv=self.n_q_per_kv,
                    page_size=self.page_size,
                    mid_start=0,
                    mid_pages=p_count,
                    max_tokens=p_count * self.page_size,
                )
                if chunk_scores is None or qh_best is None:
                    return False

            local_best_chunk = qh_best.view(
                eff, self.n_kv_heads, self.n_q_per_kv, p_count
            ).float().mean(dim=(1, 2)).round().to(torch.int32)
            local_best_chunk_cpu = local_best_chunk.to(
                device=torch.device("cpu"), dtype=torch.int32
            )
            local_best_full[:, p0 : p0 + p_count].copy_(local_best_chunk_cpu)

            starts_chunk = self.active_starts[:, p0 : p0 + p_count] + local_best_chunk_cpu
            starts_chunk = torch.clamp(
                starts_chunk - int(self.slide_half_window),
                min=lower,
                max=max_start,
            ).to(dtype=torch.int32)
            starts_full[:, p0 : p0 + p_count].copy_(starts_chunk)

            with torch.cuda.stream(self.recall_stream):
                if has_delta_partial and self.active_starts is not None:
                    kernels.recall_tokens_delta_linear_partial(
                        starts_dev,
                        prev_starts_dev,
                        self.cpu_kv[: self.eff_batch],
                        self.gpu_mid[self.active_idx],
                        out_buf,
                        self.cpu_len,
                        p0,
                        p_count,
                    )
                elif has_linear_partial:
                    kernels.recall_tokens_linear_partial(
                        starts_dev,
                        self.cpu_kv[: self.eff_batch],
                        out_buf,
                        self.cpu_len,
                        p0,
                        p_count,
                    )
                else:
                    self._fallback_recall_partial(starts_dev, out_buf, p0, p_count)

        anchors = (self.active_starts + local_best_full).to(
            device=torch.device("cpu"), dtype=torch.int32
        ).contiguous()
        self.anchors = anchors
        self._set_prefetch_hint(starts_full, int(target_seq_len))

        with torch.cuda.stream(self.recall_stream):
            self.pending_idx = pending_idx
            self.local_mid_ready[pending_idx] = False
            self.page_mid_ready[pending_idx] = False
            if self.local_k_buf[pending_idx] is not None:
                local_k_dst = self.local_k_buf[pending_idx]
                local_v_dst = self.local_v_buf[pending_idx]
                off = self.sink_len
                mid_k_hsd = out_buf[:, :, 0].permute(0, 2, 1, 3)
                mid_v_hsd = out_buf[:, :, 1].permute(0, 2, 1, 3)
                local_k_dst[:, :, off : off + self.mid_tokens].copy_(
                    mid_k_hsd, non_blocking=True
                )
                local_v_dst[:, :, off : off + self.mid_tokens].copy_(
                    mid_v_hsd, non_blocking=True
                )
                self.local_mid_ready[pending_idx] = True
            if self.page_kv_buf[pending_idx] is not None:
                self._sync_sink_to_page_buffers()
                pbuf = self.page_kv_buf[pending_idx]
                offp = self.n_sink_pages
                mid_pages_k = out_buf[:, :, 0].view(
                    self.eff_batch,
                    self.mid_pages,
                    self.page_size,
                    self.n_kv_heads,
                    self.head_dim,
                )
                mid_pages_v = out_buf[:, :, 1].view(
                    self.eff_batch,
                    self.mid_pages,
                    self.page_size,
                    self.n_kv_heads,
                    self.head_dim,
                )
                pbuf[:, offp : offp + self.mid_pages, 0].copy_(
                    mid_pages_k, non_blocking=True
                )
                pbuf[:, offp : offp + self.mid_pages, 1].copy_(
                    mid_pages_v, non_blocking=True
                )
                self.page_mid_ready[pending_idx] = True
            self.prefetch_event.record(self.recall_stream)

        self.pending_seq_len = int(target_seq_len)
        self.pending_starts = starts_full
        self.prefetch_ready = True
        return True

    def pv_from_score_cache(
        self,
        scores: torch.Tensor,     # [eff, n_qo_heads, max_tokens]
        local_v: torch.Tensor,    # [eff/bsz, n_kv_heads, s, head_dim]
        out_bsz: int,
    ) -> Optional[torch.Tensor]:
        # Split-flash stage-2: P@V from cached QK scores.
        if scores is None:
            return None
        seq_len = int(local_v.shape[2])
        out_compact = None
        if hasattr(kernels, "echo_decode_pv_from_scores_cuda"):
            try:
                out_compact = kernels.echo_decode_pv_from_scores_cuda(
                    scores,
                    local_v,
                    self.n_q_per_kv,
                    seq_len,
                )
                if out_compact is not None and out_compact.dtype != local_v.dtype:
                    out_compact = out_compact.to(dtype=local_v.dtype)
            except Exception:
                out_compact = None
        if out_compact is None:
            out_compact = _triton_flash_decode_pv_from_scores(
                scores,
                local_v,
                n_q_per_kv=self.n_q_per_kv,
                seq_len=seq_len,
            )
        if out_compact is None:
            return None
        out = out_compact.unsqueeze(1).contiguous()  # [eff, 1, n_qo_heads, head_dim]
        if out.shape[0] == 1 and out_bsz > 1:
            out = out.expand(out_bsz, -1, -1, -1).contiguous()
        return out

    def attend_and_update_anchors_fused(
        self,
        q: torch.Tensor,         # [bsz, 1, n_qo_heads, head_dim]
        local_k: torch.Tensor,   # [eff/bsz, n_kv_heads, s, head_dim]
        local_v: torch.Tensor,   # [eff/bsz, n_kv_heads, s, head_dim]
    ) -> Tuple[torch.Tensor, bool]:
        if (not self.use_triton_flash_attn) or self.mid_pages <= 0:
            return self.attend(q, local_k, local_v), False
        if self.active_starts is None:
            return self.attend(q, local_k, local_v), False

        bsz = q.shape[0]
        if local_k.shape[0] != bsz:
            q_use = q[:1]
        else:
            q_use = q
        q_compact = q_use[:, 0].contiguous()  # [eff, n_qo_heads, head_dim]

        fused_out, qh_best = _triton_flash_decode_attn_with_page_max(
            q_compact,
            local_k,
            local_v,
            n_q_per_kv=self.n_q_per_kv,
            page_size=self.page_size,
            mid_start=self.sink_len,
            mid_pages=self.mid_pages,
            max_tokens=self.local_total_cap,
        )
        if fused_out is None or qh_best is None:
            return self.attend(q, local_k, local_v), False

        # q-head aggregation to shared [eff, mid_pages] anchors.
        local_best = qh_best.view(
            fused_out.shape[0], self.n_kv_heads, self.n_q_per_kv, self.mid_pages
        ).float().mean(dim=(1, 2)).round().to(torch.int32)
        local_best_cpu = local_best.to(device=torch.device("cpu"), dtype=torch.int32)
        anchors = self.active_starts + local_best_cpu
        anchors = self._maybe_sort_anchors(anchors)
        self.anchors = anchors.to(dtype=torch.int32).contiguous()
        self._set_prefetch_hint(
            self._starts_from_local_best(local_best_cpu, self.cpu_len + 1),
            self.cpu_len + 1,
        )

        out = fused_out.unsqueeze(1).contiguous()  # [eff, 1, n_qo_heads, head_dim]
        if out.shape[0] == 1 and bsz > 1:
            out = out.expand(bsz, -1, -1, -1).contiguous()
        return out, True

    def attend(self, q: torch.Tensor, local_k: torch.Tensor, local_v: torch.Tensor) -> torch.Tensor:
        # q: [bsz, 1, n_qo_heads, head_dim]
        # local_k/local_v: [bsz, n_kv_heads, s, head_dim]
        bsz = q.shape[0]
        if local_k.shape[0] != bsz:
            q_use = q[:1]
        else:
            q_use = q
        qh = q_use.transpose(1, 2).contiguous()        # [bsz, n_qo_heads, 1, head_dim]
        kh = local_k
        vh = local_v

        try:
            out = F.scaled_dot_product_attention(
                qh,
                kh,
                vh,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=True,
            )
        except TypeError:
            kh = kh.repeat_interleave(self.n_q_per_kv, dim=1)
            vh = vh.repeat_interleave(self.n_q_per_kv, dim=1)
            out = F.scaled_dot_product_attention(
                qh, kh, vh, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        out = out.transpose(1, 2).contiguous()  # [bsz, 1, n_qo_heads, head_dim]
        if out.shape[0] == 1 and bsz > 1:
            out = out.expand(bsz, -1, -1, -1).contiguous()
        return out

    def attend_flash_attn(
        self,
        q: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        strict: bool = False,
    ) -> torch.Tensor:
        # q: [bsz, 1, n_qo_heads, head_dim]
        # local_k/local_v: [bsz, n_kv_heads, s, head_dim]
        bsz = q.shape[0]
        if local_k.shape[0] != bsz:
            q_use = q[:1]
        else:
            q_use = q

        out = None
        prefer_pkg = bool(self.prefer_flash_attn_package)

        if prefer_pkg and strict and flash_attn_func is None:
            raise RuntimeError(
                "echo_attn_backend=flash_attn requires flash_attn package, "
                "but flash_attn is not installed."
            )

        def _call_flash_attn_pkg() -> Optional[torch.Tensor]:
            if flash_attn_func is None:
                return None
            q_in = q_use.contiguous()  # [eff, 1, n_q_heads, d]
            k_in = local_k.transpose(1, 2).contiguous()  # [eff, s, n_kv_heads, d]
            v_in = local_v.transpose(1, 2).contiguous()
            try:
                # Prefer native GQA path to avoid per-step key/value head expansion.
                return flash_attn_func(
                    q_in,
                    k_in,
                    v_in,
                    dropout_p=0.0,
                    softmax_scale=None,
                    causal=False,
                )
            except Exception:
                if self.n_q_per_kv <= 1:
                    raise
                k_rep = k_in.repeat_interleave(self.n_q_per_kv, dim=2)
                v_rep = v_in.repeat_interleave(self.n_q_per_kv, dim=2)
                return flash_attn_func(
                    q_in,
                    k_rep,
                    v_rep,
                    dropout_p=0.0,
                    softmax_scale=None,
                    causal=False,
                )
            except Exception:
                if strict and prefer_pkg:
                    raise
                return None

        if prefer_pkg:
            out = _call_flash_attn_pkg()

        if out is None and self.use_triton_flash_attn:
            q_compact = q_use[:, 0].contiguous()  # [eff, n_q_heads, d]
            out_compact = _triton_flash_decode_attn_plain(
                q_compact,
                local_k,
                local_v,
                n_q_per_kv=self.n_q_per_kv,
                max_tokens=self.local_total_cap,
            )
            if out_compact is not None:
                out = out_compact.unsqueeze(1).contiguous()

        if out is None and (not prefer_pkg):
            out = _call_flash_attn_pkg()

        if out is None:
            if strict:
                raise RuntimeError(
                    "echo_attn_backend=flash_attn requested, but neither flash_attn package "
                    "nor Triton plain flash-attn kernel is available."
                )
            return self.attend(q, local_k, local_v)
        if out.shape[0] == 1 and bsz > 1:
            out = out.expand(bsz, -1, -1, -1).contiguous()
        return out
