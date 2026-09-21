# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/wy_fast.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [2, 4, 8]
#         for num_stages in [2, 3, 4]
#     ],
#     key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
# )
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    _o_t = i_t * BT + tl.arange(0, BT)
    _m_t = _o_t < T

    _p_beta = beta + bos * H + i_h + _o_t * H
    _p_g = g + bos * H + i_h + _o_t * H
    b_beta = tl.load(_p_beta, mask=_m_t, other=0.0)
    b_g = tl.exp(tl.load(_p_g, mask=_m_t, other=0.0))

    _o_Ac = tl.arange(0, BT)
    _m_A = _m_t[:, None] & (_o_Ac < BT)[None, :]
    _p_A = A + (bos * H + i_h) * BT + _o_t[:, None] * (H * BT) + _o_Ac[None, :]
    b_A = tl.load(_p_A, mask=_m_A, other=0.0)

    for i_v in range(tl.cdiv(V, BV)):
        _o_v = i_v * BV + tl.arange(0, BV)
        _m_v = _m_t[:, None] & (_o_v < V)[None, :]
        _p_v = v + (bos * H + i_h) * V + _o_t[:, None] * (H * V) + _o_v[None, :]
        _p_u = u + (bos * H + i_h) * V + _o_t[:, None] * (H * V) + _o_v[None, :]
        b_v = tl.load(_p_v, mask=_m_v, other=0.0)
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(_p_u, b_u.to(u.dtype.element_ty), mask=_m_v)

    for i_k in range(tl.cdiv(K, BK)):
        _o_k = i_k * BK + tl.arange(0, BK)
        _m_k = _m_t[:, None] & (_o_k < K)[None, :]
        _p_k = k + (bos * Hg + i_h // (H // Hg)) * K + _o_t[:, None] * (Hg * K) + _o_k[None, :]
        _p_w = w + (bos * H + i_h) * K + _o_t[:, None] * (H * K) + _o_k[None, :]
        b_k = tl.load(_p_k, mask=_m_k, other=0.0)
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(_p_w, b_w.to(w.dtype.element_ty), mask=_m_k)


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_indices: torch.LongTensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = 64
    BV = 64
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
        num_stages=3,
    )
    return w, u


fwd_recompute_w_u = recompute_w_u_fwd
