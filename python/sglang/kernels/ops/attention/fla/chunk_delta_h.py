# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_delta_h.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.kernels.ops.attention.fla.op import exp, exp2, safe_exp
from sglang.kernels.ops.attention.fla.utils import (
    autotune_cache_kwargs,
    is_nvidia_hopper,
)

NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8, 16]
CHUNK_SIZE = 64
GDN_CHUNK_H_BV = int(os.getenv("SGLANG_GDN_CHUNK_H_BV", "32"))
GDN_CHUNK_H_NUM_WARPS = int(os.getenv("SGLANG_GDN_CHUNK_H_NUM_WARPS", "4"))
GDN_CHUNK_H_NUM_STAGES = int(os.getenv("SGLANG_GDN_CHUNK_H_NUM_STAGES", "2"))


@triton.autotune(
    # Single hardcoded config. The kernel writes ht (final state) back into
    # initial_state in-place; with multiple configs, triton's autotune benchmark
    # phase invokes the kernel many times for timing and corrupts the cache pool,
    # producing silently wrong output on the first user request. Restoring via
    # `restore_value=["initial_state"]` works for unit tests but OOMs on
    # production-scale models (e.g. Kimi-Linear-48B at default mem_fraction)
    # because cloning the cache pool for each benchmark exceeds available memory.
    # NT_BUCKET is kept in the autotune key for forward-compatibility (allows
    # future per-bucket configs once the kernel is refactored to write final
    # state to a separate output buffer). The env knobs keep this single-config
    # property while allowing model/hardware-local validation of the selected
    # tile without corrupting the state pool through multi-config autotune.
    configs=[
        triton.Config(
            {"BV": GDN_CHUNK_H_BV},
            num_warps=GDN_CHUNK_H_NUM_WARPS,
            num_stages=GDN_CHUNK_H_NUM_STAGES,
        )
    ],
    key=["H", "K", "V", "BT", "USE_GK", "NT_BUCKET"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    initial_state,
    initial_state_indices,
    stride_init_state,
    cu_seqlens,
    chunk_offsets,
    track_state,
    track_chunk_idx,
    stride_track_state,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_UPDATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_BUCKET: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRACK_STATE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BV, BK]
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K

    # Slot stride comes from the caller (initial_state.stride(0)): the state pool
    # may be an envelope-strided view (page-major / unified memory), where the
    # per-slot pitch spans ALL layers' state, not H*V*K. int64: envelope pitches
    # overflow an int32 index product.
    index = tl.load(initial_state_indices + i_n).to(tl.int64)
    # Padded rows carry the -1 sentinel; the decode kernel guards on it
    # (fused_recurrent.py), the chunked extend path did not.
    valid_state = index >= 0
    h0 = initial_state + index * stride_init_state
    ht = initial_state + index * stride_init_state
    if USE_INITIAL_STATE:
        h0 = h0 + i_h * V * K
    if INPLACE_UPDATE:
        ht = ht + i_h * V * K

    if TRACK_STATE:
        i_track = tl.load(track_chunk_idx + i_n).to(tl.int32)
        p_track_base = track_state + (i_n * stride_track_state + i_h * V * K).to(
            tl.int64
        )
    else:
        i_track = -1
        p_track_base = track_state

    # load initial state
    if USE_INITIAL_STATE and valid_state:
        o_vh0 = i_v * BV + tl.arange(0, BV)
        o_k1 = tl.arange(0, 64)
        m_h0_1 = (o_vh0 < V)[:, None] & (o_k1 < K)[None, :]
        b_h1 += tl.load(h0 + o_vh0[:, None] * K + o_k1[None, :], mask=m_h0_1, other=0.0).to(tl.float32)
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            m_h0_2 = (o_vh0 < V)[:, None] & (o_k2 < K)[None, :]
            b_h2 += tl.load(h0 + o_vh0[:, None] * K + o_k2[None, :], mask=m_h0_2, other=0.0).to(tl.float32)
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            m_h0_3 = (o_vh0 < V)[:, None] & (o_k3 < K)[None, :]
            b_h3 += tl.load(h0 + o_vh0[:, None] * K + o_k3[None, :], mask=m_h0_3, other=0.0).to(tl.float32)
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            m_h0_4 = (o_vh0 < V)[:, None] & (o_k4 < K)[None, :]
            b_h4 += tl.load(h0 + o_vh0[:, None] * K + o_k4[None, :], mask=m_h0_4, other=0.0).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        # store h tiles: shape (V,K), row-stride=K, col-stride=1
        o_vh = i_v * BV + tl.arange(0, BV)
        o_k1 = tl.arange(0, 64)
        m_h1 = (o_vh < V)[:, None] & (o_k1 < K)[None, :]
        p_h1 = h + i_t * stride_h + o_vh[:, None] * K + o_k1[None, :]
        tl.store(p_h1, b_h1.to(v.dtype.element_ty), mask=m_h1)
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            m_h2 = (o_vh < V)[:, None] & (o_k2 < K)[None, :]
            p_h2 = h + i_t * stride_h + o_vh[:, None] * K + o_k2[None, :]
            tl.store(p_h2, b_h2.to(v.dtype.element_ty), mask=m_h2)
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            m_h3 = (o_vh < V)[:, None] & (o_k3 < K)[None, :]
            p_h3 = h + i_t * stride_h + o_vh[:, None] * K + o_k3[None, :]
            tl.store(p_h3, b_h3.to(v.dtype.element_ty), mask=m_h3)
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            m_h4 = (o_vh < V)[:, None] & (o_k4 < K)[None, :]
            p_h4 = h + i_t * stride_h + o_vh[:, None] * K + o_k4[None, :]
            tl.store(p_h4, b_h4.to(v.dtype.element_ty), mask=m_h4)

        if TRACK_STATE and i_t == i_track:
            o_vt = i_v * BV + tl.arange(0, BV)
            o_kt1 = tl.arange(0, 64)
            m_t1 = (o_vt < V)[:, None] & (o_kt1 < K)[None, :]
            tl.store(p_track_base + o_vt[:, None] * K + o_kt1[None, :], b_h1, mask=m_t1)
            if K > 64:
                o_kt2 = 64 + tl.arange(0, 64)
                m_t2 = (o_vt < V)[:, None] & (o_kt2 < K)[None, :]
                tl.store(p_track_base + o_vt[:, None] * K + o_kt2[None, :], b_h2, mask=m_t2)
            if K > 128:
                o_kt3 = 128 + tl.arange(0, 64)
                m_t3 = (o_vt < V)[:, None] & (o_kt3 < K)[None, :]
                tl.store(p_track_base + o_vt[:, None] * K + o_kt3[None, :], b_h3, mask=m_t3)
            if K > 192:
                o_kt4 = 192 + tl.arange(0, 64)
                m_t4 = (o_vt < V)[:, None] & (o_kt4 < K)[None, :]
                tl.store(p_track_base + o_vt[:, None] * K + o_kt4[None, :], b_h4, mask=m_t4)

        # load w tiles: shape (BT, 64), row-stride=stride_w, col-stride=1
        o_wt = i_t * BT + tl.arange(0, BT)
        o_wk1 = tl.arange(0, 64)
        m_w1 = (o_wt < T)[:, None] & (o_wk1 < K)[None, :]
        b_w = tl.load(w + o_wt[:, None] * stride_w + o_wk1[None, :], mask=m_w1, other=0.0)
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            o_wk2 = 64 + tl.arange(0, 64)
            m_w2 = (o_wt < T)[:, None] & (o_wk2 < K)[None, :]
            b_w = tl.load(w + o_wt[:, None] * stride_w + o_wk2[None, :], mask=m_w2, other=0.0)
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            o_wk3 = 128 + tl.arange(0, 64)
            m_w3 = (o_wt < T)[:, None] & (o_wk3 < K)[None, :]
            b_w = tl.load(w + o_wt[:, None] * stride_w + o_wk3[None, :], mask=m_w3, other=0.0)
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            o_wk4 = 192 + tl.arange(0, 64)
            m_w4 = (o_wt < T)[:, None] & (o_wk4 < K)[None, :]
            b_w = tl.load(w + o_wt[:, None] * stride_w + o_wk4[None, :], mask=m_w4, other=0.0)
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        # load v tiles: shape (BT, BV), row-stride=stride_v, col-stride=1
        o_vv = i_v * BV + tl.arange(0, BV)
        m_vv = (o_wt < T)[:, None] & (o_vv < V)[None, :]
        b_v = tl.load(v + o_wt[:, None] * stride_v + o_vv[None, :], mask=m_vv, other=0.0) - b_v

        if SAVE_NEW_VALUE:
            tl.store(v_new + o_wt[:, None] * stride_v + o_vv[None, :], b_v.to(v.dtype.element_ty), mask=m_vv)

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            o_gt = i_t * BT + tl.arange(0, BT)
            b_g = tl.load(g + bos * H + i_h + o_gt * H, mask=o_gt < T, other=0.0)
            b_v = b_v * safe_exp(b_g_last - b_g)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            if USE_EXP2:
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp(b_gk_last4)[None, :]
        b_v = b_v.to(k.dtype.element_ty)

        # load k tiles: shape (K,T), row-stride=1, col-stride=stride_k
        o_kt = i_t * BT + tl.arange(0, BT)
        o_kk1 = tl.arange(0, 64)
        m_k1 = (o_kk1 < K)[:, None] & (o_kt < T)[None, :]
        p_k = k + o_kk1[:, None] * 1 + o_kt[None, :] * stride_k
        b_k = tl.load(p_k, mask=m_k1, other=0.0)
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            o_kk2 = 64 + tl.arange(0, 64)
            m_k2 = (o_kk2 < K)[:, None] & (o_kt < T)[None, :]
            p_k = k + o_kk2[:, None] * 1 + o_kt[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k2, other=0.0)
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            o_kk3 = 128 + tl.arange(0, 64)
            m_k3 = (o_kk3 < K)[:, None] & (o_kt < T)[None, :]
            p_k = k + o_kk3[:, None] * 1 + o_kt[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k3, other=0.0)
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            o_kk4 = 192 + tl.arange(0, 64)
            m_k4 = (o_kk4 < K)[:, None] & (o_kt < T)[None, :]
            p_k = k + o_kk4[:, None] * 1 + o_kt[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k4, other=0.0)
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    # epilogue
    if INPLACE_UPDATE and valid_state:
        # store ht tiles: shape (V,K), row-stride=K, col-stride=1
        o_vht = i_v * BV + tl.arange(0, BV)
        o_k1 = tl.arange(0, 64)
        m_ht1 = (o_vht < V)[:, None] & (o_k1 < K)[None, :]
        p_ht = ht + o_vht[:, None] * K + o_k1[None, :]
        tl.store(p_ht, b_h1.to(v.dtype.element_ty), mask=m_ht1)
        if K > 64:
            o_k2 = 64 + tl.arange(0, 64)
            m_ht2 = (o_vht < V)[:, None] & (o_k2 < K)[None, :]
            p_ht = ht + o_vht[:, None] * K + o_k2[None, :]
            tl.store(p_ht, b_h2.to(v.dtype.element_ty), mask=m_ht2)
        if K > 128:
            o_k3 = 128 + tl.arange(0, 64)
            m_ht3 = (o_vht < V)[:, None] & (o_k3 < K)[None, :]
            p_ht = ht + o_vht[:, None] * K + o_k3[None, :]
            tl.store(p_ht, b_h3.to(v.dtype.element_ty), mask=m_ht3)
        if K > 192:
            o_k4 = 192 + tl.arange(0, 64)
            m_ht4 = (o_vht < V)[:, None] & (o_k4 < K)[None, :]
            p_ht = ht + o_vht[:, None] * K + o_k4[None, :]
            tl.store(p_ht, b_h4.to(v.dtype.element_ty), mask=m_ht4)


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    use_exp2: bool = False,
    inplace_update: bool = True,
    track_state: Optional[torch.Tensor] = None,
    track_chunk_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert not (use_exp2 and g is not None), (
        "use_exp2 covers only the per-channel gk path; scalar g stays natural-exp"
    )
    assert (track_state is None) == (track_chunk_idx is None), (
        "track_state and track_chunk_idx must be passed together"
    )
    if track_state is not None:
        # The caller rounds once to the pool dtype; a narrower buffer would
        # silently double-round the snapshot.
        assert track_state.dtype == torch.float32, (
            f"track_state must be fp32, got {track_state.dtype}"
        )
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = CHUNK_SIZE

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, V, K)

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        # Envelope-strided state pools (page-major / unified memory) have a
        # per-slot pitch != H*V*K; contiguous pools pass exactly H*V*K.
        stride_init_state=(initial_state.stride(0) if initial_state is not None else 0),
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        track_state=track_state,
        track_chunk_idx=track_chunk_idx,
        stride_track_state=(track_state.stride(0) if track_state is not None else 0),
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_INITIAL_STATE=initial_state is not None,
        INPLACE_UPDATE=inplace_update,
        SAVE_NEW_VALUE=v_new is not None,
        IS_VARLEN=cu_seqlens is not None,
        NT_BUCKET=(0 if NT <= 32 else (1 if NT <= 128 else 2)),
        USE_EXP2=use_exp2,
        TRACK_STATE=track_state is not None,
    )
    return h, v_new
