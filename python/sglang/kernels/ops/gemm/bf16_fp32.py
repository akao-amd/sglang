import functools
import importlib.util
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import (
    get_bool_env_var,
    is_gfx1250_supported,
    is_hip,
)

_linear_bf16_fp32_algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()
_HPC_GEMM_WEIGHT_CACHE_ATTR = "_sglang_bf16xfp32_weight_cache"
# The HPC-Ops bf16xfp32 GEMM consumes the fp32 weight decomposed into two
# bf16 halves: w_high = w.bf16 and w_low = ((w - w_high) / scale).bf16 with
# scale = 1/256, so that w ~= w_high + scale * w_low.
_HPC_GEMM_WEIGHT_SCALE = 1.0 / 256.0
# Set at model init, never lazily, so all ranks agree; see
# mark_hpc_bf16xfp32_gemm_enabled.
_hpc_gemm_enabled = False


@functools.cache
def _hpc_gemm_bf16xfp32_available() -> bool:
    """HPC-Ops (https://github.com/Tencent/hpc-ops) ships sm90a kernels."""
    if importlib.util.find_spec("hpc") is None:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


def _can_use_hpc_gemm_bf16xfp32(
    x: torch.Tensor, y: torch.Tensor, *, min_m: int = 8
) -> bool:
    if x.dim() != 2 or y.dim() != 2 or x.shape[1] != y.shape[1]:
        return False
    if x.shape[0] < min_m:
        return False
    if not (x.is_cuda and y.is_cuda):
        return False
    if x.dtype != torch.bfloat16 or y.dtype != torch.float32:
        return False
    if not (x.is_contiguous() and y.is_contiguous()):
        return False
    if y.shape[0] % 64 != 0:
        return False
    return _hpc_gemm_bf16xfp32_available()


def _get_bf16xfp32_weight_split(
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the fp32 weight for the HPC-Ops kernel and cache the result
    (plus the split-K flag workspace, which the kernel leaves zeroed) on the
    weight tensor.

    The cache key is layout-only: in-place loader writes
    (``param.data.copy_()``) are unobservable, and captured CUDA graphs
    replay the split buffers by address, so the split is computed once and
    online weight updates are rejected instead (see
    hpc_bf16xfp32_gemm_enabled).
    """
    import hpc

    if not hpc_bf16xfp32_gemm_enabled():
        raise RuntimeError(
            "Call mark_hpc_bf16xfp32_gemm_enabled() at model init before "
            "routing GEMMs to the HPC-Ops bf16xfp32 kernel."
        )

    cache_key = (
        y.data_ptr(),
        tuple(y.shape),
        tuple(y.stride()),
        y.device.index,
        y.dtype,
    )
    cache = getattr(y, _HPC_GEMM_WEIGHT_CACHE_ATTR, None)
    if cache is not None and cache[0] == cache_key:
        return cache[1], cache[2], cache[3]

    with torch.no_grad():
        w_high = y.to(torch.bfloat16)
        w_low = ((y - w_high.float()) / _HPC_GEMM_WEIGHT_SCALE).to(torch.bfloat16)
    split_flag = hpc.get_gemm_bf16xfp32_workspace(y.shape[0])
    setattr(y, _HPC_GEMM_WEIGHT_CACHE_ATTR, (cache_key, w_high, w_low, split_flag))
    return w_high, w_low, split_flag


def mark_hpc_bf16xfp32_gemm_enabled() -> None:
    """Declare at model init that GEMMs may route to the HPC-Ops bf16xfp32
    kernel (no-op when the kernel is unavailable). Must not be called lazily
    from a forward pass: the state must depend only on startup facts so it
    is identical on every rank."""
    global _hpc_gemm_enabled
    if _hpc_gemm_bf16xfp32_available():
        _hpc_gemm_enabled = True


def hpc_bf16xfp32_gemm_enabled() -> bool:
    """Whether this process may cache bf16xfp32 weight splits. The online
    weight-update APIs reject updates while True (the cache cannot survive
    in-place weight writes). Startup-determined, so all ranks agree."""
    if _hpc_gemm_enabled:
        return True
    return _linear_bf16_fp32_algo == "hpc" and _hpc_gemm_bf16xfp32_available()


def _linear_bf16_fp32_cublas(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.is_cuda and x.dtype == torch.bfloat16 and y.dtype == torch.bfloat16:
        return torch.mm(x, y.t(), out_dtype=torch.float32)
    return torch.mm(x.float(), y.float().t())


def _linear_bf16_fp32_hpc(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    min_m: int = 8,
) -> Optional[torch.Tensor]:
    if not _can_use_hpc_gemm_bf16xfp32(x, y, min_m=min_m):
        return None

    import hpc

    w_high, w_low, split_flag = _get_bf16xfp32_weight_split(y)
    return hpc.gemm_bf16xfp32(
        x,
        w_high,
        w_low,
        _HPC_GEMM_WEIGHT_SCALE,
        use_fp32_output=True,
        use_splitk=True,
        split_flag=split_flag,
    )


@functools.cache
def _aiter_gluon_a16w16():
    """AITER's gluon ``a16w16`` GEMM, or None when it must not be used.

    gfx1250 only. hipBLASLt has no tuned bf16 kernel for this arch, so
    ``torch.mm(..., out_dtype=torch.float32)`` lands on a ``MT256x128x64``
    macro-tile for GEMMs whose M is a decode batch -- ~316 us against ~9.8 us for
    this kernel, measured in a DeepSeek-V4-Flash decode trace at conc=8 (an
    isolated microbenchmark of the same two kernels reads ~220 us vs ~4 us, but
    its weight stays cache-resident, so the in-situ figure is the honest one).
    AITER says the same thing in ``tuned_gemm.py``: "gfx1250 has no tuned
    ASM/skinny/hipblaslt bf16 kernels, so the torch fallback lands on hipBLASLt,
    which is markedly slower than the Triton (gluon) a16w16 kernel".

    Called directly rather than through ``aiter.tuned_gemm.tgemm``, which on
    gfx1250 resolves an untuned bf16 shape to this same kernel. Two reasons to
    name it instead of asking for it: it keeps a CSV lookup, a cu-count query and
    the dispatcher off the path of a ~10 us kernel; and it is not subject to
    AITER's own caveat about that fallback -- "explicit tuned CSV entries still
    win since they are matched before this fallback is reached" -- so a gfx1250
    row added later naming a backend that allocates its output at the *input*
    dtype would silently round an fp32 request. The gluon kernel accumulates in
    fp32 and stores the accumulator, so ``dtype`` is a request for precision
    rather than a cast.

    The import and the arch query happen on the first call; Triton compiles a
    kernel per (N, K) and M-bucket as those are first seen.
    """
    if not (get_bool_env_var("SGLANG_USE_AITER") and is_hip()):
        return None
    if not is_gfx1250_supported():
        return None
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
    except ImportError:
        return None
    return gemm_a16w16


def linear_bf16_fp32(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    hpc_kernel_min_m: Optional[int] = None,
) -> torch.Tensor:
    # Preconditions are a strict subset of _linear_bf16_fp32_cublas's fast arm
    # (which does not check rank), so this branch never takes a case the fallback
    # would have handled differently; the rank checks matter because the AITER
    # entry point would reshape a 3-D input rather than decline it.
    # Ordered first, as the pre-#41019 AITER branch was: on gfx1250 the "hpc"
    # algo is unreachable (it needs device capability 9), so the only setting
    # this preempts is an explicit SGLANG_OPT_BF16_FP32_GEMM_ALGO=deep_gemm.
    if (
        x.is_cuda
        and x.dim() == 2
        and y.dim() == 2
        and x.dtype == torch.bfloat16
        and y.dtype == torch.bfloat16
    ):
        gemm_a16w16 = _aiter_gluon_a16w16()
        if gemm_a16w16 is not None:
            return gemm_a16w16(x, y, dtype=torch.float32)
    if hpc_kernel_min_m is not None:
        output = _linear_bf16_fp32_hpc(x, y, min_m=hpc_kernel_min_m)
        if output is not None:
            return output
        return _linear_bf16_fp32_cublas(x, y)
    elif _linear_bf16_fp32_algo == "hpc":
        output = _linear_bf16_fp32_hpc(x, y)
        if output is not None:
            return output
        return _linear_bf16_fp32_cublas(x, y)
    elif _linear_bf16_fp32_algo == "deep_gemm" and y.dtype == torch.bfloat16:
        from sglang.srt.layers import deep_gemm_wrapper

        z = torch.empty(x.size(0), y.size(0), dtype=torch.float32, device=x.device)
        deep_gemm_wrapper.gemm_nt_bf16bf16f32(x, y, z)
        return z
    else:
        return _linear_bf16_fp32_cublas(x, y)
