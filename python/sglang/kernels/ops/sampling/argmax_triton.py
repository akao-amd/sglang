"""Exact gfx1250 greedy sampling using the existing split-vocabulary reduction."""

import torch

from sglang.kernels.arch import current_arch


def can_argmax(logits: torch.Tensor) -> bool:
    """Coverage: gfx1250 FP32, 1..256 rows, 4096..262144 columns, unit column stride."""
    if (
        not logits.is_cuda
        or logits.ndim != 2
        or logits.dtype != torch.float32
        or logits.stride(1) != 1
        or not 1 <= logits.shape[0] <= 256
        or not 4096 <= logits.shape[1] <= 262144
    ):
        return False
    hip_arch = current_arch().hip
    return hip_arch is not None and hip_arch.gcn_arch == "gfx1250"


def argmax(logits: torch.Tensor) -> torch.Tensor:
    """Row-wise int64 indices, preferring NaNs and the lowest index on every tie.

    Accepts the coverage stated by `can_argmax`, including padded row strides.
    Each call owns its scratch; CUDA-graph capture retains those allocations.
    """
    if not can_argmax(logits):
        raise ValueError("Triton sampling argmax requires gfx1250 FP32 wide rows")
    # Reuse the exact comparator and two-stage reduction used by div_argmax.
    from sglang.kernels.ops.speculative.row_argmax import _medium_row_argmax

    return _medium_row_argmax(logits)
