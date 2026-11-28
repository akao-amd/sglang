"""Native shard reductions must match argmax after full FP32 materialization."""

import sys

import pytest
import torch

from sglang.kernels.ops.activation.softcap import softcap_to_float32_logits
from sglang.kernels.ops.sampling.softcap_row_argmax import (
    merge_row_argmax_shards,
    softcap_row_argmax,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_amd_ci(est_time=30, stage="base-b", runner_config="1-gpu-small-amd")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _check(shards, cap, vocab):
    world, rows, width = shards.shape
    candidates = torch.stack(
        [
            softcap_row_argmax(
                shards[rank],
                cap,
                max(0, min(width, vocab - rank * width)),
                rank * width,
                torch_compatible=True,
            )
            for rank in range(world)
        ]
    )
    actual = merge_row_argmax_shards(candidates, torch_compatible=True)
    full = shards.movedim(0, 1).reshape(rows, world * width)[:, :vocab]
    expected_logits = torch.empty(
        (rows, vocab), dtype=torch.float32, device=full.device
    )
    if cap:
        softcap_to_float32_logits(full, cap, expected_logits)
    else:
        expected_logits.copy_(full)
    torch.testing.assert_close(actual, expected_logits.argmax(-1), rtol=0, atol=0)


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("cap", [None, 30.0])
def test_random_shards_with_partial_chunk_and_vocab_padding(world, dtype, cap):
    shards = torch.randn(world, 8, 4099, dtype=dtype, device="cuda")
    _check(shards, cap, world * shards.shape[-1] - 7)


@pytest.mark.parametrize("rows", [8, 44, 168])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("cap", [None, 30.0])
def test_decode_workload_widths(rows, dtype, cap):
    shards = torch.randn(2, rows, 100544, dtype=dtype, device="cuda")
    _check(shards, cap, 201088)


@pytest.mark.parametrize("cap", [None, 30.0])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_nan_priority_ties_signed_zero_and_saturated_softcap(cap, dtype):
    x = torch.full((2, 8, 5000), -1.0, dtype=dtype, device="cuda")
    x[0, 0, 7], x[1, 0, 0] = 1024.0, 2048.0
    x[0, 1, 17] = x[1, 1, 0] = float("nan")
    x[1, 2, 4992] = float("nan")
    x[:, 3].fill_(float("nan"))
    x[:, 4].fill_(float("-inf"))
    x[0, 5, 13], x[1, 5, 3] = -0.0, 0.0
    x[:, 6].fill_(-2.5)
    x[0, 7, 5] = 2.0
    x[1, 7, 4993:] = float("nan")
    _check(x, cap, 9993)


@pytest.mark.parametrize("world", [2, 3, 4])
def test_empty_padding_ranks_lose_to_real_negative_infinity(world):
    x = torch.full((world, 2, 4096), float("-inf"), dtype=torch.float32, device="cuda")
    # Only the first rank owns real vocabulary. All other shards are padding.
    _check(x, None, 4096)


def test_repeated_replay_keeps_partial_rows_and_padding_masked():
    shards = torch.randn(2, 8, 8193, dtype=torch.bfloat16, device="cuda")
    for _ in range(20):
        _check(shards[:, ::2], None, 16379)


def test_empty_rows_do_not_launch_a_zero_grid():
    x = torch.empty((0, 4096), dtype=torch.float32, device="cuda")
    local = softcap_row_argmax(x, None, 4096, 0, torch_compatible=True)
    actual = merge_row_argmax_shards(torch.stack((local, local)), torch_compatible=True)
    assert local.shape == (0, 2)
    assert actual.shape == (0,) and actual.dtype == torch.int64


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
