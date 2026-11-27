"""gfx1250 sampling parity for the reused split-vocabulary argmax."""

import sys

import pytest
import torch

from sglang.kernels.ops.activation.softcap import softcap_to_float32_logits
from sglang.kernels.ops.sampling import argmax as sampling_argmax
from sglang.kernels.ops.sampling.argmax_triton import argmax, can_argmax
from sglang.kernels.ops.sampling.softcap_row_argmax import (
    merge_row_argmax_shards,
    softcap_row_argmax,
)
from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="base-a-jit-kernel-unit", runner_config="amd")


@pytest.fixture
def gfx1250():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("Requires gfx1250")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
        pytest.skip("Requires gfx1250")


@pytest.mark.parametrize(
    "rows,vocab,padding",
    [
        (1, 4096, 0),
        (8, 201088, 0),
        (44, 201088, 64),
        (168, 201088, 0),
        (3, 4097, 17),
        (4, 262144, 0),
    ],
)
def test_argmax_matches_torch_on_decode_shapes(gfx1250, rows, vocab, padding):
    full = torch.randn(rows, vocab + padding, dtype=torch.float32, device="cuda")
    logits = full[:, :vocab]
    assert can_argmax(logits)
    expected = logits.argmax(-1)
    actual = sampling_argmax(logits, backend=KernelBackend.TRITON)
    assert actual.dtype == torch.int64
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_argmax_nan_inf_and_ties_across_chunks(gfx1250):
    """Padding, signed zeros, NaNs and all-masked rows must preserve the first index."""
    x = torch.full((8, 8193), -1.0, dtype=torch.float32, device="cuda")
    x[0, 7] = x[0, 5000] = 1.0
    x[1, 3], x[1, 8000] = -0.0, 0.0
    x[2, 19] = x[2, 5001] = float("nan")
    x[2, 7] = float("inf")
    x[3].fill_(float("nan"))
    x[4].fill_(float("-inf"))
    x[5, 100] = x[5, 6000] = float("inf")
    x[6].fill_(-2.5)
    x[7, 8192] = float("nan")
    actual = argmax(x)
    torch.testing.assert_close(actual, x.argmax(-1), rtol=0, atol=0)
    torch.testing.assert_close(
        actual,
        torch.tensor([7, 3, 19, 0, 0, 100, 0, 8192], device="cuda", dtype=torch.int64),
        rtol=0,
        atol=0,
    )


def test_argmax_graph_replay_rewrites_all_scratch(gfx1250):
    """Changing winners across replays must not retain a previous chunk maximum."""
    x = torch.randn(8, 201088, dtype=torch.float32, device="cuda")
    argmax(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = argmax(x)
    for winner in (0, 4096, 201087):
        x.fill_(-1.0)
        x[:, winner] = 1.0
        graph.replay()
        torch.testing.assert_close(output, x.argmax(-1), rtol=0, atol=0)


def test_argmax_unsupported_cpu_inputs_keep_torch_fallback():
    x = torch.tensor([[0.0, float("nan"), 2.0], [-0.0, 0.0, -1.0]])
    assert not can_argmax(x)
    torch.testing.assert_close(
        sampling_argmax(x, backend=KernelBackend.TRITON), x.argmax(-1), rtol=0, atol=0
    )


def _check_sharded_argmax(shards, cap, crop=0):
    world, rows, width = shards.shape
    vocab = world * width - crop
    local = [
        softcap_row_argmax(
            shards[rank],
            cap,
            max(0, min(width, vocab - rank * width)),
            rank * width,
            torch_compatible=True,
        )
        for rank in range(world)
    ]
    gathered = torch.stack(local)
    actual = merge_row_argmax_shards(gathered, torch_compatible=True)
    reordered = shards.movedim(0, 1).reshape(rows, world * width)[:, :vocab]
    expected_logits = torch.empty(
        (rows, vocab), dtype=torch.float32, device=shards.device
    )
    if cap:
        softcap_to_float32_logits(reordered, cap, expected_logits)
    else:
        expected_logits.copy_(reordered)
    torch.testing.assert_close(actual, expected_logits.argmax(-1), rtol=0, atol=0)


@pytest.mark.parametrize(
    "rows,dtype",
    [
        (8, torch.bfloat16),
        (44, torch.bfloat16),
        (168, torch.bfloat16),
        (8, torch.float32),
    ],
)
def test_sharded_softcap_argmax_matches_full_logits(gfx1250, rows, dtype):
    shards = torch.randn(2, rows, 100544, dtype=dtype, device="cuda")
    _check_sharded_argmax(shards, 30.0, crop=7)


@pytest.mark.parametrize("cap", [None, 30.0])
def test_sharded_argmax_nan_ties_and_vocab_crop(gfx1250, cap):
    """Select after softcap; raw maxima can differ while their saturated outputs tie."""
    x = torch.full((2, 8, 5000), -1.0, dtype=torch.bfloat16, device="cuda")
    x[0, 0, 7], x[1, 0, 0] = 1024.0, 2048.0
    x[0, 1, 17] = x[1, 1, 0] = float("nan")
    x[1, 2, 4992] = float("nan")
    x[:, 3].fill_(float("nan"))
    x[:, 4].fill_(float("-inf"))
    x[0, 5, 13], x[1, 5, 3] = -0.0, 0.0
    x[:, 6].fill_(-2.5)
    x[0, 7, 5] = 2.0
    x[1, 7, 4993:] = float("nan")
    _check_sharded_argmax(x, cap, crop=7)


def test_sharded_argmax_empty_padding_rank(gfx1250):
    x = torch.full((2, 2, 4096), float("-inf"), dtype=torch.float32, device="cuda")
    _check_sharded_argmax(x, None, crop=4096)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
