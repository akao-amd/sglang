"""The ROCm bf16 x bf16 -> fp32 GEMMs of DeepSeek-V4 keep fp32 accuracy."""

import unittest

import torch

from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32
from sglang.kernels.ops.gemm.router_gemv_hip import (
    ROCM_ROUTER_MAX_TOKENS,
    rocm_router_gemv_split_k,
    rocm_router_reduce_partials,
)
from sglang.srt.utils import is_gfx95_supported, is_gfx1250_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

NUM_EXPERTS, HIDDEN = 384, 5120


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "ROCm gfx95 only")
class TestRouterGemvHip(CustomTestCase):
    def test_gemv_accuracy_and_batch_invariance(self):
        """The split-K GEMV is within fp32 rounding of fp64, and every M runs the same
        16-row tile, so a row's result does not depend on the batch."""
        gen = torch.Generator(device="cuda").manual_seed(0)
        weight = (
            torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", generator=gen) * 0.02
        ).to(torch.bfloat16)
        x = torch.randn(
            ROCM_ROUTER_MAX_TOKENS, HIDDEN, device="cuda", generator=gen
        ).to(torch.bfloat16)
        ref = (x.double() @ weight.double().T).float()
        full = torch.empty(ROCM_ROUTER_MAX_TOKENS, NUM_EXPERTS, device="cuda")
        rocm_router_reduce_partials(rocm_router_gemv_split_k(x, weight), full)
        self.assertTrue(torch.allclose(full, ref, atol=2e-3, rtol=1e-4))
        part = torch.empty(17, NUM_EXPERTS, device="cuda")
        rocm_router_reduce_partials(rocm_router_gemv_split_k(x[:17], weight), part)
        self.assertTrue(torch.equal(part, full[:17]))

    def test_linear_bf16_fp32_is_not_rounded_to_bf16(self):
        """linear_bf16_fp32 feeds the V4 compressor and router logits in fp32; a route
        that rounds its output to bf16 on ROCm lands far outside fp32 accumulation error."""
        torch.manual_seed(0)
        x = torch.randn(37, HIDDEN, device="cuda", dtype=torch.bfloat16)
        w = (torch.randn(512, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
        out = linear_bf16_fp32(x, w)
        self.assertEqual(out.dtype, torch.float32)
        # bf16 x bf16 products are exact in fp64, so this is the reference up to fp32 order
        ref = x.double() @ w.double().t()
        err = (out.double() - ref).abs().max()
        bf16_err = (out.bfloat16().double() - ref).abs().max()
        self.assertLess(err, bf16_err)
        torch.testing.assert_close(out.double(), ref, rtol=1e-4, atol=1e-4)


# The class above is gfx95-only, and on gfx1250 linear_bf16_fp32 takes a different
# route (AITER's gluon a16w16 kernel), so nothing there exercises it. Upstream CI has
# no gfx1250 runner, so this skips in CI too -- it guards the contract for anyone
# running on the hardware, which is where the path was introduced and measured.
@unittest.skipUnless(is_hip() and is_gfx1250_supported(), "ROCm gfx1250 only")
class TestLinearBf16Fp32Gfx1250(CustomTestCase):
    # The DeepSeek-V4-Flash compressor's wkv_gate widths at hidden_size 4096:
    # 2*coff*head_dim for head_dim 512 (coff 1 or 2) and for the indexer's 128.
    SHAPES = ((8, 512), (8, 1024), (8, 2048), (37, 1024), (2048, 2048))

    def test_gluon_route_is_not_rounded_to_bf16(self):
        """Same contract as the gfx95 test above, on the gluon route: a kernel that
        rounded its output to bf16 would land far outside fp32 accumulation error."""
        torch.manual_seed(0)
        for m, n in self.SHAPES:
            with self.subTest(m=m, n=n):
                x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
                w = (torch.randn(n, 4096, device="cuda") * 0.02).to(torch.bfloat16)
                out = linear_bf16_fp32(x, w)
                self.assertEqual(out.dtype, torch.float32)
                ref = x.double() @ w.double().t()
                err = (out.double() - ref).abs().max()
                bf16_err = (out.bfloat16().double() - ref).abs().max()
                self.assertLess(err, bf16_err)
                torch.testing.assert_close(out.double(), ref, rtol=1e-4, atol=1e-4)

    def test_non_bf16_and_non_2d_still_take_the_fallback(self):
        """The gluon branch's preconditions match _linear_bf16_fp32_cublas's fast arm,
        so anything outside them must still reach the fallback rather than raise."""
        torch.manual_seed(0)
        x3 = torch.randn(2, 4, 4096, device="cuda", dtype=torch.bfloat16)
        w = (torch.randn(512, 4096, device="cuda") * 0.02).to(torch.bfloat16)
        self.assertEqual(linear_bf16_fp32(x3.reshape(-1, 4096), w).dtype, torch.float32)

        x_cpu = torch.randn(4, 4096, dtype=torch.bfloat16)
        w_cpu = (torch.randn(512, 4096) * 0.02).to(torch.bfloat16)
        self.assertEqual(linear_bf16_fp32(x_cpu, w_cpu).dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
