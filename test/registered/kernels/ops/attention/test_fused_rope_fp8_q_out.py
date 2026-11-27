"""gfx1250: the fused RoPE + KV-write kernel writing Q directly in fp8 (SGLANG_2X_DECODE_Q_FP8_ROPE).

The fp8 Q must be the fp32 rotated Q (the same kernel with an fp32 q_out) clamped to the e4m3 range and cast,
and K/V cache writes must be unchanged."""

import pytest
import torch

from sglang.kernels.ops.attention.fused_qk_rope_reshape_and_cache import (
    fused_qk_rope_reshape_and_cache,
)
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="base-a-jit-kernel-unit", runner_config="amd")


@pytest.fixture
def gfx1250():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("Requires gfx1250")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
        pytest.skip("Requires gfx1250")


def _run(q, k, v, kc, vc, slots, pos, cos_sin, q_out, scale):
    return fused_qk_rope_reshape_and_cache(
        q=q,
        k=k,
        v=v,
        key_cache=kc,
        value_cache=vc,
        slot_mapping=slots,
        pos=pos,
        cos_sin=cos_sin,
        k_scale=scale,
        v_scale=scale,
        is_neox=True,
        flash_layout=True,
        offs=None,
        q_out=q_out,
        k_out=k.clone(),
        output_zeros=False,
    )


@pytest.mark.parametrize(
    "tokens,qh,kh,cos_dtype",
    [
        (1, 32, 4, torch.float32),
        (44, 32, 4, torch.bfloat16),
        (168, 32, 4, torch.float32),
        (84, 64, 8, torch.float32),
        (192, 32, 4, torch.bfloat16),
        # >= 256 tokens with an fp32 cos/sin table takes the tiled kernel
        (512, 32, 4, torch.float32),
    ],
)
@pytest.mark.parametrize("magnitude", [1.0, 600.0])
def test_fp8_q_out_matches_fp32_clamped(gfx1250, tokens, qh, kh, cos_dtype, magnitude):
    torch.manual_seed(tokens * 31 + qh)
    d, page, pages = 64, 64, 64
    dev = "cuda"
    # q as a strided view into a fused qkv row, as gpt-oss hands it over
    qkv = torch.randn(tokens, (qh + 2 * kh) * d, device=dev, dtype=torch.bfloat16) * magnitude
    q = qkv[:, : qh * d].view(tokens, qh, d)
    k = qkv[:, qh * d : (qh + kh) * d].view(tokens, kh, d)
    v = qkv[:, (qh + kh) * d :].reshape(tokens, kh, d)
    pos = torch.randint(0, 4096, (tokens,), device=dev)
    inv = 1.0 / (150000 ** (torch.arange(0, d, 2, device=dev, dtype=torch.float32) / d))
    t = torch.arange(4096, device=dev, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(cos_dtype)
    slots = torch.randperm(pages * page, device=dev)[:tokens].long()
    scale = torch.ones(1, device=dev, dtype=torch.float32)

    def caches():
        return (
            torch.zeros(pages, page, kh, d, device=dev, dtype=fp8_dtype),
            torch.zeros(pages, page, kh, d, device=dev, dtype=fp8_dtype),
        )

    qkv_before = qkv.clone()
    kc0, vc0 = caches()
    q32 = torch.empty(tokens, qh, d, device=dev, dtype=torch.float32)
    _run(q, k, v, kc0, vc0, slots, pos, cos_sin, q32, scale)
    kc1, vc1 = caches()
    q8 = torch.empty(tokens, qh, d, device=dev, dtype=fp8_dtype)
    _run(q, k, v, kc1, vc1, slots, pos, cos_sin, q8, scale)
    torch.cuda.synchronize()

    fmax = torch.finfo(fp8_dtype).max
    want = q32.clamp(-fmax, fmax).to(fp8_dtype)
    assert torch.isfinite(q8.float()).all()
    assert torch.equal(q8.view(torch.uint8), want.view(torch.uint8))
    assert torch.equal(kc0.view(torch.uint8), kc1.view(torch.uint8))
    assert torch.equal(vc0.view(torch.uint8), vc1.view(torch.uint8))
    # neither call writes the input qkv when q_out and k_out are separate tensors
    assert torch.equal(qkv.view(torch.int16), qkv_before.view(torch.int16))
