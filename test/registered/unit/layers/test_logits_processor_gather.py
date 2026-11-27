"""CPU dispatch and buffer contracts for gathered BF16 decode logits."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers import logits_processor
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGatheredLogitsGate(CustomTestCase):
    def setUp(self):
        self.logits = mock.Mock(spec=torch.Tensor)
        self.logits.is_cuda = True
        self.logits.dtype = torch.bfloat16
        self.logits.dim.return_value = 2
        self.logits.is_contiguous.return_value = True
        self.processor = SimpleNamespace(
            do_tensor_parallel_all_gather=True,
            use_attn_tp_group=False,
            do_tensor_parallel_all_gather_dp_attn=False,
            final_logit_softcapping=None,
            _can_use_tp_lm_head_all_to_all=lambda *args: False,
        )
        self.metadata = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        patcher = mock.patch.multiple(
            logits_processor,
            _is_cpu=False,
            _is_npu=False,
            is_gfx1250_supported=lambda: True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def allowed(self, flag):
        with logits_processor.envs.SGLANG_2X_BF16_GATHER_SOFTCAP.override(flag):
            return logits_processor.LogitsProcessor._can_gather_softcap_fused(
                self.processor, self.logits, None, None, self.metadata
            )

    def test_none_cap_bf16_decode_requires_opt_in(self):
        self.assertFalse(self.allowed(False))
        self.assertTrue(self.allowed(True))

    def test_existing_capped_fp32_route_is_preserved(self):
        self.logits.dtype = torch.float32
        for cap in (None, 0.0, 30.0):
            self.processor.final_logit_softcapping = cap
            for flag in (False, True):
                with self.subTest(cap=cap, flag=flag):
                    self.assertEqual(self.allowed(flag), bool(cap))

    def test_capped_bf16_uses_same_opt_in(self):
        self.processor.final_logit_softcapping = 30.0
        self.assertFalse(self.allowed(False))
        self.assertTrue(self.allowed(True))

    def test_zero_cap_keeps_the_original_route(self):
        self.processor.final_logit_softcapping = 0.0
        self.assertFalse(self.allowed(True))

    def test_other_forward_modes_keep_the_original_route(self):
        for mode in (ForwardMode.EXTEND, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                self.metadata.forward_mode = mode
                self.assertFalse(self.allowed(True))

    def test_architecture_and_tensor_guards(self):
        with mock.patch.object(
            logits_processor, "is_gfx1250_supported", return_value=False
        ):
            self.assertFalse(self.allowed(True))
        for field, value in (("is_cuda", False), ("dtype", torch.float16)):
            with (
                self.subTest(field=field),
                mock.patch.object(self.logits, field, value),
            ):
                self.assertFalse(self.allowed(True))
        with mock.patch.object(self.logits, "dim", return_value=3):
            self.assertFalse(self.allowed(True))
        with mock.patch.object(self.logits, "is_contiguous", return_value=False):
            self.assertFalse(self.allowed(True))

    def test_collective_layout_guards(self):
        for field, value in (
            ("do_tensor_parallel_all_gather", False),
            ("use_attn_tp_group", True),
            ("do_tensor_parallel_all_gather_dp_attn", True),
            ("_can_use_tp_lm_head_all_to_all", lambda *args: True),
        ):
            with (
                self.subTest(field=field),
                mock.patch.object(self.processor, field, value),
            ):
                self.assertFalse(self.allowed(True))


class TestGatheredLogitsBuffer(CustomTestCase):
    def test_none_cap_preserves_crop_and_reuses_fp32_output(self):
        shards = torch.arange(30, dtype=torch.float32).to(torch.bfloat16).view(2, 3, 5)
        output_storage = torch.full((3, 13), -123.0)
        output = output_storage[:, :9]
        metadata = SimpleNamespace(next_token_logits_buffer=output)
        processor = SimpleNamespace(vocab_size=9, final_logit_softcapping=None)

        def all_gather(destination, source):
            self.assertTrue(torch.equal(source, shards[0]))
            destination.copy_(shards.view(-1, 5))

        def cast_gathered(gathered, cap, result):
            self.assertIsNone(cap)
            result.copy_(gathered.movedim(0, 1).reshape(3, 10)[:, :9])
            return result

        group = SimpleNamespace(world_size=2, all_gather_into_tensor=all_gather)
        with (
            mock.patch.object(
                logits_processor,
                "get_parallel",
                return_value=SimpleNamespace(tp_group=group),
            ),
            mock.patch.object(
                logits_processor, "gathered_softcap_logits", side_effect=cast_gathered
            ),
        ):
            actual = logits_processor.LogitsProcessor._gather_softcap_fused(
                processor, shards[0], metadata, use_buffer=True
            )
        self.assertIs(actual, output)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, shards.movedim(0, 1).reshape(3, 10)[:, :9]))
        self.assertTrue(torch.all(output_storage[:, 9:] == -123.0))


if __name__ == "__main__":
    unittest.main()
