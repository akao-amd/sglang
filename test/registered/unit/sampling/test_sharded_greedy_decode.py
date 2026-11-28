"""Deferred decode must choose the same collective before any logits consumer."""

import datetime
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.environ import envs
from sglang.srt.layers import logits_processor
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.unlimited_ocr_v2_support.greedy_graph import UocrGreedyGraph
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_pipeline import SamplingPipeline
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import published_topology

register_cpu_ci(est_time=70, stage="base-a", runner_config="cpu")


def _processor():
    processor = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.defer_decode_gather = True
    processor.beam_trie_output_head = None
    processor.vocab_size = 8
    return processor


def _case(full=None):
    sampler = Sampler.__new__(Sampler)
    torch.nn.Module.__init__(sampler)
    if full is None:
        full = torch.arange(8, dtype=torch.float32).view(1, 8)
    events = []

    class Processor:
        def sharded_greedy_consensus(self, eligible):
            events.append(("consensus", eligible))
            return eligible

        def finish_deferred_gather(self, local):
            events.append(("gather", local.shape[1]))
            return torch.cat((local, full[:, 4:]), dim=1)

        def deferred_argmax(self, local, *, torch_compatible):
            assert torch_compatible
            events.append(("argmax", local.shape[1]))
            return full.argmax(-1)

    output = LogitsProcessorOutput(
        next_token_logits=full[:, :4].clone(),
        next_token_logits_deferred_processor=Processor(),
    )
    batch = SimpleNamespace(
        sampling_info=SamplingBatchInfo(vocab_size=8, device="cpu", is_all_greedy=True),
        return_logprob=False,
        top_logprobs_nums=[0],
        token_ids_logprobs=[None],
        positions=torch.tensor([0]),
        forward_mode=ForwardMode.DECODE,
    )
    return SamplingPipeline(sampler=sampler, watermark=None), output, batch, events


class TestDeferredConsumers(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name in (
            "SGLANG_SANITIZE_NAN_LOGITS",
            "SGLANG_ENABLE_NAN_LOGITS_CHECK",
            "SGLANG_ENABLE_ASYNC_ASSERT",
        ):
            stack.enter_context(getattr(envs, name).override(False))
        stack.enter_context(
            patch("sglang.srt.layers.sampler.SYNC_TOKEN_IDS_ACROSS_TP", False)
        )
        stack.enter_context(patch("sglang.srt.layers.sampler._use_aiter", False))
        stack.enter_context(patch("sglang.srt.layers.sampler._is_hip", True))

    def test_active_torch_greedy_matches_baseline_without_gather(self):
        for use_aiter in (False, True):
            for values, expected in (
                ([float("nan"), 1, -2, -3, -4, -5, -6, -7], 0),
                ([-1, 2, -2, -3, 2, -5, -6, -7], 1),
            ):
                with (
                    self.subTest(use_aiter=use_aiter, expected=expected),
                    patch("sglang.srt.layers.sampler._use_aiter", use_aiter),
                    patch(
                        "sglang.srt.layers.sampler._disable_aiter_greedy_sample",
                        use_aiter,
                    ),
                    patch(
                        "sglang.srt.layers.sampler._aiter_greedy_sample", create=True
                    ) as aiter_sample,
                ):
                    full = torch.tensor([values], dtype=torch.float32)
                    pipeline, baseline, batch, _ = _case(full)
                    baseline.next_token_logits = full.clone()
                    baseline.next_token_logits_deferred_processor = None
                    baseline_ids = pipeline.sample(
                        baseline, batch, observer=None, graph_token_ids=None
                    )
                    pipeline, output, batch, events = _case(full)
                    self.assertTrue(pipeline.sampler.supports_torch_greedy_argmax())
                    ids = pipeline.sample(
                        output, batch, observer=None, graph_token_ids=None
                    )
                    self.assertEqual(ids.tolist(), [expected])
                    self.assertEqual(ids.tolist(), baseline_ids.tolist())
                    self.assertEqual(events, [("consensus", True), ("argmax", 4)])
                    self.assertIsNone(output.next_token_logits)
                    aiter_sample.assert_not_called()

    def test_active_aiter_greedy_vetoes_and_preserves_baseline(self):
        for values in (
            [float("nan"), 1, -2, -3, -4, -5, -6, -7],
            [-1, 2, -2, -3, 2, -5, -6, -7],
        ):
            with (
                self.subTest(values=values),
                patch("sglang.srt.layers.sampler._use_aiter", True),
                patch("sglang.srt.layers.sampler._disable_aiter_greedy_sample", False),
                patch(
                    "sglang.srt.layers.sampler._aiter_greedy_sample",
                    side_effect=_cpu_aiter_greedy,
                    create=True,
                ) as aiter_sample,
            ):
                full = torch.tensor([values], dtype=torch.float32)
                pipeline, baseline, batch, _ = _case(full)
                baseline.next_token_logits = full.clone()
                baseline.next_token_logits_deferred_processor = None
                baseline_ids = pipeline.sample(
                    baseline, batch, observer=None, graph_token_ids=None
                )
                pipeline, output, batch, events = _case(full)
                self.assertFalse(pipeline.sampler.supports_torch_greedy_argmax())
                ids = pipeline.sample(
                    output, batch, observer=None, graph_token_ids=None
                )
                self.assertEqual(ids.tolist(), [1])
                self.assertEqual(ids.tolist(), baseline_ids.tolist())
                self.assertEqual(events, [("consensus", False), ("gather", 4)])
                self.assertEqual(output.next_token_logits.shape, (1, 8))
                self.assertIsNone(output.next_token_logits_deferred_processor)
                self.assertEqual(aiter_sample.call_count, 2)

    def test_sampling_observer_sees_full_logits_before_grammar(self):
        pipeline, output, batch, events = _case()
        observer = Mock(is_active=Mock(return_value=True))
        observer.before_grammar.side_effect = lambda logits, info: (
            events.append(("observer", logits.shape[1])) or "state"
        )
        ids = pipeline.sample(output, batch, observer=observer, graph_token_ids=None)
        self.assertEqual(ids.tolist(), [7])
        self.assertEqual(
            events[:3], [("consensus", False), ("gather", 4), ("observer", 8)]
        )
        observer.after_sample.assert_called_once_with("state", ids)
        self.assertIsNone(output.next_token_logits_deferred_processor)

    def test_inactive_observer_does_not_require_materialization(self):
        pipeline, output, batch, events = _case()
        observer = Mock(is_active=Mock(return_value=False))
        ids = pipeline.sample(output, batch, observer=observer, graph_token_ids=None)
        self.assertEqual(ids.tolist(), [7])
        self.assertEqual(events, [("consensus", True), ("argmax", 4)])
        observer.before_grammar.assert_not_called()

    def test_full_logits_consumers_veto_candidate_reduction(self):
        # The normal stages below are mocked only after materialization: each
        # consumer verifies it receives the peer vocabulary before using it.
        for consumer in (
            "logprobs",
            "grammar",
            "penalty",
            "custom",
            "sampling_mask",
            "stochastic",
            "graph_ids",
        ):
            with self.subTest(consumer=consumer):
                pipeline, output, batch, events = _case()
                graph_ids = None
                if consumer == "logprobs":
                    batch.return_logprob = True
                elif consumer == "grammar":
                    batch.sampling_info.grammars = [object()]
                elif consumer == "penalty":
                    batch.sampling_info.step_penalties = Mock()
                elif consumer == "custom":
                    batch.sampling_info.has_custom_logit_processor = True
                elif consumer == "sampling_mask":
                    batch.sampling_info.sampling_mask_batch_indices = torch.tensor([0])
                elif consumer == "stochastic":
                    batch.sampling_info.is_all_greedy = False
                else:
                    graph_ids = torch.tensor([6])

                def check_full(logits, *args, **kwargs):
                    self.assertEqual(logits.shape, (1, 8))

                def sample_full(logits_output, *args, **kwargs):
                    check_full(logits_output.next_token_logits)
                    return torch.tensor([7])

                with (
                    patch(
                        "sglang.srt.sampling.sampling_pipeline._adjust_before_watermark",
                        side_effect=check_full,
                    ),
                    patch(
                        "sglang.srt.sampling.sampling_pipeline._adjust_after_watermark",
                        side_effect=check_full,
                    ),
                    patch.object(pipeline.sampler, "forward", side_effect=sample_full),
                ):
                    ids = pipeline.sample(
                        output, batch, observer=None, graph_token_ids=graph_ids
                    )
                self.assertEqual(ids.tolist(), [6 if graph_ids is not None else 7])
                self.assertEqual(events, [("consensus", False), ("gather", 4)])
                self.assertIsNone(output.next_token_logits_deferred_processor)

    def test_watermark_receives_full_logits_and_commits(self):
        pipeline, output, batch, events = _case()
        step = SimpleNamespace(
            sampling_info=batch.sampling_info,
            logprob_logits=None,
            token_ids=torch.tensor([5]),
        )
        watermark = Mock()
        watermark.apply.side_effect = lambda logits, **kw: (
            self.assertEqual(logits.shape, (1, 8)) or step
        )
        engine = Mock(bind_sample=Mock(return_value=watermark))
        pipeline = SamplingPipeline(sampler=pipeline.sampler, watermark=engine)
        ids = pipeline.sample(output, batch, observer=None, graph_token_ids=None)
        self.assertEqual(events, [("consensus", False), ("gather", 4)])
        watermark.commit.assert_called_once_with(ids)

    def test_diagnostics_keep_full_logits(self):
        for name in (
            "SGLANG_SANITIZE_NAN_LOGITS",
            "SGLANG_ENABLE_NAN_LOGITS_CHECK",
            "SGLANG_ENABLE_ASYNC_ASSERT",
        ):
            with self.subTest(name=name), getattr(envs, name).override(True):
                pipeline, output, batch, events = _case()
                with (
                    patch("sglang.srt.sampling.sampling_pipeline.maybe_detect_nan"),
                    patch(
                        "sglang.srt.sampling.sampling_pipeline._adjust_after_watermark"
                    ),
                ):
                    pipeline.sample(output, batch, observer=None, graph_token_ids=None)
                self.assertEqual(events, [("consensus", False), ("gather", 4)])
                self.assertEqual(output.next_token_logits.shape, (1, 8))

    def test_custom_sampler_gets_full_logits(self):
        pipeline, output, batch, events = _case()
        sampler = Mock(return_value=torch.tensor([7]))
        pipeline = SamplingPipeline(sampler=sampler, watermark=None)
        pipeline.sample(output, batch, observer=None, graph_token_ids=None)
        self.assertEqual(events, [("consensus", False), ("gather", 4)])
        self.assertEqual(sampler.call_args.args[0].next_token_logits.shape, (1, 8))

    def test_logprobs_only_materializes_once_before_preprocessing(self):
        pipeline, output, batch, events = _case()
        batch.token_ids_logprobs = [[6]]

        def compute(logits_output, *args, preprocess_fn):
            self.assertEqual(logits_output.next_token_logits.shape, (1, 8))
            preprocess_fn(logits_output.next_token_logits)

        pipeline.sampler.compute_logprobs_only = Mock(side_effect=compute)
        pipeline.compute_logprobs_only(output, batch)
        pipeline.compute_logprobs_only(output, batch)
        self.assertEqual(events, [("gather", 4)])
        self.assertIsNone(output.next_token_logits_deferred_processor)

    def test_in_graph_argmax_materializes_before_sampling_and_relay(self):
        _, output, batch, events = _case()
        graph = UocrGreedyGraph(max_bs=1, device="cpu")
        relayed = []
        graph.bind_relay_tail(
            lambda runner, out, forward_batch, num_tokens: relayed.append(
                out.next_token_logits.shape
            )
        )
        graph.capture_tail_hook(None, output, batch, 1)
        self.assertEqual(graph.ids.tolist(), [7])
        self.assertEqual(events, [("gather", 4)])
        self.assertEqual(relayed, [(1, 8)])
        self.assertIsNone(output.next_token_logits_deferred_processor)


class TestDecodeShardGate(unittest.TestCase):
    def setUp(self):
        self.processor = _processor()
        self.head = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        torch.nn.Module.__init__(self.head)
        self.head.shard_indices = SimpleNamespace(num_added_elements_padded=0)
        self.head.num_embeddings_padded = 8
        self.logits = SimpleNamespace(
            is_cuda=True, dtype=torch.bfloat16, ndim=2, stride=lambda axis: 1
        )
        self.metadata = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(logits_processor, "_is_cpu", False))
        stack.enter_context(patch.object(logits_processor, "_is_npu", False))

    def allows(self):
        return self.processor._can_defer_decode_gather(
            self.logits, self.head, self.metadata
        )

    def test_supported_metadata_without_model_or_tp_size(self):
        self.assertTrue(self.allows())
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            self.logits.dtype = dtype
            self.assertTrue(self.allows())

    def test_other_forward_modes_and_beam_keep_full_logits(self):
        for mode in (ForwardMode.EXTEND, ForwardMode.TARGET_VERIFY, ForwardMode.IDLE):
            self.metadata.forward_mode = mode
            self.assertFalse(self.allows())
        self.metadata.forward_mode = ForwardMode.DECODE
        self.processor.beam_trie_output_head = object()
        self.assertFalse(self.allows())

    def test_added_vocab_and_unrepresentable_indices_keep_full_logits(self):
        self.head.shard_indices.num_added_elements_padded = 1
        self.assertFalse(self.allows())
        self.head.shard_indices.num_added_elements_padded = 0
        self.head.num_embeddings_padded = 2**24
        self.assertFalse(self.allows())
        self.head.num_embeddings_padded = 8
        self.processor.vocab_size = 0
        self.assertFalse(self.allows())

    def test_unsupported_tensor_layout_dtype_and_cpu_keep_full_logits(self):
        for attr, value in (
            ("is_cuda", False),
            ("dtype", torch.int32),
            ("ndim", 3),
            ("stride", lambda axis: 2),
        ):
            original = getattr(self.logits, attr)
            setattr(self.logits, attr, value)
            self.assertFalse(self.allows())
            setattr(self.logits, attr, original)
        self.processor.defer_decode_gather = False
        self.assertFalse(self.allows())


class TestDecodeShardInitialization(unittest.TestCase):
    def enabled(self, *, opt_in, hip=True, **topology):
        group = SimpleNamespace(cpu_group=None)
        parallel = SimpleNamespace(
            enable_dp_lm_head=False,
            enable_tp_lm_head_all_to_all=False,
            tp_size=4,
            attn_tp_size=4,
            attn_dp_size=1,
            tp_group=group,
            attn_tp_group=group,
        )
        parallel.__dict__.update(topology)
        with (
            published_topology(tp_size=4),
            envs.SGLANG_ENABLE_SHARDED_GREEDY_DECODE.override(opt_in),
            patch.object(logits_processor, "get_parallel", return_value=parallel),
            patch.object(logits_processor, "is_hip", return_value=hip),
            patch.object(logits_processor.triton_symm_mem_ag, "MultimemAllGatherer"),
            patch.object(
                logits_processor.triton_symm_mem_ag,
                "recommended_max_tokens",
                return_value=128,
            ),
            patch.object(LogitsProcessor, "input_logprob_processor_cls"),
        ):
            processor = LogitsProcessor(
                SimpleNamespace(vocab_size=8, model_type="other_decoder")
            )
        return processor.defer_decode_gather

    def test_opt_in_works_for_tp4_and_another_model(self):
        self.assertTrue(self.enabled(opt_in=True))
        self.assertFalse(self.enabled(opt_in=False))

    def test_incompatible_collective_or_argmax_contract_keeps_baseline(self):
        for topology in (
            dict(tp_size=1),
            dict(enable_dp_lm_head=True),
            dict(attn_dp_size=2),
            dict(enable_tp_lm_head_all_to_all=True),
        ):
            with self.subTest(topology=topology):
                self.assertFalse(self.enabled(opt_in=True, **topology))
        self.assertFalse(self.enabled(opt_in=True, hip=False))


def _cpu_local_argmax(logits, cap, valid_width, col_offset, *, torch_compatible):
    # CPU oracle substitutes only the Triton launch in this collective test.
    # Native tests separately verify the actual local reduction and merge.
    assert torch_compatible
    values = logits[:, :valid_width].float()
    if cap:
        values = cap * torch.tanh(values / cap)
    index = values.argmax(-1)
    best = values.gather(1, index[:, None]).squeeze(1)
    return torch.stack((best, (index + col_offset).float()), dim=-1)


def _cpu_aiter_greedy(ids, logits):
    # CPU oracle for aiter::ArgMax's scan: NaNs never replace the initial -inf.
    # Native worker tests must compare against the actual selected GPU sampler.
    for row, values in enumerate(logits.tolist()):
        winner, maximum = 0, float("-inf")
        for index, value in enumerate(values):
            if value > maximum or (value == maximum and index < winner):
                winner, maximum = index, value
        ids[row] = winner


def _tp_worker(rank, world, rendezvous, veto_kind="observer"):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=30),
    )
    try:

        class Group:
            cpu_group = dist.group.WORLD
            world_size = world
            rank_in_group = rank

            def all_gather_into_tensor(self, output, local):
                dist.all_gather_into_tensor(output, local)

        processor = _processor()
        processor.final_logit_softcapping = None
        processor._logits_gatherer = lambda local: torch.cat(
            _all_gather(local, world), dim=-1
        )
        with (
            patch.object(
                logits_processor,
                "get_parallel",
                return_value=SimpleNamespace(tp_group=Group()),
            ),
            patch.object(
                logits_processor, "softcap_row_argmax", side_effect=_cpu_local_argmax
            ),
            patch.object(logits_processor, "is_hip", return_value=False),
            patch("sglang.srt.layers.sampler.SYNC_TOKEN_IDS_ACROSS_TP", False),
            patch("sglang.srt.layers.sampler._is_hip", True),
            patch("sglang.srt.layers.sampler._disable_aiter_greedy_sample", False),
            patch(
                "sglang.srt.layers.sampler._aiter_greedy_sample",
                side_effect=_cpu_aiter_greedy,
                create=True,
            ),
            envs.SGLANG_SANITIZE_NAN_LOGITS.override(False),
            envs.SGLANG_ENABLE_NAN_LOGITS_CHECK.override(False),
            envs.SGLANG_ENABLE_ASYNC_ASSERT.override(False),
        ):
            for veto_rank in (None, 0, world - 1, None):
                pipeline, output, batch, _ = _case()
                width = 8 // world
                full = torch.arange(8, dtype=torch.float32).view(1, 8)
                output.next_token_logits = full[
                    :, rank * width : (rank + 1) * width
                ].contiguous()
                output.next_token_logits_deferred_processor = processor
                observer = None
                if veto_kind == "observer" and rank == veto_rank:
                    observer = Mock(
                        is_active=Mock(return_value=True),
                        before_grammar=Mock(return_value=None),
                    )
                    observer.before_grammar.side_effect = lambda logits, info: (
                        torch.testing.assert_close(logits, full) or None
                    )
                use_aiter = veto_kind == "sampler" and rank == veto_rank
                with patch("sglang.srt.layers.sampler._use_aiter", use_aiter):
                    assert pipeline.sampler.supports_torch_greedy_argmax() == (
                        not use_aiter
                    )
                    ids = pipeline.sample(
                        output, batch, observer=observer, graph_token_ids=None
                    )
                assert ids.tolist() == [7]
                assert output.next_token_logits_deferred_processor is None
                assert (output.next_token_logits is None) == (veto_rank is None)
                if veto_rank is not None:
                    torch.testing.assert_close(output.next_token_logits, full)
            # Reusing the agreement for a later step is unsafe: local consumers
            # changed above, and each step used the real CPU MIN collective.
    finally:
        dist.destroy_process_group()


def _all_gather(local, world):
    peers = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(peers, local)
    return peers


class TestDistributedConsumers(unittest.TestCase):
    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "CPU Gloo required"
    )
    def test_rank_local_observer_switches_every_rank_between_collectives(self):
        for world in (2, 4):
            with self.subTest(world=world), tempfile.TemporaryDirectory() as directory:
                mp.spawn(
                    _tp_worker,
                    args=(world, str(Path(directory) / "rendezvous")),
                    nprocs=world,
                    join=True,
                )

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "CPU Gloo required"
    )
    def test_rank_local_sampler_capability_switches_every_rank_between_collectives(
        self,
    ):
        for world in (2, 4):
            with self.subTest(world=world), tempfile.TemporaryDirectory() as directory:
                mp.spawn(
                    _tp_worker,
                    args=(world, str(Path(directory) / "rendezvous"), "sampler"),
                    nprocs=world,
                    join=True,
                )


if __name__ == "__main__":
    unittest.main()
