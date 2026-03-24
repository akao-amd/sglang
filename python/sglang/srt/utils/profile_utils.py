import gzip
import json
import logging
import os
import time
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from sglang.srt.managers.io_struct import ProfileReqOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_npu

_is_npu = is_npu()
if _is_npu:
    import torch_npu

    patches = [
        ["profiler.profile", torch_npu.profiler.profile],
        ["profiler.ProfilerActivity.CUDA", torch_npu.profiler.ProfilerActivity.NPU],
        ["profiler.ProfilerActivity.CPU", torch_npu.profiler.ProfilerActivity.CPU],
    ]
    torch_npu._apply_patches(patches)

logger = logging.getLogger(__name__)


def _annotate_trace_with_cuda_graph_attribution(trace_path: str, output_dir: str):
    """Annotate Chrome trace with CUDA graph kernel attribution.

    Looks for cuda_graph_kernel_attribution.json in the current directory and
    annotates kernel events in the trace with their originating CPU operations.
    """
    # Look for attribution file in common locations
    attribution_paths = [
        "cuda_graph_kernel_attribution.json",  # CWD
        os.path.join(output_dir, "cuda_graph_kernel_attribution.json"),  # Output dir
        os.path.join(os.getcwd(), "cuda_graph_kernel_attribution.json"),  # Explicit CWD
    ]

    attribution_data = None
    attribution_file = None
    for path in attribution_paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    attribution_data = json.load(f)
                attribution_file = path
                break
            except Exception as e:
                logger.debug(f"Failed to load attribution from {path}: {e}")

    if attribution_data is None:
        logger.debug(
            "No CUDA graph attribution file found. "
            "Skipping trace annotation. "
            "Use --enable-cuda-graph-attribution to generate attribution data."
        )
        return

    kernel_map = attribution_data.get("kernel_to_cpu_op", {})
    if not kernel_map:
        logger.debug("Attribution file found but contains no kernel mappings")
        return

    # Load trace (handle gzip)
    is_gzipped = trace_path.endswith('.gz')
    if is_gzipped:
        with gzip.open(trace_path, 'rt') as f:
            trace = json.load(f)
    else:
        with open(trace_path) as f:
            trace = json.load(f)

    # Annotate kernel events
    annotated_count = 0
    for event in trace.get("traceEvents", []):
        # Look for GPU kernel events
        if event.get("cat") == "kernel" or (
            event.get("ph") == "X" and "name" in event
        ):
            kernel_name = event["name"]
            if kernel_name in kernel_map:
                attributions = kernel_map[kernel_name]
                # Get unique CPU ops for this kernel
                cpu_ops = list(set(attr["cpu_op"] for attr in attributions))
                modules = list(
                    set(
                        attr.get("module", "")
                        for attr in attributions
                        if attr.get("module")
                    )
                )

                # Annotate event name with CPU op
                if cpu_ops:
                    op_suffix = ", ".join(cpu_ops[:2])  # Limit to first 2 ops
                    if len(cpu_ops) > 2:
                        op_suffix += f", +{len(cpu_ops)-2} more"
                    event["name"] = f"{kernel_name} [{op_suffix}]"

                # Add detailed attribution as metadata
                event["args"] = event.get("args", {})
                event["args"]["cuda_graph_attribution"] = {
                    "cpu_ops": cpu_ops,
                    "modules": modules,
                    "source": os.path.basename(attribution_file),
                }
                annotated_count += 1

    # Add metadata to trace
    if "traceEvents" in trace:
        trace["traceEvents"].append({
            "name": "cuda_graph_attribution_metadata",
            "ph": "M",
            "pid": 0,
            "tid": 0,
            "args": {
                "attribution_file": os.path.basename(attribution_file),
                "kernels_annotated": annotated_count,
                "total_unique_kernels": len(kernel_map),
            },
        })

    # Save annotated trace
    if is_gzipped:
        with gzip.open(trace_path, 'wt') as f:
            json.dump(trace, f)
    else:
        with open(trace_path, 'w') as f:
            json.dump(trace, f)

    logger.info(
        f"Annotated {annotated_count} kernel events in trace with CUDA graph attribution "
        f"(from {os.path.basename(attribution_file)})"
    )


class ProfileManager:
    def __init__(self, tp_rank: int, cpu_group, gpu_id: int):
        self.stage_based_trigger = _StageBasedTrigger(
            on_start=self._do_start,
            on_stop=self._do_stop,
        )
        self.tp_rank = tp_rank
        self.cpu_group = cpu_group
        self.first_rank_in_node = gpu_id == get_global_server_args().base_gpu_id
        self.profiler_kwargs = None
        self.profiler = None

    def step(self, forward_mode: ForwardMode):
        stage = _get_stage_from_forward_mode(forward_mode)
        if stage is None:
            return

        self.stage_based_trigger.step(stage=stage)

    def configure(
        self,
        *,
        output_dir: Optional[str],
        start_step: Optional[int],
        num_steps: Optional[int],
        activities: Optional[List[str]],
        with_stack: Optional[bool],
        record_shapes: Optional[bool],
        profile_by_stage: bool,
        profile_id: str,
        merge_profiles: bool,
        profile_prefix: str,
        profile_stages: Optional[List[str]] = None,
    ):
        # not supported yet
        assert start_step is None
        assert (
            profile_by_stage
        ), "only support profile_by_stage=true now"  # `false` can be easily supported
        assert not merge_profiles

        if output_dir is None:
            output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        if activities is None:
            activities = ["CPU", "GPU"]

        self.profiler_kwargs = dict(
            activities=activities,
            with_stack=with_stack,
            record_shapes=record_shapes,
            output_dir=output_dir,
            output_prefix=profile_prefix,
            profile_id=profile_id,
        )

        self.stage_based_trigger.configure(
            num_steps=num_steps,
            interesting_stages=profile_stages or ["prefill", "decode"],
        )

        return ProfileReqOutput(success=True, message="Succeeded")

    def manual_start(self):
        raise NotImplementedError("manually start is only supported yet")

    def manual_stop(self):
        raise NotImplementedError("manually stop is only supported yet")

    def _do_start(self, stage: Optional[str] = None):
        logger.info(
            f"Profiling starts{f' for {stage}' if stage else ''}. "
            f"Traces will be saved to: {self.profiler_kwargs['output_dir']} "
            f"(with profile id: {self.profiler_kwargs['profile_id']})",
        )

        assert self.profiler is None
        self.profiler = _ProfilerBase.create(
            **self.profiler_kwargs,
            tp_rank=self.tp_rank,
            cpu_group=self.cpu_group,
            first_rank_in_node=self.first_rank_in_node,
            output_suffix=f"-{stage}" if stage else "",
        )
        self.profiler.start()

    def _do_stop(self):
        logger.info("Stop profiling...")
        self.profiler.stop()
        logger.info(
            f"Profiling done. Traces are saved to: {self.profiler_kwargs['output_dir']}"
        )
        self.profiler = None


def _get_stage_from_forward_mode(forward_mode: ForwardMode):
    if forward_mode.is_prefill():
        return "prefill"
    elif forward_mode.is_decode():
        return "decode"
    elif forward_mode.is_idle():
        return None
    else:
        raise RuntimeError(f"unsupported profile stage: {forward_mode=}")


# ======================================== Stage related ==========================================


class _StageBasedTrigger:
    @dataclass
    class _StageConfig:
        target_count: int

    @dataclass
    class _RunningState:
        curr_stage: str
        curr_count: int

    def __init__(self, on_start: Callable, on_stop: Callable):
        self.on_start = on_start
        self.on_stop = on_stop

        self.running_state: Optional[_StageBasedTrigger._RunningState] = None
        # When a stage is in the dict, it means it is being or should be executed
        self.stage_configs: Dict[str, _StageBasedTrigger._StageConfig] = {}

    def configure(self, num_steps: int, interesting_stages: List[str]):
        assert self.running_state is None
        self.stage_configs = {
            stage: self._StageConfig(target_count=num_steps)
            for stage in interesting_stages
        }

    def step(self, stage: str):
        # Incr counter
        if (s := self.running_state) is not None:
            s.curr_count += 1

        # Maybe stop
        if ((s := self.running_state) is not None) and (
            (s.curr_count > self.stage_configs[s.curr_stage].target_count)
            or (stage != s.curr_stage)
        ):
            del self.stage_configs[s.curr_stage]
            self.running_state = None
            self.on_stop()

        # Maybe start
        if (self.running_state is None) and (stage in self.stage_configs):
            self.running_state = self._RunningState(
                curr_stage=stage,
                curr_count=0,
            )
            self.on_start(stage=stage)

        # Sanity check
        assert (self.running_state is not None) == (stage in self.stage_configs)
        if (s := self.running_state) is not None:
            assert s.curr_stage == stage


# ======================================== Concrete profilers ==========================================


class _ProfilerBase(ABC):
    @staticmethod
    def create(activities, with_stack, record_shapes, **kwargs):
        inners = []
        if ("CPU" in activities) or ("GPU" in activities):
            inners.append(
                _ProfilerTorch(
                    **kwargs,
                    activities=activities,
                    with_stack=with_stack,
                    record_shapes=record_shapes,
                )
            )
        if "MEM" in activities:
            inners.append(_ProfilerMemory(**kwargs))
        if "CUDA_PROFILER" in activities:
            inners.append(_ProfilerCudart(**kwargs))
        if "RPD" in activities:  # for ROCM
            inners.append(_ProfilerRPD(**kwargs))

        return _ProfilerList(inners)

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class _ProfilerList(_ProfilerBase):
    def __init__(self, inners: List[_ProfilerBase]):
        self.inners = inners

    def start(self):
        for inner in self.inners:
            inner.start()

    def stop(self):
        for inner in self.inners:
            inner.stop()


class _ProfilerConcreteBase(_ProfilerBase):
    def __init__(
        self,
        output_dir: str,
        output_prefix: str,
        output_suffix: str,
        profile_id: str,
        tp_rank: int,
        cpu_group,
        first_rank_in_node: bool,
    ):
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.output_suffix = output_suffix
        self.profile_id = profile_id
        self.tp_rank = tp_rank
        self.cpu_group = cpu_group
        self.first_rank_in_node = first_rank_in_node


class _ProfilerTorch(_ProfilerConcreteBase):
    def __init__(self, with_stack: bool, record_shapes: bool, activities, **kwargs):
        super().__init__(**kwargs)
        self.with_stack = with_stack
        self.record_shapes = record_shapes
        self.activities = activities

    def start(self):
        activity_map = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "GPU": torch.profiler.ProfilerActivity.CUDA,
        }
        torchprof_activities = [
            activity_map[a] for a in self.activities if a in activity_map
        ]

        self.torch_profiler = torch.profiler.profile(
            activities=torchprof_activities,
            with_stack=self.with_stack if self.with_stack is not None else True,
            record_shapes=(
                self.record_shapes if self.record_shapes is not None else False
            ),
            on_trace_ready=(
                None
                if not _is_npu
                else torch_npu.profiler.tensorboard_trace_handler(self.output_dir)
            ),
        )
        self.torch_profiler.start()

    def stop(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.torch_profiler.stop()
        if not _is_npu:
            # Build filename with only non-zero ranks to maintain backward compatibility
            filename_parts = [self.profile_id, f"TP-{self.tp_rank}"]

            # Only add other ranks if parallelism is enabled (size > 1)
            if getattr(self, "dp_size", 1) > 1:
                filename_parts.append(f"DP-{getattr(self, 'dp_rank', 0)}")
            if getattr(self, "pp_size", 1) > 1:
                filename_parts.append(f"PP-{getattr(self, 'pp_rank', 0)}")
            if getattr(self, "moe_ep_size", 1) > 1:
                filename_parts.append(f"EP-{getattr(self, 'moe_ep_rank', 0)}")

            filename = (
                (self.output_prefix + "-" if self.output_prefix else "")
                + "-".join(filename_parts)
                + self.output_suffix
                + ".trace.json.gz"
            )

            trace_path = os.path.join(self.output_dir, filename)
            self.torch_profiler.export_chrome_trace(trace_path)

            # Post-process: annotate with CUDA graph attribution if available
            try:
                _annotate_trace_with_cuda_graph_attribution(trace_path, self.output_dir)
            except Exception as e:
                logger.warning(f"Failed to annotate trace with CUDA graph attribution: {e}")

        torch.distributed.barrier(self.cpu_group)

        # TODO: migrate `_merge_profile_traces`


class _ProfilerMemory(_ProfilerConcreteBase):
    def start(self):
        torch.cuda.memory._record_memory_history(max_entries=100000)

    def stop(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        memory_profile_path = os.path.join(
            self.output_dir,
            str(time.time())
            + f"-TP-{self.tp_rank}-memory"
            + self.output_suffix
            + ".pickle",
        )
        torch.cuda.memory._dump_snapshot(memory_profile_path)
        torch.cuda.memory._record_memory_history(enabled=None)


class _ProfilerCudart(_ProfilerConcreteBase):
    def start(self):
        if self.first_rank_in_node:
            logger.info(f"Call cudaProfilerStart")
            torch.cuda.cudart().cudaProfilerStart()

    def stop(self):
        if self.first_rank_in_node:
            logger.info(f"Call cudaProfilerStop")
            torch.cuda.cudart().cudaProfilerStop()


class _ProfilerRPD(_ProfilerConcreteBase):
    def start(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        from rpdTracerControl import rpdTracerControl

        rpdTracerControl.skipCreate()

        self.rpd_profile_path = os.path.join(
            self.output_dir,
            "rpd-" + str(time.time()) + f"-TP-{self.tp_rank}" + ".trace.json.gz",
        )

        if self.tp_rank == 0:
            import sqlite3

            from rocpd.schema import RocpdSchema

            if os.path.exists("trace.rpd"):
                os.unlink("trace.rpd")
            schema = RocpdSchema()
            connection = sqlite3.connect("trace.rpd")
            schema.writeSchema(connection)
            connection.commit()
            del connection
        torch.distributed.barrier(self.cpu_group)

        self.rpd_profiler = rpdTracerControl()
        self.rpd_profiler.setPythonTrace(True)
        self.rpd_profiler.start()
        self.rpd_profiler.rangePush("", "rpd profile range", "")

    def stop(self):
        self.rpd_profiler.rangePop()
        self.rpd_profiler.stop()
        self.rpd_profiler.flush()

        torch.distributed.barrier(self.cpu_group)
        if self.tp_rank == 0:
            from sglang.srt.utils.rpd_utils import rpd_to_chrome_trace

            rpd_to_chrome_trace("trace.rpd", self.rpd_profile_path)
