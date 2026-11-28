"""CPU ordering tests for starting the stage-based GPU profiler."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import sglang.srt.observability.profile_utils as profile_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "available,activities,expected",
    [
        (True, ["CPU", "GPU"], ["sync", "start", "stop"]),
        (True, ["CPU"], ["start", "stop"]),
        (False, ["CPU", "GPU"], ["start", "stop"]),
        (True, None, ["start", "stop"]),
    ],
)
def test_device_drains_before_gpu_trace(monkeypatch, available, activities, expected):
    calls = []
    profiler = SimpleNamespace(
        start=lambda: calls.append("start"), stop=lambda: calls.append("stop")
    )
    factory = Mock(return_value=profiler)
    annotations = Mock()
    monkeypatch.setattr(profile_utils._ProfilerBase, "create", factory)
    monkeypatch.setattr(profile_utils, "set_detailed_annotations_enabled", annotations)
    monkeypatch.setattr(profile_utils.torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(
        profile_utils.torch.cuda, "synchronize", lambda: calls.append("sync")
    )
    manager = object.__new__(profile_utils.ProfileManager)
    manager.profiler = None
    manager.profiler_kwargs = dict(
        activities=activities, output_dir="unused", profile_id="unit-test"
    )
    manager.cpu_group = None
    manager.first_rank_in_node = True
    manager.detailed_annotations = True
    manager._do_start(stage="prefill")
    assert manager.profiler is profiler
    assert factory.call_args.kwargs["output_suffix"] == "-prefill"
    manager._do_stop()
    assert manager.profiler is None
    assert calls == expected
    assert [call.args[0] for call in annotations.call_args_list] == [True, False]
