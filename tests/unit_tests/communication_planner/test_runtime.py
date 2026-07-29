# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import math

import megatron.core.communication_planner.runtime as runtime_module
from megatron.core.communication_planner import (
    GTPCudaEventRecorder,
    GTPPhase,
    GTPRuntimeProfileConfig,
    GTPRuntimeProfiler,
    GTPWorkKind,
)


class _FakeStream:
    def __init__(self, time_ms):
        self.time_ms = time_ms


class _FakeEvent:
    def __init__(self):
        self.time_ms = None
        self.complete = True

    def record(self, stream=None):
        self.time_ms = 0.0 if stream is None else stream.time_ms

    def query(self):
        return self.complete

    def elapsed_time(self, other):
        return other.time_ms - self.time_ms


def _recorder():
    return GTPCudaEventRecorder(_FakeEvent)


class _FakeHandle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class _FakeParameter:
    is_distributed_weight = True

    def __init__(self, name, *, expert_idx=None):
        self._debug_name = name
        self.is_routed_expert = expert_idx is not None
        self.expert_idx = expert_idx


class _FakeModule:
    def __init__(self, parameters):
        self._parameters = parameters
        self.handles = []

    def named_parameters(self):
        return ((parameter._debug_name, parameter) for parameter in self._parameters)

    def named_modules(self):
        return (("", self),)

    def parameters(self, recurse=True):
        del recurse
        return iter(self._parameters)

    def register_forward_hook(self, hook):
        handle = _FakeHandle()
        self.handles.append((hook, handle))
        return handle


def test_runtime_profiler_builds_cuda_model_and_closes_forward_explicitly(tmp_path):
    profiler = GTPRuntimeProfiler(
        GTPRuntimeProfileConfig(warmup_iters=0, profile_iters=1, log_dir=tmp_path),
        _recorder(),
    )
    module = _FakeModule([_FakeParameter("output_layer")])
    profiler.attach_model(module)
    profiler.begin_iteration(7, _FakeStream(0.0))

    forward_ag = profiler.ag_ready("output_layer", GTPPhase.FORWARD)
    profiler.communication_start(forward_ag, _FakeStream(0.2))
    profiler.communication_end(forward_ag, _FakeStream(1.2))
    profiler.compute_start("output_layer", GTPPhase.FORWARD, _FakeStream(1.3))
    original_current_stream = runtime_module._current_cuda_stream
    runtime_module._current_cuda_stream = lambda: _FakeStream(3.3)
    try:
        module.handles[0][0](module, (), None)
    finally:
        runtime_module._current_cuda_stream = original_current_stream

    backward_ag = profiler.ag_ready("output_layer", GTPPhase.BACKWARD)
    profiler.communication_start(backward_ag, _FakeStream(4.1))
    profiler.communication_end(backward_ag, _FakeStream(5.1))
    profiler.compute_start("output_layer", GTPPhase.BACKWARD, _FakeStream(5.2))
    rs = profiler.rs_ready("output_layer", _FakeStream(7.2))
    profiler.communication_start(rs, _FakeStream(7.3))
    profiler.communication_end(rs, _FakeStream(8.3))

    profiler.end_iteration(_FakeStream(20.0))
    model = profiler.build_model()

    forward_compute = next(
        key
        for key in model.statistics
        if key.scope == "output_layer"
        and key.phase is GTPPhase.FORWARD
        and key.kind is GTPWorkKind.COMPUTE
    )
    assert math.isclose(model.statistics[forward_compute].p50_us, 2000.0)
    assert not profiler.errors

    artifact = tmp_path / "rank00000_gtp_execution_model.json"
    payload = json.loads(artifact.read_text())
    assert payload["diagnostics"]["timing_source"] == "cuda_events"
    assert payload["diagnostics"]["trace_file"] == "rank00000_gtp_execution_trace.json"
    assert {
        dependency["kind"] for dependency in payload["dependencies"]
    } == {"ag_before_compute", "compute_before_rs"}

    trace = json.loads((tmp_path / payload["diagnostics"]["trace_file"]).read_text())
    spans = [event for event in trace["traceEvents"] if event["ph"] == "X"]
    assert {event["args"]["kind"] for event in spans} == {"compute", "ag", "rs"}
    assert sum(event["ph"] == "s" for event in trace["traceEvents"]) == 3
    assert not any("finalize" in event["name"] or "bucket" in event["name"] for event in spans)
    assert profiler.complete
    assert len(module.handles) == 1
    assert module.handles[0][1].removed
    profiler.begin_iteration(8, _FakeStream(21.0))
    assert not profiler.active


def test_unfinished_compute_aborts_profile_instead_of_extending_to_iteration_end(tmp_path):
    profiler = GTPRuntimeProfiler(
        GTPRuntimeProfileConfig(warmup_iters=0, profile_iters=1, log_dir=tmp_path),
        _recorder(),
    )
    profiler.begin_iteration(2, _FakeStream(0.0))
    profiler.compute_start("output_layer", GTPPhase.FORWARD, _FakeStream(1.0))
    profiler.end_iteration(_FakeStream(100.0))

    assert "unfinished compute operations" in profiler.errors[0]
    assert profiler.recorder.pending_iterations == 0
    assert not (tmp_path / "rank00000_gtp_execution_model.json").exists()


def test_attach_model_profiles_only_the_routed_expert_group_leader(tmp_path):
    leader = _FakeParameter("experts.weight0", expert_idx=0)
    sibling = _FakeParameter("experts.weight1", expert_idx=1)
    module = _FakeModule([leader, sibling])
    profiler = GTPRuntimeProfiler(
        GTPRuntimeProfileConfig(warmup_iters=0, profile_iters=1, log_dir=tmp_path),
        _recorder(),
    )

    assert profiler.attach_model(module) == 1
    assert profiler._parameters == {"experts.weight0": leader}
    assert len(module.handles) == 1


def test_forward_end_does_not_close_backward_compute(tmp_path):
    profiler = GTPRuntimeProfiler(
        GTPRuntimeProfileConfig(warmup_iters=1, profile_iters=1, log_dir=tmp_path),
        _recorder(),
    )
    profiler.begin_iteration(1, _FakeStream(0.0))
    profiler.compute_start("layer", GTPPhase.BACKWARD, _FakeStream(1.0))

    profiler.forward_compute_end("layer", _FakeStream(2.0))
    rs = profiler.rs_ready("layer", _FakeStream(3.0))

    assert rs is not None
    assert rs.key.occurrence == 0
    profiler.end_iteration(_FakeStream(4.0))
    assert not profiler.errors
