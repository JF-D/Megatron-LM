# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import math

import megatron.core.communication_planner.runtime as runtime_module
from megatron.core.communication_planner import (
    GTPCudaEventRecorder,
    GTPCommDomain,
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
        self.backward_pre_handles = []

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

    def register_full_backward_pre_hook(self, hook):
        handle = _FakeHandle()
        self.backward_pre_handles.append((hook, handle))
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
    assert payload["format_version"] == 3
    assert payload["diagnostics"]["timing_source"] == "cuda_events"
    assert payload["diagnostics"]["trace_file"] == "rank00000_gtp_execution_trace.json"
    assert payload["diagnostics"]["consumer_count"] == 2
    assert payload["diagnostics"]["compute_element_count"] == 1
    assert {
        operation["communication_domain"] for operation in payload["operations"]
    } == {"gtp"}
    assert {
        dependency["kind"] for dependency in payload["dependencies"]
    } == {
        "ag_before_compute",
        "compute_before_materialize",
        "compute_before_rs",
        "materialize_before_compute",
    }

    trace = json.loads((tmp_path / payload["diagnostics"]["trace_file"]).read_text())
    spans = [event for event in trace["traceEvents"] if event["ph"] == "X"]
    assert {event["args"]["kind"] for event in spans} == {
        "compute_element",
        "materialize",
        "compute",
        "ag",
        "rs",
    }
    assert sum(event["ph"] == "s" for event in trace["traceEvents"]) == 3
    assert not any("finalize" in event["name"] or "bucket" in event["name"] for event in spans)
    assert profiler.complete
    assert len(module.handles) == 1
    assert module.handles[0][1].removed
    profiler.begin_iteration(8, _FakeStream(21.0))
    assert not profiler.active


def test_compute_element_captures_work_until_next_consumer(tmp_path):
    profiler = GTPRuntimeProfiler(
        GTPRuntimeProfileConfig(warmup_iters=0, profile_iters=1, log_dir=tmp_path),
        _recorder(),
    )
    profiler.begin_iteration(4, _FakeStream(0.0))

    first_ready = profiler.consumer_ready(
        "mamba.in_proj",
        GTPPhase.FORWARD,
        _FakeStream(1.0),
    )
    profiler.compute_start(
        "mamba.in_proj",
        GTPPhase.FORWARD,
        _FakeStream(2.0),
        materialize_token=first_ready,
    )
    profiler.forward_compute_end("mamba.in_proj", _FakeStream(3.0))
    second_ready = profiler.consumer_ready(
        "mamba.out_proj",
        GTPPhase.FORWARD,
        _FakeStream(8.0),
    )
    profiler.compute_start(
        "mamba.out_proj",
        GTPPhase.FORWARD,
        _FakeStream(9.0),
        materialize_token=second_ready,
    )
    profiler.forward_compute_end("mamba.out_proj", _FakeStream(10.0))
    profiler.end_iteration(_FakeStream(11.0))
    model = profiler.build_model()

    element = next(
        key
        for key in model.statistics
        if key.scope == "mamba.in_proj"
        and key.kind is GTPWorkKind.COMPUTE_ELEMENT
    )
    materialize = next(
        key
        for key in model.statistics
        if key.scope == "mamba.out_proj"
        and key.kind is GTPWorkKind.MATERIALIZE
    )
    assert math.isclose(model.statistics[element].p50_us, 6000.0)
    assert math.isclose(model.statistics[materialize].p50_us, 1000.0)
    assert model.compute_element_targets[element] == materialize


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
    assert profiler._communication_domains == {"experts.weight0": GTPCommDomain.EGTP}
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


def test_embedding_hooks_capture_direct_gradient_backward_compute(tmp_path):
    module = _FakeModule([_FakeParameter("embedding.weight")])
    module._gtp_runtime_profile_embedding = True
    profiler = GTPRuntimeProfiler(
        GTPRuntimeProfileConfig(warmup_iters=0, profile_iters=1, log_dir=tmp_path),
        _recorder(),
    )
    assert profiler.attach_model(module) == 1
    assert profiler._communication_domains == {
        "embedding.weight": GTPCommDomain.GTP
    }
    assert len(module.handles) == 1
    assert len(module.backward_pre_handles) == 1

    profiler.begin_iteration(3, _FakeStream(0.0))
    profiler.compute_start("embedding.weight", GTPPhase.FORWARD, _FakeStream(1.0))
    original_current_stream = runtime_module._current_cuda_stream
    try:
        runtime_module._current_cuda_stream = lambda: _FakeStream(3.0)
        module.handles[0][0](module, (), None)
        runtime_module._current_cuda_stream = lambda: _FakeStream(5.0)
        module.backward_pre_handles[0][0](module, ())
    finally:
        runtime_module._current_cuda_stream = original_current_stream

    rs = profiler.rs_ready("embedding.weight", _FakeStream(8.0))
    profiler.communication_start(rs, _FakeStream(8.1))
    profiler.communication_end(rs, _FakeStream(9.1))
    profiler.end_iteration(_FakeStream(10.0))
    model = profiler.build_model()

    compute = {
        key.phase: statistics.p50_us
        for key, statistics in model.statistics.items()
        if key.scope == "embedding.weight" and key.kind is GTPWorkKind.COMPUTE
    }
    assert math.isclose(compute[GTPPhase.FORWARD], 2000.0)
    assert math.isclose(compute[GTPPhase.BACKWARD], 3000.0)
    assert not profiler.errors


def test_hybrid_module_symbols_label_gtp_parameters():
    mamba_weight = _FakeParameter("decoder.layers.0.mixer.in_proj.weight")
    attention_weight = _FakeParameter("decoder.layers.1.self_attention.qkv.weight")
    expert_weight = _FakeParameter("decoder.layers.2.mlp.experts.weight0", expert_idx=0)
    mamba_layer = _FakeModule([mamba_weight])
    attention_layer = _FakeModule([attention_weight])
    expert_layer = _FakeModule([expert_weight])
    mamba_layer.layer_number = 1
    attention_layer.layer_number = 2
    expert_layer.layer_number = 3

    class _HybridContainer:
        layer_type_list = ["M", "*", "E"]
        layers = [mamba_layer, attention_layer, expert_layer]

    container = _HybridContainer()
    named_modules = [
        ("decoder", container),
        ("decoder.layers.0", mamba_layer),
        ("decoder.layers.1", attention_layer),
        ("decoder.layers.2", expert_layer),
    ]
    module_names = {
        id(module): name for name, module in named_modules
    }

    mapping = runtime_module._hybrid_module_info_by_parameter(
        named_modules,
        module_names,
    )

    assert mapping[id(mamba_weight)].symbol == "M"
    assert mapping[id(mamba_weight)].layer_number == 1
    assert mapping[id(attention_weight)].symbol == "*"
    assert mapping[id(expert_weight)].symbol == "E"
    assert mapping[id(expert_weight)].scope == "decoder.layers.2"
