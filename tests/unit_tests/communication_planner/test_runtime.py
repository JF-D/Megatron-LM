# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from contextlib import nullcontext

from megatron.core.communication_planner import (
    RuntimeCommunicationPlannerRuntime,
    RuntimePlannerConfig,
    RuntimePlannerMode,
)


class _FakeStream:
    def __init__(self, time_ms):
        self.time_ms = time_ms


class _FakeEvent:
    complete = True

    def __init__(self):
        self.time_ms = None

    def record(self, stream=None):
        self.time_ms = 0.0 if stream is None else stream.time_ms

    def query(self):
        return type(self).complete

    def elapsed_time(self, other):
        return other.time_ms - self.time_ms


def _runtime(
    tmp_path,
    *,
    mode=RuntimePlannerMode.SHADOW,
    warmup_iters=1,
    profile_iters=1,
    completion_stream_factory=None,
    stream_context=None,
):
    return RuntimeCommunicationPlannerRuntime(
        RuntimePlannerConfig(
            mode=mode,
            warmup_iters=warmup_iters,
            profile_iters=profile_iters,
            log_dir=str(tmp_path),
            dump_plan=True,
            validate_ranks="all",
        ),
        event_factory=_FakeEvent,
        completion_stream_factory=completion_stream_factory,
        stream_context=stream_context,
    )


def _observe_dense_ag(runtime, iteration, *, reusable_buffers=()):
    runtime.begin_iteration(iteration, _FakeStream(0.0))
    runtime.gtp_consumer_ready("decoder.layers.0.linear_qkv.weight", "forward", _FakeStream(0.1))
    token = runtime.gtp_ag_ready(
        "decoder.layers.0.linear_qkv.weight",
        expert=False,
        direction="forward",
        communicator_size=4,
        payload_bytes=4096,
        reusable_buffers=reusable_buffers,
        stream=_FakeStream(0.1),
    )
    runtime.collective_start(token, _FakeStream(0.2))
    runtime.collective_end(token, _FakeStream(0.8))
    runtime.gtp_consumer_resume(
        "decoder.layers.0.linear_qkv.weight", "forward", _FakeStream(0.8)
    )
    runtime.end_iteration(_FakeStream(1.0))


def _observe_dense_ag_with_param_bucket(runtime, iteration):
    scope = "decoder.layers.0.linear_qkv.weight"
    runtime.begin_iteration(iteration, _FakeStream(0.0))
    runtime.param_bucket_ready(
        "ddp_param_bucket_0",
        (scope,),
        _FakeStream(0.05),
    )
    runtime.gtp_consumer_ready(scope, "forward", _FakeStream(0.1))
    token = runtime.gtp_ag_ready(
        scope,
        expert=False,
        direction="forward",
        communicator_size=4,
        payload_bytes=4096,
        parameter_scopes=(scope,),
        stream=_FakeStream(0.1),
    )
    runtime.collective_start(token, _FakeStream(0.2))
    runtime.collective_end(token, _FakeStream(0.8))
    runtime.gtp_consumer_resume(scope, "forward", _FakeStream(0.8))
    runtime.end_iteration(_FakeStream(1.0))


def _observe_ep_combine(runtime, iteration, payload_bytes):
    from megatron.core.communication_planner import OperationKind, Phase

    runtime.begin_iteration(iteration, _FakeStream(0.0))
    token = runtime.ep_start(
        "decoder.layers.1.mlp.token_dispatcher",
        phase=Phase.FORWARD,
        kind=OperationKind.EP_COMBINE,
        communicator_size=4,
        payload_bytes=payload_bytes,
        stream=_FakeStream(0.1),
    )
    runtime.ep_end(token, _FakeStream(0.8))
    runtime.end_iteration(_FakeStream(1.0))


def _observe_direct_dense_rs(runtime, iteration):
    runtime.begin_iteration(iteration, _FakeStream(0.0))
    token = runtime.gtp_rs_ready(
        "embedding.word_embeddings.weight",
        expert=False,
        communicator_size=4,
        payload_bytes=4096,
        stream=_FakeStream(0.2),
    )
    runtime.collective_start(token, _FakeStream(0.3))
    runtime.collective_end(token, _FakeStream(0.8))
    runtime.gtp_rs_consumer_ready(
        "embedding.word_embeddings.weight", _FakeStream(0.8)
    )
    runtime.gtp_rs_consumer_resume(
        "embedding.word_embeddings.weight", _FakeStream(0.8)
    )
    runtime.gtp_rs_finalize_end(
        "embedding.word_embeddings.weight", _FakeStream(0.9)
    )
    runtime.end_iteration(_FakeStream(1.0))


def test_runtime_discovers_profiles_compiles_and_dumps(tmp_path):
    runtime = _runtime(tmp_path)

    _observe_dense_ag(runtime, 0)
    _observe_dense_ag(runtime, 1)

    diagnostics = runtime.diagnostics
    assert diagnostics["requested_mode"] == "shadow"
    assert diagnostics["effective_mode"] == "shadow"
    assert diagnostics["enforcement_active"] is False
    assert diagnostics["graph_acyclic"] is True
    assert diagnostics["telemetry_complete"] is True
    assert diagnostics["plan_fingerprint"] is not None
    assert diagnostics["plan_consensus"] is True
    assert diagnostics["consensus_group_ranks"] == [0]
    assert diagnostics["consensus_samples_per_operation"] == 1
    assert diagnostics["hook_cpu_overhead_us"] >= 0.0
    assert diagnostics["control_plane_cpu_overhead_us"] >= 0.0
    assert diagnostics["total_planner_cpu_overhead_us"] == (
        diagnostics["setup_cpu_overhead_us"]
        + diagnostics["hook_cpu_overhead_us"]
        + diagnostics["control_plane_cpu_overhead_us"]
    )

    graph = json.loads((tmp_path / "rank00000_graph.json").read_text())
    operation_ids = {operation["id"] for operation in graph["operations"]}
    ag_id = "forward:mb0:decoder.layers.0.linear_qkv.weight:fwd_ag:0"
    compute_id = "forward:mb0:decoder.layers.0.linear_qkv.weight:fwd_compute:0"
    assert ag_id in operation_ids
    ag = next(operation for operation in graph["operations"] if operation["id"] == ag_id)
    assert ag["deadline_trigger"] == f"{ag_id}:op_consumer_ready"
    assert {
        "src": ag_id,
        "dst": compute_id,
        "kind": "data",
    } in graph["dependencies"]
    assert graph["fingerprint"] == diagnostics["graph_fingerprint"]
    assert (tmp_path / "rank00000_telemetry.json").exists()
    assert (tmp_path / "rank00000_plan.json").exists()


def test_runtime_serializes_reusable_buffer_metadata(tmp_path):
    runtime = _runtime(tmp_path)
    buffer = ("gtp_cache:dense_gtp", 3, 4096, 0)

    _observe_dense_ag(runtime, 0, reusable_buffers=(buffer,))
    _observe_dense_ag(runtime, 1, reusable_buffers=(buffer,))

    graph = json.loads((tmp_path / "rank00000_graph.json").read_text())
    plan = json.loads((tmp_path / "rank00000_plan.json").read_text())
    ag_id = "forward:mb0:decoder.layers.0.linear_qkv.weight:fwd_ag:0"
    expected = [
        {
            "arena": "gtp_cache:dense_gtp",
            "slot": 3,
            "capacity_bytes": 4096,
            "generation": 0,
        }
    ]
    graph_ag = next(operation for operation in graph["operations"] if operation["id"] == ag_id)
    plan_ag = next(action for action in plan["actions"] if action["op_id"] == ag_id)

    assert graph_ag["reusable_buffers"] == expected
    assert plan_ag["reusable_buffers"] == expected


def test_async_completion_is_distinct_from_production_drain(tmp_path):
    completion_stream = _FakeStream(0.5)
    runtime = _runtime(
        tmp_path,
        completion_stream_factory=lambda: completion_stream,
        stream_context=lambda stream: nullcontext(stream),
    )

    class FakeWork:
        wait_calls = 0

        def wait(self):
            self.wait_calls += 1

    def observe(iteration):
        runtime.begin_iteration(iteration, _FakeStream(0.0))
        scope = "decoder.layers.0.linear_qkv.weight"
        runtime.gtp_consumer_ready(scope, "forward", _FakeStream(0.1))
        token = runtime.gtp_ag_ready(
            scope,
            expert=False,
            direction="forward",
            communicator_size=4,
            payload_bytes=4096,
            stream=_FakeStream(0.1),
        )
        runtime.collective_start(token, _FakeStream(0.2))
        work = FakeWork()
        keepalive = (object(), object())
        runtime.collective_completion(token, work, keepalive=keepalive)
        retained_at_completion = token.keepalive
        runtime.collective_end(token, _FakeStream(0.8))
        retained_after_drain = token.keepalive
        runtime.gtp_consumer_resume(scope, "forward", _FakeStream(0.8))
        runtime.end_iteration(_FakeStream(1.0))
        return work, keepalive, retained_at_completion, retained_after_drain

    (
        discovery_work,
        discovery_expected_keepalive,
        discovery_keepalive,
        discovery_after_drain,
    ) = observe(0)
    (
        profile_work,
        profile_expected_keepalive,
        profile_keepalive,
        profile_after_drain,
    ) = observe(1)

    assert discovery_work.wait_calls == 0
    assert discovery_keepalive == ()
    assert discovery_after_drain == ()
    assert discovery_expected_keepalive
    assert profile_work.wait_calls == 1
    assert profile_keepalive == profile_expected_keepalive
    assert profile_after_drain == ()
    telemetry = json.loads((tmp_path / "rank00000_telemetry.json").read_text())
    ag_id = "forward:mb0:decoder.layers.0.linear_qkv.weight:fwd_ag:0"
    sample = next(item for item in telemetry["samples"] if item["op_id"] == ag_id)
    assert sample["start_us"] == 200.0
    assert sample["end_us"] == 500.0
    assert sample["service_us"] == 300.0
    assert sample["drain_us"] == 800.0
    assert sample["drain_delay_us"] == 300.0


def test_runtime_status_dump_is_state_bounded_and_finalized(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    writes = []
    original_write_json = runtime._write_json

    def record_write(stem, payload):
        if stem == "diagnostics":
            writes.append(payload["profile_iterations_completed"])
        original_write_json(stem, payload)

    monkeypatch.setattr(runtime, "_write_json", record_write)
    _observe_dense_ag(runtime, 0)
    _observe_dense_ag(runtime, 1)
    writes_after_plan = len(writes)

    for iteration in range(2, 8):
        _observe_dense_ag(runtime, iteration)

    assert len(writes) == writes_after_plan
    runtime.finalize()
    assert len(writes) == writes_after_plan + 1


def test_runtime_compiles_from_topology_equivalent_rank_telemetry(tmp_path):
    runtime = _runtime(tmp_path)

    def gather(payload):
        remote = dict(payload)
        remote["rank"] = 1
        return [payload, remote]

    runtime._all_gather_object = gather
    _observe_dense_ag(runtime, 0)
    _observe_dense_ag(runtime, 1)

    diagnostics = runtime.diagnostics
    assert diagnostics["fallback_reason"] is None
    assert diagnostics["plan_consensus"] is True
    assert diagnostics["consensus_group_ranks"] == [0, 1]
    assert diagnostics["consensus_samples_per_operation"] == 2


def test_forward_gtp_ag_uses_ddp_param_bucket_readiness(tmp_path):
    runtime = _runtime(tmp_path)

    _observe_dense_ag_with_param_bucket(runtime, 0)
    _observe_dense_ag_with_param_bucket(runtime, 1)

    graph = json.loads((tmp_path / "rank00000_graph.json").read_text())
    ag_id = "forward:mb0:decoder.layers.0.linear_qkv.weight:fwd_ag:0"
    bucket_id = "forward:mb0:ddp_param_bucket_0:dp_param_ready:0"
    ag = next(operation for operation in graph["operations"] if operation["id"] == ag_id)
    assert ag["ready_trigger"] == f"{bucket_id}:op_end"
    assert {
        "src": bucket_id,
        "dst": ag_id,
        "kind": "data",
    } in graph["dependencies"]

    telemetry = json.loads((tmp_path / "rank00000_telemetry.json").read_text())
    ag_sample = next(
        sample for sample in telemetry["samples"] if sample["op_id"] == ag_id
    )
    assert ag_sample["ready_us"] == 50.0
    assert runtime.diagnostics["gtp_ag_ready_sources"][ag_id] == "ddp_param_bucket"
    assert runtime.diagnostics["missing_forward_param_ready_scopes"] == []


def test_runtime_waits_for_all_ranks_before_telemetry_exchange(tmp_path):
    runtime = _runtime(tmp_path)
    runtime._world_size = 2
    readiness_round = 0

    def gather(payload):
        nonlocal readiness_round
        remote = dict(payload)
        remote["rank"] = 1
        if "ready" in payload:
            readiness_round += 1
            if readiness_round == 1:
                remote["ready"] = False
                remote["profile_completed"] = 0
                remote["pending_iterations"] = 1
        return [payload, remote]

    runtime._all_gather_object = gather
    _observe_dense_ag(runtime, 0)
    _observe_dense_ag(runtime, 1)

    assert runtime.diagnostics["plan_fingerprint"] is None
    assert runtime.diagnostics["profile_readiness_rounds"] == 1

    _observe_dense_ag(runtime, 2)

    diagnostics = runtime.diagnostics
    assert diagnostics["plan_consensus"] is True
    assert diagnostics["plan_fingerprint"] is not None
    assert diagnostics["profile_readiness_rounds"] == 2


def test_runtime_propagates_profile_failure_before_plan_collectives(tmp_path):
    runtime = _runtime(tmp_path)
    runtime._world_size = 2

    def gather(payload):
        remote = dict(payload)
        remote["rank"] = 1
        if "ready" in payload:
            remote["ready"] = False
            remote["disabled"] = True
            remote["error"] = "semantic graph changed during profile"
        return [payload, remote]

    runtime._all_gather_object = gather
    _observe_dense_ag(runtime, 0)
    _observe_dense_ag(runtime, 1)

    diagnostics = runtime.diagnostics
    assert diagnostics["plan_consensus"] is False
    assert diagnostics["plan_fingerprint"] is None
    assert "rank-local failure" in diagnostics["fallback_reason"]


def test_topology_equivalence_ignores_rank_local_parallel_coordinates():
    rank0 = {
        "world_size": 32,
        "gtp_size": 16,
        "parallel_coordinates": {
            "pipeline_parallel_rank": 0,
            "gtp_rank": 0,
            "expert_parallel_rank": 0,
            "data_parallel_rank": 0,
        },
        "communicator_memberships": {"dense_gtp": list(range(16))},
    }
    rank17 = {
        "world_size": 32,
        "gtp_size": 16,
        "parallel_coordinates": {
            "pipeline_parallel_rank": 0,
            "gtp_rank": 1,
            "expert_parallel_rank": 1,
            "data_parallel_rank": 1,
        },
        "communicator_memberships": {"dense_gtp": list(range(16, 32))},
    }

    signature = RuntimeCommunicationPlannerRuntime._topology_equivalence_signature
    assert signature(rank0) == signature(rank17)

    rank17["parallel_coordinates"]["pipeline_parallel_rank"] = 1
    assert signature(rank0) != signature(rank17)


def test_enforce_request_remains_observable_shadow_fallback(tmp_path):
    runtime = _runtime(tmp_path, mode=RuntimePlannerMode.ENFORCE)

    _observe_dense_ag(runtime, 0)
    _observe_dense_ag(runtime, 1)

    diagnostics = runtime.diagnostics
    assert diagnostics["requested_mode"] == "enforce"
    assert diagnostics["effective_mode"] == "shadow"
    assert diagnostics["enforcement_active"] is False
    assert "rank-consensus" in diagnostics["fallback_reason"]
    assert diagnostics["plan_fingerprint"] is not None


def test_pending_cuda_event_iterations_are_bounded_by_profile_window(tmp_path):
    runtime = _runtime(tmp_path, profile_iters=2)
    _observe_dense_ag(runtime, 0)

    _FakeEvent.complete = False
    try:
        _observe_dense_ag(runtime, 1)
        _observe_dense_ag(runtime, 2)
        _observe_dense_ag(runtime, 3)
        _observe_dense_ag(runtime, 4)
    finally:
        _FakeEvent.complete = True

    diagnostics = runtime.diagnostics
    assert diagnostics["profile_iterations_started"] == 2
    assert diagnostics["pending_event_iterations"] == 2
    assert diagnostics["max_pending_event_iterations"] <= 2
    assert diagnostics["skipped_profile_iterations"] == 0


def test_lazy_initialization_before_final_warmup_does_not_pollute_graph(tmp_path):
    runtime = _runtime(tmp_path, warmup_iters=2)

    # This first-use operation is burn-in only.
    _observe_dense_ag(runtime, 0)
    _observe_ep_combine(runtime, 1, 1024)
    _observe_ep_combine(runtime, 2, 2048)

    diagnostics = runtime.diagnostics
    assert diagnostics["fallback_reason"] is None
    assert diagnostics["plan_fingerprint"] is not None
    graph = json.loads((tmp_path / "rank00000_graph.json").read_text())
    operation_ids = {operation["id"] for operation in graph["operations"]}
    assert not any("linear_qkv" in op_id for op_id in operation_ids)
    assert any("ep_combine" in op_id for op_id in operation_ids)


def test_dynamic_ep_payload_is_not_part_of_graph_identity(tmp_path):
    runtime = _runtime(tmp_path)

    _observe_ep_combine(runtime, 0, 1024)
    _observe_ep_combine(runtime, 1, 2048)

    diagnostics = runtime.diagnostics
    assert diagnostics["fallback_reason"] is None
    payloads = diagnostics["dynamic_payload_bytes"]
    op_id = "forward:mb0:decoder.layers.1.mlp.token_dispatcher:ep_combine:0"
    assert payloads[op_id] == {
        "samples": [1024, 2048],
        "count": 2,
        "min": 1024,
        "max": 2048,
    }
    graph = json.loads((tmp_path / "rank00000_graph.json").read_text())
    combine = next(operation for operation in graph["operations"] if operation["id"] == op_id)
    assert combine["bytes"] == 0


def test_profile_graph_drift_disables_shadow_without_raising(tmp_path):
    runtime = _runtime(tmp_path)

    _observe_dense_ag(runtime, 0)
    runtime.begin_iteration(1, _FakeStream(0.0))
    runtime.end_iteration(_FakeStream(1.0))

    diagnostics = runtime.diagnostics
    assert diagnostics["enforcement_active"] is False
    assert "semantic graph changed during profile" in diagnostics["fallback_reason"]
    assert diagnostics["plan_fingerprint"] is None


def test_direct_rs_input_records_point_producer(tmp_path):
    runtime = _runtime(tmp_path)

    _observe_direct_dense_rs(runtime, 0)
    _observe_direct_dense_rs(runtime, 1)

    diagnostics = runtime.diagnostics
    assert diagnostics["fallback_reason"] is None
    assert diagnostics["telemetry_complete"] is True
    telemetry = json.loads((tmp_path / "rank00000_telemetry.json").read_text())
    producer = next(
        sample
        for sample in telemetry["samples"]
        if sample["op_id"]
        == "backward:mb0:embedding.word_embeddings.weight:bwd_compute:0"
    )
    assert producer["start_us"] == producer["end_us"] == 200.0
