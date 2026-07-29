# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math

from megatron.core.communication_planner import (
    GTPCudaSample,
    GTPCommDomain,
    GTPExecutionModel,
    GTPModuleInfo,
    GTPPhase,
    GTPProfileKey,
    GTPWorkKind,
)
from megatron.core.communication_planner.trace import build_gtp_chrome_trace


def _key(scope, phase, kind, domain=GTPCommDomain.GTP):
    return GTPProfileKey(scope=scope, phase=phase, kind=kind, domain=domain)


def _sample(key, iteration, duration_us):
    return GTPCudaSample(
        key=key,
        iteration=iteration,
        start_us=10.0,
        end_us=10.0 + duration_us,
    )


def test_compact_model_has_only_required_gtp_dependencies():
    f0 = _key("layer0", GTPPhase.FORWARD, GTPWorkKind.COMPUTE)
    f1 = _key("layer1", GTPPhase.FORWARD, GTPWorkKind.COMPUTE)
    b1 = _key("layer1", GTPPhase.BACKWARD, GTPWorkKind.COMPUTE)
    b0 = _key("layer0", GTPPhase.BACKWARD, GTPWorkKind.COMPUTE)
    operations = [
        f0,
        f1,
        b1,
        b0,
        _key("layer0", GTPPhase.FORWARD, GTPWorkKind.AG),
        _key("layer1", GTPPhase.FORWARD, GTPWorkKind.AG),
        _key("layer1", GTPPhase.BACKWARD, GTPWorkKind.AG),
        _key("layer0", GTPPhase.BACKWARD, GTPWorkKind.AG),
        _key("layer1", GTPPhase.BACKWARD, GTPWorkKind.RS),
        _key("layer0", GTPPhase.BACKWARD, GTPWorkKind.RS),
    ]
    samples = [
        _sample(key, iteration, 100.0 + iteration)
        for key in operations
        for iteration in (3, 4)
    ]

    model = GTPExecutionModel(
        phase_orders={
            GTPPhase.FORWARD: (f0, f1),
            GTPPhase.BACKWARD: (b1, b0),
        },
        samples=samples,
        parameter_chains={"GTP_ungraphed": ("layer0", "layer1")},
    )

    edges = {(edge.src, edge.dst, edge.kind) for edge in model.dependencies}
    assert (f0, f1, "compute_order") in edges
    assert (b1, b0, "compute_order") in edges
    assert (
        _key("layer1", GTPPhase.BACKWARD, GTPWorkKind.AG),
        b1,
        "ag_before_compute",
    ) in edges
    assert (
        b1,
        _key("layer1", GTPPhase.BACKWARD, GTPWorkKind.RS),
        "compute_before_rs",
    ) in edges
    assert not any(
        edge.src.kind is GTPWorkKind.RS and edge.dst.kind is GTPWorkKind.COMPUTE
        for edge in model.dependencies
    )
    assert math.isclose(model.statistics[f0].p50_us, 103.5)
    assert model.to_dict()["parameter_chains"]["GTP_ungraphed"] == ["layer0", "layer1"]


def test_backward_direct_rs_does_not_require_an_ag_node():
    embedding = _key("embedding", GTPPhase.BACKWARD, GTPWorkKind.COMPUTE)
    rs = _key("embedding", GTPPhase.BACKWARD, GTPWorkKind.RS)
    model = GTPExecutionModel(
        phase_orders={GTPPhase.BACKWARD: (embedding,)},
        samples=(_sample(embedding, 0, 0.0), _sample(rs, 0, 200.0)),
    )

    assert [(edge.src, edge.dst) for edge in model.dependencies] == [(embedding, rs)]


def test_trace_separates_gtp_and_egtp_communication_lanes():
    dense_compute = _key("dense", GTPPhase.FORWARD, GTPWorkKind.COMPUTE)
    expert_compute = _key(
        "expert",
        GTPPhase.FORWARD,
        GTPWorkKind.COMPUTE,
        GTPCommDomain.EGTP,
    )
    dense_ag = _key("dense", GTPPhase.FORWARD, GTPWorkKind.AG)
    expert_ag = _key(
        "expert",
        GTPPhase.FORWARD,
        GTPWorkKind.AG,
        GTPCommDomain.EGTP,
    )
    samples = tuple(
        _sample(key, 0, 100.0)
        for key in (dense_compute, expert_compute, dense_ag, expert_ag)
    )
    model = GTPExecutionModel(
        phase_orders={GTPPhase.FORWARD: (dense_compute, expert_compute)},
        samples=samples,
    )

    payload = model.to_dict()
    assert payload["format_version"] == 3
    assert {
        operation["communication_domain"] for operation in payload["operations"]
    } == {"gtp", "egtp"}
    assert {
        (edge.src.domain, edge.dst.domain)
        for edge in model.dependencies
        if edge.kind == "ag_before_compute"
    } == {
        (GTPCommDomain.GTP, GTPCommDomain.GTP),
        (GTPCommDomain.EGTP, GTPCommDomain.EGTP),
    }

    trace = build_gtp_chrome_trace(samples, model.dependencies, rank=0)
    lane_names = {
        event["args"]["name"]: event["tid"]
        for event in trace["traceEvents"]
        if event["ph"] == "M" and event["name"] == "thread_name"
    }
    assert lane_names["Forward GTP AG"] != lane_names["Forward EGTP AG"]
    communication_spans = [
        event
        for event in trace["traceEvents"]
        if event["ph"] == "X" and event["args"]["kind"] == "ag"
    ]
    assert {
        (event["args"]["communication_domain"], event["tid"])
        for event in communication_spans
    } == {
        ("gtp", lane_names["Forward GTP AG"]),
        ("egtp", lane_names["Forward EGTP AG"]),
    }


def test_compute_element_spans_consecutive_gtp_consumers_and_modules():
    source_compute = _key("mamba.in_proj", GTPPhase.FORWARD, GTPWorkKind.COMPUTE)
    target_compute = _key("mamba.out_proj", GTPPhase.FORWARD, GTPWorkKind.COMPUTE)
    element = _key(
        "mamba.in_proj",
        GTPPhase.FORWARD,
        GTPWorkKind.COMPUTE_ELEMENT,
    )
    target_materialize = _key(
        "mamba.out_proj",
        GTPPhase.FORWARD,
        GTPWorkKind.MATERIALIZE,
    )
    modules = {
        "mamba.in_proj": GTPModuleInfo("decoder.layers.0", "M", 1),
        "mamba.out_proj": GTPModuleInfo("decoder.layers.0", "M", 1),
    }
    samples = (
        _sample(source_compute, 0, 100.0),
        _sample(target_compute, 0, 100.0),
        GTPCudaSample(element, 0, 10.0, 910.0),
        GTPCudaSample(target_materialize, 0, 910.0, 960.0),
    )
    model = GTPExecutionModel(
        phase_orders={GTPPhase.FORWARD: (source_compute, target_compute)},
        samples=samples,
        parameter_modules=modules,
        compute_element_targets={element: target_materialize},
    )

    assert (
        element,
        target_materialize,
        "compute_before_materialize",
    ) in {
        (edge.src, edge.dst, edge.kind) for edge in model.dependencies
    }
    element_payload = next(
        operation
        for operation in model.to_dict()["operations"]
        if operation["kind"] == "compute_element"
    )
    assert model.to_dict()["compute_element_order"] == [element.stable_key]
    assert element_payload["module"]["symbol"] == "M"
    assert element_payload["next_consumer"]["scope"] == "mamba.out_proj"

    trace = build_gtp_chrome_trace(
        samples,
        model.dependencies,
        rank=0,
        parameter_modules=model.parameter_modules,
        compute_element_targets=model.compute_element_targets,
    )
    span = next(
        event
        for event in trace["traceEvents"]
        if event.get("ph") == "X"
        and event["args"]["kind"] == "compute_element"
    )
    assert span["args"]["cuda_duration_us"] == 900.0
    assert span["args"]["module"]["symbol"] == "M"
    assert span["args"]["next_consumer"]["scope"] == "mamba.out_proj"
    materialize_span = next(
        event
        for event in trace["traceEvents"]
        if event.get("ph") == "X"
        and event["args"]["kind"] == "materialize"
    )
    assert span["tid"] == materialize_span["tid"]
