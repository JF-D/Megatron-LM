# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math

from megatron.core.communication_planner import (
    GTPCudaSample,
    GTPExecutionModel,
    GTPPhase,
    GTPProfileKey,
    GTPWorkKind,
)


def _key(scope, phase, kind):
    return GTPProfileKey(scope=scope, phase=phase, kind=kind)


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
