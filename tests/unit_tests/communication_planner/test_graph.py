# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.communication_planner import (
    DependencyKind,
    OperationGraphBuilder,
    OperationKind,
    OperationSpec,
    Phase,
    ReusableBufferSpec,
    SemanticOpId,
    Trigger,
)


def _id(scope, role, phase=Phase.FORWARD):
    return SemanticOpId(scope=scope, phase=phase, role=role)


def test_builder_derives_data_and_communicator_edges_without_observed_order():
    producer = _id("L10", "producer")
    consumer = _id("L10", "consumer")
    ag0 = _id("L10", "ag0")
    ag1 = _id("L11", "ag1")

    builder = OperationGraphBuilder()
    for op_id in (producer, consumer):
        builder.add_operation(OperationSpec(op_id=op_id, kind=OperationKind.COMPUTE))
    builder.add_operation(
        OperationSpec(
            op_id=ag0,
            kind=OperationKind.GTP_DENSE_AG,
            ready_trigger=Trigger.op_end(producer),
            deadline_trigger=Trigger.op_start(consumer),
            communicator_id="dense_gtp",
            sequence=0,
        )
    )
    builder.add_operation(
        OperationSpec(
            op_id=ag1,
            kind=OperationKind.GTP_DENSE_AG,
            ready_trigger=Trigger.window_start(Phase.FORWARD),
            communicator_id="dense_gtp",
            sequence=1,
        )
    )

    graph = builder.build()
    dependencies = {(edge.src, edge.dst, edge.kind) for edge in graph.dependencies}

    assert (producer, ag0, DependencyKind.DATA) in dependencies
    assert (ag0, consumer, DependencyKind.DATA) in dependencies
    assert (ag0, ag1, DependencyKind.COMMUNICATOR_ORDER) in dependencies
    assert (consumer, ag1, DependencyKind.CONTROL) not in dependencies


def test_graph_fingerprint_is_independent_of_registration_order():
    ids = [_id("L10", "compute"), _id("L10", "ag")]
    compute = OperationSpec(op_id=ids[0], kind=OperationKind.COMPUTE)
    ag = OperationSpec(
        op_id=ids[1],
        kind=OperationKind.GTP_DENSE_AG,
        ready_trigger=Trigger.window_start(Phase.FORWARD),
        deadline_trigger=Trigger.op_start(ids[0]),
        resources=frozenset({"fabric"}),
    )

    first = OperationGraphBuilder().add_operation(compute).add_operation(ag).build()
    second = OperationGraphBuilder().add_operation(ag).add_operation(compute).build()

    assert first.fingerprint == second.fingerprint
    assert first.topological_order() == second.topological_order()


def test_consumer_ready_deadline_keeps_explicit_data_dependency():
    producer = _id("L10", "producer")
    consumer = _id("L10", "consumer")
    ag = _id("L10", "ag")
    builder = OperationGraphBuilder()
    builder.add_operation(OperationSpec(op_id=producer, kind=OperationKind.COMPUTE))
    builder.add_operation(OperationSpec(op_id=consumer, kind=OperationKind.COMPUTE))
    builder.add_operation(
        OperationSpec(
            op_id=ag,
            kind=OperationKind.GTP_DENSE_AG,
            ready_trigger=Trigger.op_end(producer),
            deadline_trigger=Trigger.consumer_ready(ag),
            release_trigger=Trigger.op_end(consumer),
        )
    )
    builder.add_dependency(ag, consumer)

    graph = builder.build()
    dependencies = {(edge.src, edge.dst, edge.kind) for edge in graph.dependencies}

    assert (producer, ag, DependencyKind.DATA) in dependencies
    assert (ag, consumer, DependencyKind.DATA) in dependencies
    assert all(edge.src != edge.dst for edge in graph.dependencies)


def test_builder_rejects_cycles():
    left = _id("L10", "left")
    right = _id("L10", "right")
    builder = OperationGraphBuilder()
    builder.add_operation(OperationSpec(op_id=left, kind=OperationKind.COMPUTE))
    builder.add_operation(OperationSpec(op_id=right, kind=OperationKind.COMPUTE))
    builder.add_dependency(left, right)
    builder.add_dependency(right, left)

    with pytest.raises(ValueError, match="contains a cycle"):
        builder.build()


def test_builder_rejects_duplicate_communicator_sequence():
    first = _id("L10", "ag")
    second = _id("L11", "ag")
    builder = OperationGraphBuilder()
    for op_id in (first, second):
        builder.add_operation(
            OperationSpec(
                op_id=op_id,
                kind=OperationKind.GTP_DENSE_AG,
                ready_trigger=Trigger.window_start(Phase.FORWARD),
                communicator_id="dense_gtp",
                sequence=0,
            )
        )

    with pytest.raises(ValueError, match="Duplicate sequence"):
        builder.build()


def test_reusable_buffer_requires_release_trigger_and_changes_fingerprint():
    op_id = _id("L10", "ag")
    first_slot = ReusableBufferSpec(arena="gtp_cache", slot=0, capacity_bytes=4096)
    second_slot = ReusableBufferSpec(arena="gtp_cache", slot=1, capacity_bytes=4096)

    with pytest.raises(ValueError, match="requires a release_trigger"):
        OperationSpec(
            op_id=op_id,
            kind=OperationKind.GTP_DENSE_AG,
            ready_trigger=Trigger.window_start(Phase.FORWARD),
            reusable_buffers=(first_slot,),
        )

    def build(slot):
        return (
            OperationGraphBuilder()
            .add_operation(
                OperationSpec(
                    op_id=op_id,
                    kind=OperationKind.GTP_DENSE_AG,
                    ready_trigger=Trigger.window_start(Phase.FORWARD),
                    release_trigger=Trigger.consumer_ready(op_id),
                    reusable_buffers=(slot,),
                )
            )
            .build()
        )

    assert build(first_slot).fingerprint != build(second_slot).fingerprint
