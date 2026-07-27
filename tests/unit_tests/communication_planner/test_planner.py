# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.communication_planner import (
    OperationGraphBuilder,
    OperationKind,
    OperationSample,
    OperationSpec,
    Phase,
    RuntimeCommunicationPlanner,
    RuntimePlanExecutor,
    SemanticOpId,
    SymmetricBufferSpec,
    TelemetryStore,
    Trigger,
)


def _id(scope, role, phase):
    return SemanticOpId(scope=scope, phase=phase, role=role)


def _example():
    route = _id("L9", "routing", Phase.FORWARD)
    ep_consumer = _id("L9", "residual", Phase.FORWARD)
    dense_consumer = _id("L10", "qkv_gemm", Phase.FORWARD)
    wgrad = _id("L10", "wgrad", Phase.BACKWARD)
    dgrad_consumer = _id("L10", "dgrad", Phase.BACKWARD)
    tail = _id("iteration", "tail", Phase.BACKWARD)

    ep = _id("L9", "ep_combine", Phase.FORWARD)
    ag = _id("L10", "fwd_ag", Phase.FORWARD)
    dgrad_ag = _id("L10", "dgrad_ag", Phase.BACKWARD)
    rs = _id("L11", "wgrad_rs", Phase.BACKWARD)

    builder = OperationGraphBuilder()
    for op_id in (route, ep_consumer, dense_consumer, wgrad, dgrad_consumer, tail):
        builder.add_operation(OperationSpec(op_id=op_id, kind=OperationKind.COMPUTE))

    builder.add_operation(
        OperationSpec(
            op_id=ep,
            kind=OperationKind.EP_COMBINE,
            ready_trigger=Trigger.window_start(Phase.FORWARD),
            deadline_trigger=Trigger.op_start(ep_consumer),
            resources=frozenset({"cross_domain_fabric"}),
            communicator_id="ep",
            sequence=0,
            priority=-1,
        )
    )
    builder.add_operation(
        OperationSpec(
            op_id=ag,
            kind=OperationKind.GTP_DENSE_AG,
            ready_trigger=Trigger.window_start(Phase.FORWARD),
            deadline_trigger=Trigger.op_start(dense_consumer),
            release_trigger=Trigger.op_end(dense_consumer),
            resources=frozenset({"cross_domain_fabric"}),
            communicator_id="dense_gtp_fwd",
            sequence=0,
        )
    )
    builder.add_operation(
        OperationSpec(
            op_id=dgrad_ag,
            kind=OperationKind.GTP_DENSE_AG,
            ready_trigger=Trigger.op_end(wgrad),
            deadline_trigger=Trigger.op_start(dgrad_consumer),
            release_trigger=Trigger.op_end(dgrad_consumer),
            resources=frozenset({"cross_domain_fabric"}),
            communicator_id="dense_gtp_bwd",
            sequence=0,
        )
    )
    builder.add_operation(
        OperationSpec(
            op_id=rs,
            kind=OperationKind.GTP_DENSE_RS,
            ready_trigger=Trigger.op_end(wgrad),
            resources=frozenset({"cross_domain_fabric"}),
            communicator_id="dense_gtp_rs",
            sequence=0,
        )
    )
    graph = builder.build()

    intervals = {
        route: (0.0, 100.0),
        ep_consumer: (1000.0, 1500.0),
        dense_consumer: (3000.0, 4500.0),
        wgrad: (6000.0, 7000.0),
        dgrad_consumer: (8500.0, 9000.0),
        tail: (10000.0, 10500.0),
        ep: (0.0, 1000.0),
        ag: (0.0, 1200.0),
        dgrad_ag: (7000.0, 8100.0),
        rs: (7000.0, 8500.0),
    }
    telemetry = TelemetryStore()
    telemetry.add_iteration(
        [
            OperationSample(op_id=op_id, iteration=0, start_us=start, end_us=end)
            for op_id, (start, end) in intervals.items()
        ],
        graph,
    )
    return graph, telemetry, {"ep": ep, "ag": ag, "dgrad_ag": dgrad_ag, "rs": rs, "wgrad": wgrad}


def test_planner_protects_ep_and_prioritizes_deadline_ag_over_rs():
    graph, telemetry, ids = _example()
    plan = RuntimeCommunicationPlanner().compile(graph, telemetry, epoch=3)
    by_id = {action.op_id: action for action in plan.actions}

    assert plan.epoch == 3
    assert plan.graph_fingerprint == graph.fingerprint
    assert plan.diagnostics.is_feasible

    assert by_id[ids["ep"]].planned_end_us <= by_id[ids["ag"]].planned_start_us
    assert by_id[ids["ag"]].planned_end_us <= 3000.0
    assert by_id[ids["dgrad_ag"]].planned_end_us <= by_id[ids["rs"]].planned_start_us
    assert by_id[ids["dgrad_ag"]].planned_end_us <= 8500.0
    assert by_id[ids["dgrad_ag"]].issue_trigger == Trigger.op_end(ids["wgrad"])


def test_planner_allows_independent_resources_to_overlap():
    consumer_a = _id("A", "consumer", Phase.FORWARD)
    consumer_b = _id("B", "consumer", Phase.FORWARD)
    ag_a = _id("A", "ag", Phase.FORWARD)
    ag_b = _id("B", "ag", Phase.FORWARD)
    builder = OperationGraphBuilder()
    for consumer in (consumer_a, consumer_b):
        builder.add_operation(OperationSpec(op_id=consumer, kind=OperationKind.COMPUTE))
    for ag, consumer, resource in ((ag_a, consumer_a, "rail0"), (ag_b, consumer_b, "rail1")):
        builder.add_operation(
            OperationSpec(
                op_id=ag,
                kind=OperationKind.GTP_DENSE_AG,
                ready_trigger=Trigger.window_start(Phase.FORWARD),
                deadline_trigger=Trigger.op_start(consumer),
                resources=frozenset({resource}),
            )
        )
    graph = builder.build()
    telemetry = TelemetryStore()
    telemetry.add_iteration(
        [
            OperationSample(op_id=consumer_a, iteration=0, start_us=2000.0, end_us=3000.0),
            OperationSample(op_id=consumer_b, iteration=0, start_us=2000.0, end_us=3000.0),
            OperationSample(op_id=ag_a, iteration=0, start_us=0.0, end_us=1000.0),
            OperationSample(op_id=ag_b, iteration=0, start_us=0.0, end_us=1000.0),
        ],
        graph,
    )

    plan = RuntimeCommunicationPlanner().compile(graph, telemetry, epoch=0)
    starts = {action.op_id: action.planned_start_us for action in plan.actions}
    assert starts[ag_a] == starts[ag_b] == 1000.0


def test_plan_reports_infeasible_deadline():
    consumer = _id("L10", "consumer", Phase.FORWARD)
    ag = _id("L10", "ag", Phase.FORWARD)
    graph = (
        OperationGraphBuilder()
        .add_operation(OperationSpec(op_id=consumer, kind=OperationKind.COMPUTE))
        .add_operation(
            OperationSpec(
                op_id=ag,
                kind=OperationKind.GTP_DENSE_AG,
                ready_trigger=Trigger.window_start(Phase.FORWARD),
                deadline_trigger=Trigger.op_start(consumer),
                resources=frozenset({"fabric"}),
            )
        )
        .build()
    )
    telemetry = TelemetryStore()
    telemetry.add_iteration(
        [
            OperationSample(op_id=consumer, iteration=0, start_us=1000.0, end_us=1500.0),
            OperationSample(op_id=ag, iteration=0, start_us=0.0, end_us=2000.0),
        ],
        graph,
    )

    plan = RuntimeCommunicationPlanner().compile(graph, telemetry, epoch=0)
    assert not plan.diagnostics.is_feasible
    assert plan.diagnostics.deadline_misses[0].op_id == ag
    assert plan.diagnostics.deadline_misses[0].lateness_us == 1000.0


def test_planner_reserves_symmetric_buffer_until_consumer_release():
    consumer_a = _id("A", "consumer", Phase.FORWARD)
    consumer_b = _id("B", "consumer", Phase.FORWARD)
    ag_a = _id("A", "ag", Phase.FORWARD)
    ag_b = _id("B", "ag", Phase.FORWARD)
    slot = SymmetricBufferSpec(arena="dense_gtp", slot=0, offset_bytes=0, capacity_bytes=4096)
    builder = OperationGraphBuilder()
    for consumer in (consumer_a, consumer_b):
        builder.add_operation(OperationSpec(op_id=consumer, kind=OperationKind.COMPUTE))
    builder.add_operation(
        OperationSpec(
            op_id=ag_a,
            kind=OperationKind.GTP_DENSE_AG,
            ready_trigger=Trigger.window_start(Phase.FORWARD),
            deadline_trigger=Trigger.op_start(consumer_a),
            release_trigger=Trigger.op_end(consumer_a),
            resources=frozenset({"rail0"}),
            symmetric_buffer=slot,
        )
    )
    builder.add_operation(
        OperationSpec(
            op_id=ag_b,
            kind=OperationKind.GTP_DENSE_AG,
            ready_trigger=Trigger.window_start(Phase.FORWARD),
            deadline_trigger=Trigger.op_start(consumer_b),
            release_trigger=Trigger.op_end(consumer_b),
            resources=frozenset({"rail1"}),
            symmetric_buffer=slot,
        )
    )
    graph = builder.build()
    telemetry = TelemetryStore()
    telemetry.add_iteration(
        [
            OperationSample(op_id=consumer_a, iteration=0, start_us=1000.0, end_us=1500.0),
            OperationSample(op_id=consumer_b, iteration=0, start_us=3000.0, end_us=3500.0),
            OperationSample(op_id=ag_a, iteration=0, start_us=0.0, end_us=500.0),
            OperationSample(op_id=ag_b, iteration=0, start_us=1500.0, end_us=2000.0),
        ],
        graph,
    )

    plan = RuntimeCommunicationPlanner().compile(graph, telemetry, epoch=0)
    by_id = {action.op_id: action for action in plan.actions}
    assert by_id[ag_a].buffer_release_us == 1500.0
    assert by_id[ag_b].planned_start_us >= by_id[ag_a].buffer_release_us


def test_executor_fires_each_action_once():
    graph, telemetry, _ = _example()
    plan = RuntimeCommunicationPlanner().compile(graph, telemetry, epoch=0)
    observed = []
    executor = RuntimePlanExecutor(plan, lambda action: observed.append(action.op_id))

    triggers = {
        trigger
        for action in plan.actions
        for trigger in (action.issue_trigger, *action.wait_for, Trigger.op_end(action.op_id))
    }
    for trigger in triggers:
        executor.fire(trigger)
        executor.fire(trigger)

    assert len(observed) == len(plan.actions)
    assert set(observed) == {action.op_id for action in plan.actions}
    assert executor.pending == frozenset()


def test_executor_holds_action_until_actual_resource_predecessor_finishes():
    graph, telemetry, ids = _example()
    plan = RuntimeCommunicationPlanner().compile(graph, telemetry, epoch=0)
    observed = []
    executor = RuntimePlanExecutor(plan, lambda action: observed.append(action.op_id))
    ag = next(action for action in plan.actions if action.op_id == ids["ag"])

    executor.fire(ag.issue_trigger)
    executor.fire(Trigger.window_start(Phase.FORWARD))
    assert ids["ag"] not in observed

    for guard in ag.wait_for:
        executor.fire(guard)
    assert ids["ag"] in observed
