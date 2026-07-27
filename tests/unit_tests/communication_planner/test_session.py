# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.communication_planner import (
    CudaEventRecorder,
    OperationKind,
    OperationSpec,
    Phase,
    RuntimePlanningSession,
    SemanticOpId,
    TimelineMarker,
    Trigger,
)


class _FakeStream:
    def __init__(self, time_ms):
        self.time_ms = time_ms


class _FakeEvent:
    def __init__(self):
        self.time_ms = None

    def record(self, stream=None):
        self.time_ms = 0.0 if stream is None else stream.time_ms

    def query(self):
        return True

    def elapsed_time(self, other):
        return other.time_ms - self.time_ms


def test_session_runs_discovery_profile_and_compile_lifecycle():
    consumer = SemanticOpId(scope="L10", phase=Phase.FORWARD, role="consumer")
    ag = SemanticOpId(scope="L10", phase=Phase.FORWARD, role="ag")
    session = RuntimePlanningSession(CudaEventRecorder(_FakeEvent))
    session.register_operation(OperationSpec(op_id=consumer, kind=OperationKind.COMPUTE))
    session.register_operation(
        OperationSpec(
            op_id=ag,
            kind=OperationKind.GTP_DENSE_AG,
            ready_trigger=Trigger.window_start(Phase.FORWARD),
            deadline_trigger=Trigger.op_start(consumer),
            resources=frozenset({"fabric"}),
        )
    )
    session.freeze_graph()

    with pytest.raises(RuntimeError, match="graph is frozen"):
        session.register_operation(
            OperationSpec(
                op_id=SemanticOpId(scope="L11", phase=Phase.FORWARD, role="late"),
                kind=OperationKind.COMPUTE,
            )
        )

    session.begin_iteration(0, _FakeStream(0.0))
    session.record(ag, TimelineMarker.START, _FakeStream(0.0))
    session.record(ag, TimelineMarker.END, _FakeStream(1.0))
    session.record(consumer, TimelineMarker.START, _FakeStream(2.0))
    session.record(consumer, TimelineMarker.END, _FakeStream(3.0))
    session.end_iteration(_FakeStream(3.0))

    completed = session.collect_completed()
    assert len(completed) == 1
    plan = session.compile(epoch=1)
    assert plan.epoch == 1
    assert plan.diagnostics.is_feasible
