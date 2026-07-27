# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.communication_planner import (
    CudaEventRecorder,
    OperationGraphBuilder,
    OperationKind,
    OperationSample,
    OperationSpec,
    Phase,
    SemanticOpId,
    TelemetryStore,
    TimelineMarker,
    Trigger,
)


def _id(scope, role):
    return SemanticOpId(scope=scope, phase=Phase.FORWARD, role=role)


def test_operation_sample_separates_service_queue_wait_and_lifetime():
    sample = OperationSample(
        op_id=_id("L10", "ag"),
        iteration=3,
        ready_us=100.0,
        start_us=120.0,
        end_us=220.0,
        consumer_ready_us=200.0,
        consumer_resume_us=220.0,
        release_us=300.0,
    )

    assert sample.service_us == 100.0
    assert sample.queue_delay_us == 20.0
    assert sample.exposed_wait_us == 20.0
    assert sample.deadline_slack_us == -20.0
    assert sample.lifetime_us == 180.0


def test_store_estimates_triggers_durations_and_overlap_context():
    ep = _id("L9", "ep_combine")
    ag = _id("L10", "ag")
    graph = (
        OperationGraphBuilder()
        .add_operation(
            OperationSpec(
                op_id=ep,
                kind=OperationKind.EP_COMBINE,
                ready_trigger=Trigger.window_start(Phase.FORWARD),
            )
        )
        .add_operation(
            OperationSpec(
                op_id=ag,
                kind=OperationKind.GTP_DENSE_AG,
                ready_trigger=Trigger.window_start(Phase.FORWARD),
            )
        )
        .build()
    )
    store = TelemetryStore()
    stored = store.add_iteration(
        [
            OperationSample(op_id=ep, iteration=0, start_us=0.0, end_us=1000.0),
            OperationSample(op_id=ag, iteration=0, start_us=200.0, end_us=1400.0),
        ],
        graph,
    )

    by_id = {sample.op_id: sample for sample in stored}
    assert by_id[ep].overlap_kinds == frozenset({OperationKind.GTP_DENSE_AG})
    assert by_id[ag].overlap_kinds == frozenset({OperationKind.EP_COMBINE})
    assert store.estimate_duration(ag) == 1200.0
    assert store.estimate_trigger(Trigger.op_start(ag)) == 200.0
    assert store.estimate_trigger(Trigger.op_end(ag)) == 1400.0
    assert store.estimate_iteration_end() == 1400.0


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


def test_cuda_recorder_collects_completed_iterations_without_sync():
    events = []

    def event_factory():
        event = _FakeEvent()
        events.append(event)
        return event

    op_id = _id("L10", "ag")
    recorder = CudaEventRecorder(event_factory)
    recorder.begin_iteration(4, _FakeStream(10.0))
    recorder.record(op_id, TimelineMarker.READY, _FakeStream(10.2))
    recorder.record(op_id, TimelineMarker.START, _FakeStream(10.4))
    recorder.record(op_id, TimelineMarker.END, _FakeStream(11.6))
    recorder.record(op_id, TimelineMarker.CONSUMER_READY, _FakeStream(11.5))
    recorder.record(op_id, TimelineMarker.CONSUMER_RESUME, _FakeStream(11.6))
    recorder.end_iteration(_FakeStream(12.0))

    events[-1].complete = False
    assert recorder.collect_completed() == ()
    assert recorder.pending_iterations == 1

    events[-1].complete = True
    completed = recorder.collect_completed()
    assert recorder.pending_iterations == 0
    assert len(completed) == 1
    iteration, samples = completed[0]
    assert iteration == 4
    assert samples[0].ready_us == pytest.approx(200.0)
    assert samples[0].start_us == pytest.approx(400.0)
    assert samples[0].end_us == pytest.approx(1600.0)
    assert samples[0].exposed_wait_us == pytest.approx(100.0)
