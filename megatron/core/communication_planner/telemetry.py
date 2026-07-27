# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Runtime telemetry for semantic communication planning.

CUDA events are recorded on the streams that execute the logical operations.
Completed iterations are collected later with ``query()``; the recorder never
introduces a host synchronization on the training critical path.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import Enum

from .graph import OperationGraph, OperationKind, SemanticOpId, Trigger, TriggerKind


class MissingTelemetryError(RuntimeError):
    """Raised when a plan cannot be compiled from the available samples."""


class TimelineMarker(str, Enum):
    """Device-timeline markers collected for one logical operation."""

    READY = "ready"
    START = "start"
    END = "end"
    CONSUMER_READY = "consumer_ready"
    CONSUMER_RESUME = "consumer_resume"
    RELEASE = "release"


@dataclass(frozen=True)
class OperationSample:
    """One device-timeline observation for a logical operation.

    All timestamps are microseconds relative to the beginning of one iteration.
    ``start_us`` and ``end_us`` describe device service time, while optional
    markers separate queueing, exposed consumer wait, and buffer lifetime.
    """

    op_id: SemanticOpId
    iteration: int
    start_us: float
    end_us: float
    ready_us: float | None = None
    consumer_ready_us: float | None = None
    consumer_resume_us: float | None = None
    release_us: float | None = None
    overlap_kinds: frozenset[OperationKind] = frozenset()

    def __post_init__(self) -> None:
        values = (
            self.start_us,
            self.end_us,
            self.ready_us,
            self.consumer_ready_us,
            self.consumer_resume_us,
            self.release_us,
        )
        if any(value is not None and (not math.isfinite(value) or value < 0) for value in values):
            raise ValueError("OperationSample timestamps must be finite and non-negative")
        if self.iteration < 0:
            raise ValueError("OperationSample.iteration must be non-negative")
        if self.end_us < self.start_us:
            raise ValueError("OperationSample.end_us must not precede start_us")
        if self.ready_us is not None and self.start_us < self.ready_us:
            raise ValueError("OperationSample.start_us must not precede ready_us")
        if (
            self.consumer_ready_us is not None
            and self.consumer_resume_us is not None
            and self.consumer_resume_us < self.consumer_ready_us
        ):
            raise ValueError("consumer_resume_us must not precede consumer_ready_us")
        if self.release_us is not None and self.release_us < self.start_us:
            raise ValueError("release_us must not precede start_us")

    @property
    def service_us(self) -> float:
        """Device service time."""

        return self.end_us - self.start_us

    @property
    def queue_delay_us(self) -> float | None:
        """Delay from data readiness to device execution."""

        return self.start_us - self.ready_us if self.ready_us is not None else None

    @property
    def exposed_wait_us(self) -> float | None:
        """Consumer-stream time exposed by waiting for the operation."""

        if self.consumer_ready_us is None or self.consumer_resume_us is None:
            return None
        return self.consumer_resume_us - self.consumer_ready_us

    @property
    def deadline_slack_us(self) -> float | None:
        """Time between operation completion and the consumer becoming ready."""

        if self.consumer_ready_us is None:
            return None
        return self.consumer_ready_us - self.end_us

    @property
    def lifetime_us(self) -> float | None:
        """Lifetime from issue until the output/send buffer may be reused."""

        return self.release_us - self.start_us if self.release_us is not None else None


@dataclass(frozen=True)
class OperationStatistics:
    """Compact statistics for one operation and sample selection."""

    count: int
    p50_us: float
    p95_us: float
    max_us: float


def _quantile(values: Iterable[float], percentile: float) -> float:
    data = sorted(values)
    if not data:
        raise MissingTelemetryError("No samples are available")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be in [0, 1]")
    if len(data) == 1:
        return data[0]
    position = percentile * (len(data) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return data[lower]
    fraction = position - lower
    return data[lower] * (1.0 - fraction) + data[upper] * fraction


class TelemetryStore:
    """Bounded operation-sample store used by the runtime planner."""

    def __init__(self, max_samples_per_operation: int = 64) -> None:
        if max_samples_per_operation <= 0:
            raise ValueError("max_samples_per_operation must be positive")
        self._max_samples = max_samples_per_operation
        self._samples: dict[SemanticOpId, deque[OperationSample]] = defaultdict(
            lambda: deque(maxlen=self._max_samples)
        )
        self._iteration_ends: deque[float] = deque(maxlen=self._max_samples)

    def add_iteration(
        self, samples: Iterable[OperationSample], graph: OperationGraph | None = None
    ) -> tuple[OperationSample, ...]:
        """Store one iteration and annotate naturally overlapping comm classes.

        Args:
            samples: Completed samples from a single iteration.
            graph: Optional semantic graph. When supplied, communication
                intervals are annotated with the kinds that overlapped them.

        Returns:
            The samples actually stored, including overlap annotations.
        """

        items = list(samples)
        if not items:
            return ()
        iterations = {sample.iteration for sample in items}
        if len(iterations) != 1:
            raise ValueError("add_iteration expects samples from exactly one iteration")
        if len({sample.op_id for sample in items}) != len(items):
            raise ValueError("An iteration may contain at most one sample per operation ID")

        if graph is not None:
            unknown = [sample.op_id for sample in items if sample.op_id not in graph.operations]
            if unknown:
                raise ValueError(f"Telemetry contains operations absent from the graph: {unknown}")
            items = self._annotate_overlaps(items, graph)

        for sample in items:
            self._samples[sample.op_id].append(sample)
        self._iteration_ends.append(
            max(
                max(
                    value
                    for value in (sample.end_us, sample.consumer_resume_us, sample.release_us)
                    if value is not None
                )
                for sample in items
            )
        )
        return tuple(items)

    def samples(self, op_id: SemanticOpId) -> tuple[OperationSample, ...]:
        """Return retained samples for one operation."""

        return tuple(self._samples.get(op_id, ()))

    def estimate_duration(
        self,
        op_id: SemanticOpId,
        percentile: float = 0.95,
        overlap_kinds: frozenset[OperationKind] | None = None,
    ) -> float:
        """Estimate service time, optionally for an exact overlap context."""

        samples = self.samples(op_id)
        if overlap_kinds is not None:
            samples = tuple(sample for sample in samples if sample.overlap_kinds == overlap_kinds)
        if not samples:
            context = f" with overlap {overlap_kinds}" if overlap_kinds is not None else ""
            raise MissingTelemetryError(f"No duration samples for {op_id}{context}")
        return _quantile((sample.service_us for sample in samples), percentile)

    def estimate_trigger(self, trigger: Trigger, percentile: float = 0.5) -> float:
        """Estimate when a semantic trigger fires relative to window start."""

        if trigger.kind is TriggerKind.WINDOW_START:
            return 0.0
        assert trigger.op_id is not None
        samples = self.samples(trigger.op_id)
        if not samples:
            raise MissingTelemetryError(f"No trigger samples for {trigger.op_id}")
        values = (
            (sample.start_us for sample in samples)
            if trigger.kind is TriggerKind.OP_START
            else (sample.end_us for sample in samples)
        )
        return _quantile(values, percentile)

    def estimate_iteration_end(self, percentile: float = 0.95) -> float:
        """Estimate the logical iteration end for soft communication deadlines."""

        return _quantile(self._iteration_ends, percentile)

    def statistics(self, op_id: SemanticOpId) -> OperationStatistics:
        """Return service-time summary statistics for an operation."""

        values = [sample.service_us for sample in self.samples(op_id)]
        if not values:
            raise MissingTelemetryError(f"No duration samples for {op_id}")
        return OperationStatistics(
            count=len(values),
            p50_us=_quantile(values, 0.50),
            p95_us=_quantile(values, 0.95),
            max_us=max(values),
        )

    @staticmethod
    def _annotate_overlaps(
        samples: list[OperationSample], graph: OperationGraph
    ) -> list[OperationSample]:
        overlaps: dict[SemanticOpId, set[OperationKind]] = defaultdict(set)
        active: list[OperationSample] = []
        for sample in sorted(samples, key=lambda item: (item.start_us, item.end_us)):
            active = [other for other in active if other.end_us > sample.start_us]
            sample_kind = graph.operations[sample.op_id].kind
            if sample_kind.is_communication:
                for other in active:
                    other_kind = graph.operations[other.op_id].kind
                    if not other_kind.is_communication:
                        continue
                    overlaps[sample.op_id].add(other_kind)
                    overlaps[other.op_id].add(sample_kind)
            active.append(sample)
        return [
            replace(sample, overlap_kinds=frozenset(overlaps[sample.op_id])) for sample in samples
        ]


@dataclass
class _CudaIteration:
    iteration: int
    origin: object
    end: object | None
    markers: dict[SemanticOpId, dict[TimelineMarker, object]]


class CudaEventRecorder:
    """Non-synchronizing CUDA-event recorder for semantic operation timelines.

    ``begin_iteration`` records a common timing origin. Callers must begin
    recording before work from the observed iteration reaches auxiliary
    streams. ``collect_completed`` checks event readiness and leaves incomplete
    iterations pending; it never calls ``synchronize``.

    Args:
        event_factory: Optional zero-argument event factory used by CPU tests.
            Production defaults to timing-enabled ``torch.cuda.Event``.
    """

    def __init__(self, event_factory: Callable[[], object] | None = None) -> None:
        if event_factory is None:
            import torch

            event_factory = lambda: torch.cuda.Event(enable_timing=True)
        self._event_factory = event_factory
        self._active: _CudaIteration | None = None
        self._pending: deque[_CudaIteration] = deque()

    def begin_iteration(self, iteration: int, stream: object | None = None) -> None:
        """Begin one observed iteration and record its common timing origin."""

        if iteration < 0:
            raise ValueError("iteration must be non-negative")
        if self._active is not None:
            raise RuntimeError("An observed iteration is already active")
        origin = self._new_recorded_event(stream)
        self._active = _CudaIteration(
            iteration=iteration, origin=origin, end=None, markers=defaultdict(dict)
        )

    def record(
        self, op_id: SemanticOpId, marker: TimelineMarker, stream: object | None = None
    ) -> None:
        """Record one semantic marker on its actual execution stream."""

        if self._active is None:
            raise RuntimeError("begin_iteration must be called before record")
        markers = self._active.markers[op_id]
        if marker in markers:
            raise ValueError(f"Duplicate {marker.value} marker for {op_id}")
        markers[marker] = self._new_recorded_event(stream)

    def end_iteration(self, stream: object | None = None) -> None:
        """Close the active iteration and queue it for asynchronous collection."""

        if self._active is None:
            raise RuntimeError("No observed iteration is active")
        self._active.end = self._new_recorded_event(stream)
        self._pending.append(self._active)
        self._active = None

    def collect_completed(self) -> tuple[tuple[int, tuple[OperationSample, ...]], ...]:
        """Collect ready iterations without blocking the host."""

        completed = []
        remaining: deque[_CudaIteration] = deque()
        while self._pending:
            iteration = self._pending.popleft()
            events = [iteration.origin, iteration.end]
            events.extend(
                event for markers in iteration.markers.values() for event in markers.values()
            )
            if any(event is None or not event.query() for event in events):
                remaining.append(iteration)
                continue
            completed.append((iteration.iteration, self._samples_from_events(iteration)))
        self._pending = remaining
        return tuple(completed)

    @property
    def pending_iterations(self) -> int:
        """Number of closed iterations waiting for device completion."""

        return len(self._pending)

    def _new_recorded_event(self, stream: object | None) -> object:
        event = self._event_factory()
        if stream is None:
            event.record()
        else:
            event.record(stream)
        return event

    @staticmethod
    def _samples_from_events(iteration: _CudaIteration) -> tuple[OperationSample, ...]:
        def timestamp(event: object | None) -> float | None:
            if event is None:
                return None
            return float(iteration.origin.elapsed_time(event)) * 1000.0

        samples = []
        for op_id, markers in iteration.markers.items():
            if TimelineMarker.START not in markers or TimelineMarker.END not in markers:
                raise RuntimeError(f"Operation {op_id} is missing START or END timing markers")
            samples.append(
                OperationSample(
                    op_id=op_id,
                    iteration=iteration.iteration,
                    ready_us=timestamp(markers.get(TimelineMarker.READY)),
                    start_us=timestamp(markers[TimelineMarker.START]),
                    end_us=timestamp(markers[TimelineMarker.END]),
                    consumer_ready_us=timestamp(markers.get(TimelineMarker.CONSUMER_READY)),
                    consumer_resume_us=timestamp(markers.get(TimelineMarker.CONSUMER_RESUME)),
                    release_us=timestamp(markers.get(TimelineMarker.RELEASE)),
                )
            )
        return tuple(sorted(samples, key=lambda sample: sample.op_id.stable_key))
