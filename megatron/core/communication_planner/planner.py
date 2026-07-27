# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deterministic first-version runtime communication planner.

The compiler uses profiled semantic trigger times and p95 service durations. It
first builds an earliest-deadline resource-feasible order, then compacts that
fixed order toward deadlines to reduce prefetch lifetime. Shared resources are
serialized conservatively; learned overlap costs can replace that policy later
without changing the graph or plan formats.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .graph import (
    OperationGraph,
    OperationKind,
    OperationSpec,
    SemanticOpId,
    SymmetricBufferSpec,
    Trigger,
)
from .telemetry import TelemetryStore


@dataclass(frozen=True)
class PlannerConfig:
    """Configuration for deterministic runtime plan compilation."""

    duration_percentile: float = 0.95
    trigger_percentile: float = 0.50
    iteration_end_percentile: float = 0.95
    deadline_guard_us: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("duration_percentile", self.duration_percentile),
            ("trigger_percentile", self.trigger_percentile),
            ("iteration_end_percentile", self.iteration_end_percentile),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.deadline_guard_us < 0:
            raise ValueError("deadline_guard_us must be non-negative")


@dataclass(frozen=True)
class DeadlineMiss:
    """Predicted communication deadline miss."""

    op_id: SemanticOpId
    deadline_us: float
    completion_us: float

    @property
    def lateness_us(self) -> float:
        """Predicted time exposed beyond the deadline."""

        return self.completion_us - self.deadline_us


@dataclass(frozen=True)
class PlanDiagnostics:
    """Planner feasibility and audit information."""

    deadline_misses: tuple[DeadlineMiss, ...]
    serialized_resources: tuple[str, ...]

    @property
    def is_feasible(self) -> bool:
        """Whether every profiled deadline is predicted to be met."""

        return not self.deadline_misses


@dataclass(frozen=True)
class ScheduledAction:
    """One communication issue action in the compiled semantic plan."""

    op_id: SemanticOpId
    kind: OperationKind
    issue_trigger: Trigger
    wait_for: tuple[Trigger, ...]
    planned_start_us: float
    planned_end_us: float
    ready_us: float
    deadline_us: float
    resources: frozenset[str]
    communicator_id: str | None
    sequence: int | None
    symmetric_buffer: SymmetricBufferSpec | None
    buffer_release_us: float
    priority: int

    @property
    def predicted_slack_us(self) -> float:
        """Predicted deadline slack after the communication completes."""

        return self.deadline_us - self.planned_end_us


class RuntimePlan:
    """Immutable event-triggered communication plan."""

    def __init__(
        self,
        *,
        epoch: int,
        graph_fingerprint: str,
        actions: Iterable[ScheduledAction],
        diagnostics: PlanDiagnostics,
    ) -> None:
        if epoch < 0:
            raise ValueError("Plan epoch must be non-negative")
        self.epoch = epoch
        self.graph_fingerprint = graph_fingerprint
        self.actions = tuple(
            sorted(
                actions,
                key=lambda action: (
                    action.planned_start_us,
                    action.priority,
                    action.op_id.stable_key,
                ),
            )
        )
        grouped: dict[Trigger, list[ScheduledAction]] = defaultdict(list)
        for action in self.actions:
            grouped[action.issue_trigger].append(action)
        self._actions_by_trigger: Mapping[Trigger, tuple[ScheduledAction, ...]] = MappingProxyType(
            {
                trigger: tuple(
                    sorted(
                        trigger_actions,
                        key=lambda action: (
                            action.planned_start_us,
                            action.priority,
                            action.op_id.stable_key,
                        ),
                    )
                )
                for trigger, trigger_actions in grouped.items()
            }
        )
        self.diagnostics = diagnostics

    def actions_for(self, trigger: Trigger) -> tuple[ScheduledAction, ...]:
        """Return ordered actions issued when ``trigger`` fires."""

        return self._actions_by_trigger.get(trigger, ())

    @property
    def fingerprint(self) -> str:
        """Content hash for rank-consensus checks."""

        payload = {
            "epoch": self.epoch,
            "graph": self.graph_fingerprint,
            "actions": [
                {
                    "op": action.op_id.stable_key,
                    "kind": action.kind.value,
                    "trigger": action.issue_trigger.stable_key,
                    "wait_for": [trigger.stable_key for trigger in action.wait_for],
                    "start_us": round(action.planned_start_us, 6),
                    "end_us": round(action.planned_end_us, 6),
                    "deadline_us": round(action.deadline_us, 6),
                    "resources": sorted(action.resources),
                    "communicator": action.communicator_id,
                    "sequence": action.sequence,
                    "buffer_release_us": round(action.buffer_release_us, 6),
                }
                for action in self.actions
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Reservation:
    start_us: float
    end_us: float
    op_id: SemanticOpId


class _ResourceCalendars:
    def __init__(self) -> None:
        self._calendars: dict[str, list[_Reservation]] = defaultdict(list)

    def find_earliest(
        self,
        *,
        ready_us: float,
        duration_us: float,
        execution_resources: frozenset[str],
        buffer_resource: str | None,
        release_us: float,
    ) -> float:
        """Find the first interval free on every required resource."""

        candidate = ready_us
        while True:
            blocked_until = candidate
            execution_end = candidate + duration_us
            for resource in execution_resources:
                blocked_until = max(
                    blocked_until, self._first_conflict_end(resource, candidate, execution_end)
                )
            if buffer_resource is not None:
                buffer_end = max(execution_end, release_us)
                blocked_until = max(
                    blocked_until, self._first_conflict_end(buffer_resource, candidate, buffer_end)
                )
            if blocked_until <= candidate:
                return candidate
            candidate = blocked_until

    def reserve(
        self,
        *,
        op_id: SemanticOpId,
        start_us: float,
        end_us: float,
        execution_resources: frozenset[str],
        buffer_resource: str | None,
        buffer_release_us: float,
    ) -> None:
        """Reserve execution and optional buffer-lifetime intervals."""

        for resource in execution_resources:
            self._insert(resource, _Reservation(start_us, end_us, op_id))
        if buffer_resource is not None:
            self._insert(
                buffer_resource, _Reservation(start_us, max(end_us, buffer_release_us), op_id)
            )

    @property
    def calendars(self) -> Mapping[str, tuple[_Reservation, ...]]:
        """Return immutable views of the current resource reservations."""

        return {resource: tuple(reservations) for resource, reservations in self._calendars.items()}

    def _first_conflict_end(self, resource: str, start_us: float, end_us: float) -> float:
        for reservation in self._calendars[resource]:
            if reservation.end_us <= start_us:
                continue
            if reservation.start_us >= end_us:
                break
            return reservation.end_us
        return start_us

    def _insert(self, resource: str, reservation: _Reservation) -> None:
        calendar = self._calendars[resource]
        calendar.append(reservation)
        calendar.sort(key=lambda item: (item.start_us, item.end_us, item.op_id.stable_key))


@dataclass(frozen=True)
class _PlanningJob:
    spec: OperationSpec
    duration_us: float
    ready_us: float
    deadline_us: float
    release_us: float
    execution_resources: frozenset[str]
    buffer_resource: str | None


class RuntimeCommunicationPlanner:
    """Compile profiled semantic operations into an event-triggered plan."""

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()

    def compile(
        self, graph: OperationGraph, telemetry: TelemetryStore, *, epoch: int
    ) -> RuntimePlan:
        """Compile a deterministic resource-aware communication plan.

        The first version treats every shared resource as exclusive. This is a
        conservative admission policy for GTP/EP interference; future planners
        may admit selected overlaps using the same operation and plan schemas.
        """

        jobs = self._build_jobs(graph, telemetry)
        forward_start, calendars = self._forward_schedule(graph, jobs)
        compacted_start = self._compact_toward_deadlines(graph, jobs, forward_start, calendars)
        wait_for = self._build_wait_guards(graph, jobs, calendars)
        actions = self._build_actions(graph, telemetry, jobs, compacted_start, wait_for)
        misses = tuple(
            DeadlineMiss(
                op_id=action.op_id,
                deadline_us=action.deadline_us,
                completion_us=action.planned_end_us,
            )
            for action in actions
            if action.planned_end_us > action.deadline_us
        )
        serialized_resources = sorted(
            {resource for job in jobs.values() for resource in job.execution_resources}
        )
        return RuntimePlan(
            epoch=epoch,
            graph_fingerprint=graph.fingerprint,
            actions=actions,
            diagnostics=PlanDiagnostics(
                deadline_misses=misses, serialized_resources=tuple(serialized_resources)
            ),
        )

    def _build_jobs(
        self, graph: OperationGraph, telemetry: TelemetryStore
    ) -> dict[SemanticOpId, _PlanningJob]:
        iteration_end = telemetry.estimate_iteration_end(self.config.iteration_end_percentile)
        jobs = {}
        for op_id, spec in graph.operations.items():
            if not spec.kind.is_communication:
                continue
            assert spec.ready_trigger is not None
            ready_us = telemetry.estimate_trigger(
                spec.ready_trigger, self.config.trigger_percentile
            )
            deadline_us = (
                telemetry.estimate_trigger(spec.deadline_trigger, self.config.trigger_percentile)
                - self.config.deadline_guard_us
                if spec.deadline_trigger is not None
                else iteration_end
            )
            release_us = (
                telemetry.estimate_trigger(spec.release_trigger, self.config.trigger_percentile)
                if spec.release_trigger is not None
                else deadline_us
            )
            execution_resources = set(spec.resources)
            if spec.communicator_id is not None:
                execution_resources.add(f"communicator:{spec.communicator_id}")
            jobs[op_id] = _PlanningJob(
                spec=spec,
                duration_us=telemetry.estimate_duration(op_id, self.config.duration_percentile),
                ready_us=ready_us,
                deadline_us=max(0.0, deadline_us),
                release_us=max(ready_us, release_us),
                execution_resources=frozenset(execution_resources),
                buffer_resource=(
                    spec.symmetric_buffer.resource_key
                    if spec.symmetric_buffer is not None
                    else None
                ),
            )
        return jobs

    @staticmethod
    def _communication_dependencies(
        graph: OperationGraph, jobs: Mapping[SemanticOpId, _PlanningJob]
    ) -> tuple[dict[SemanticOpId, set[SemanticOpId]], dict[SemanticOpId, set[SemanticOpId]]]:
        predecessors: dict[SemanticOpId, set[SemanticOpId]] = {op_id: set() for op_id in jobs}
        successors: dict[SemanticOpId, set[SemanticOpId]] = {op_id: set() for op_id in jobs}
        for edge in graph.dependencies:
            if edge.src in jobs and edge.dst in jobs:
                predecessors[edge.dst].add(edge.src)
                successors[edge.src].add(edge.dst)
        return predecessors, successors

    def _forward_schedule(
        self, graph: OperationGraph, jobs: Mapping[SemanticOpId, _PlanningJob]
    ) -> tuple[dict[SemanticOpId, float], _ResourceCalendars]:
        predecessors, successors = self._communication_dependencies(graph, jobs)
        indegree = {op_id: len(preds) for op_id, preds in predecessors.items()}
        ready = []
        for op_id, degree in indegree.items():
            if degree == 0:
                job = jobs[op_id]
                heapq.heappush(ready, (job.deadline_us, job.spec.priority, op_id.stable_key, op_id))

        starts: dict[SemanticOpId, float] = {}
        ends: dict[SemanticOpId, float] = {}
        calendars = _ResourceCalendars()
        while ready:
            _, _, _, op_id = heapq.heappop(ready)
            job = jobs[op_id]
            dependency_ready = max(
                (ends[predecessor] for predecessor in predecessors[op_id]), default=0.0
            )
            earliest = max(job.ready_us, dependency_ready)
            start_us = calendars.find_earliest(
                ready_us=earliest,
                duration_us=job.duration_us,
                execution_resources=job.execution_resources,
                buffer_resource=job.buffer_resource,
                release_us=job.release_us,
            )
            end_us = start_us + job.duration_us
            buffer_release_us = max(end_us, job.release_us)
            calendars.reserve(
                op_id=op_id,
                start_us=start_us,
                end_us=end_us,
                execution_resources=job.execution_resources,
                buffer_resource=job.buffer_resource,
                buffer_release_us=buffer_release_us,
            )
            starts[op_id] = start_us
            ends[op_id] = end_us
            for successor in successors[op_id]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    successor_job = jobs[successor]
                    heapq.heappush(
                        ready,
                        (
                            successor_job.deadline_us,
                            successor_job.spec.priority,
                            successor.stable_key,
                            successor,
                        ),
                    )
        if len(starts) != len(jobs):
            raise ValueError("Communication dependency subgraph contains a cycle")
        return starts, calendars

    def _compact_toward_deadlines(
        self,
        graph: OperationGraph,
        jobs: Mapping[SemanticOpId, _PlanningJob],
        forward_start: Mapping[SemanticOpId, float],
        calendars: _ResourceCalendars,
    ) -> dict[SemanticOpId, float]:
        _, successors = self._communication_dependencies(graph, jobs)

        # Preserve the resource order selected by the forward list scheduler.
        for resource, reservations in calendars.calendars.items():
            if resource.startswith("symmetric_buffer:"):
                continue
            ordered = [reservation.op_id for reservation in reservations]
            for previous, current in zip(ordered, ordered[1:]):
                successors[previous].add(current)

        ordered = sorted(jobs, key=lambda op_id: (forward_start[op_id], op_id.stable_key))
        starts = dict(forward_start)
        for op_id in reversed(ordered):
            job = jobs[op_id]
            latest_end = job.deadline_us
            if successors[op_id]:
                latest_end = min(
                    latest_end, min(starts[successor] for successor in successors[op_id])
                )
            candidate = latest_end - job.duration_us
            starts[op_id] = max(forward_start[op_id], candidate)
        return starts

    def _build_actions(
        self,
        graph: OperationGraph,
        telemetry: TelemetryStore,
        jobs: Mapping[SemanticOpId, _PlanningJob],
        starts: Mapping[SemanticOpId, float],
        wait_for: Mapping[SemanticOpId, tuple[Trigger, ...]],
    ) -> tuple[ScheduledAction, ...]:
        trigger_times = self._trigger_times(graph, telemetry, jobs, starts)
        actions = []
        for op_id, job in jobs.items():
            start_us = starts[op_id]
            end_us = start_us + job.duration_us
            issue_trigger = self._snap_to_trigger(
                op_id=op_id,
                ready_trigger=job.spec.ready_trigger,
                ready_us=job.ready_us,
                planned_start_us=start_us,
                trigger_times=trigger_times,
            )
            actions.append(
                ScheduledAction(
                    op_id=op_id,
                    kind=job.spec.kind,
                    issue_trigger=issue_trigger,
                    wait_for=wait_for[op_id],
                    planned_start_us=start_us,
                    planned_end_us=end_us,
                    ready_us=job.ready_us,
                    deadline_us=job.deadline_us,
                    resources=job.execution_resources,
                    communicator_id=job.spec.communicator_id,
                    sequence=job.spec.sequence,
                    symmetric_buffer=job.spec.symmetric_buffer,
                    buffer_release_us=max(end_us, job.release_us),
                    priority=job.spec.priority,
                )
            )
        return tuple(actions)

    def _build_wait_guards(
        self,
        graph: OperationGraph,
        jobs: Mapping[SemanticOpId, _PlanningJob],
        calendars: _ResourceCalendars,
    ) -> dict[SemanticOpId, tuple[Trigger, ...]]:
        """Build actual-event guards for every modeled ordering decision."""

        guards: dict[SemanticOpId, set[Trigger]] = {}
        for op_id, job in jobs.items():
            assert job.spec.ready_trigger is not None
            guards[op_id] = {job.spec.ready_trigger}
        predecessors, _ = self._communication_dependencies(graph, jobs)
        for op_id, operation_predecessors in predecessors.items():
            guards[op_id].update(
                Trigger.op_end(predecessor) for predecessor in operation_predecessors
            )

        for resource, reservations in calendars.calendars.items():
            for previous, current in zip(reservations, reservations[1:]):
                if resource.startswith("symmetric_buffer:"):
                    previous_release = jobs[previous.op_id].spec.release_trigger
                    guards[current.op_id].add(previous_release or Trigger.op_end(previous.op_id))
                else:
                    guards[current.op_id].add(Trigger.op_end(previous.op_id))

        return {
            op_id: tuple(sorted(op_guards, key=lambda trigger: trigger.stable_key))
            for op_id, op_guards in guards.items()
        }

    def _trigger_times(
        self,
        graph: OperationGraph,
        telemetry: TelemetryStore,
        jobs: Mapping[SemanticOpId, _PlanningJob],
        starts: Mapping[SemanticOpId, float],
    ) -> dict[Trigger, float]:
        times: dict[Trigger, float] = {}
        for op_id, spec in graph.operations.items():
            times[Trigger.window_start(op_id.phase, op_id.microbatch)] = 0.0
            start_trigger = Trigger.op_start(op_id)
            end_trigger = Trigger.op_end(op_id)
            if spec.kind.is_communication:
                times[start_trigger] = starts[op_id]
                times[end_trigger] = starts[op_id] + jobs[op_id].duration_us
            else:
                times[start_trigger] = telemetry.estimate_trigger(
                    start_trigger, self.config.trigger_percentile
                )
                times[end_trigger] = telemetry.estimate_trigger(
                    end_trigger, self.config.trigger_percentile
                )
        return times

    @staticmethod
    def _snap_to_trigger(
        *,
        op_id: SemanticOpId,
        ready_trigger: Trigger | None,
        ready_us: float,
        planned_start_us: float,
        trigger_times: Mapping[Trigger, float],
    ) -> Trigger:
        assert ready_trigger is not None
        candidates = [
            (time_us, trigger.stable_key, trigger)
            for trigger, time_us in trigger_times.items()
            if trigger.op_id != op_id
            and trigger.phase is op_id.phase
            and trigger.microbatch == op_id.microbatch
            and ready_us <= time_us <= planned_start_us
        ]
        ready_time = trigger_times.get(ready_trigger, ready_us)
        if ready_time <= planned_start_us:
            candidates.append((ready_time, ready_trigger.stable_key, ready_trigger))
        if not candidates:
            return ready_trigger
        return max(candidates, key=lambda item: (item[0], item[1]))[2]


class RuntimePlanExecutor:
    """Minimal trigger dispatcher for shadow and eager integrations."""

    def __init__(self, plan: RuntimePlan, issue: Callable[[ScheduledAction], None]) -> None:
        self._plan = plan
        self._issue = issue
        self._issued: set[SemanticOpId] = set()
        self._fired: set[Trigger] = set()

    def fire(self, trigger: Trigger) -> tuple[ScheduledAction, ...]:
        """Publish a trigger and issue every action whose guards are satisfied."""

        self._fired.add(trigger)
        issued = []
        for action in self._plan.actions:
            if action.op_id in self._issued:
                continue
            if action.issue_trigger not in self._fired:
                continue
            if any(guard not in self._fired for guard in action.wait_for):
                continue
            self._issue(action)
            self._issued.add(action.op_id)
            issued.append(action)
        return tuple(issued)

    @property
    def pending(self) -> frozenset[SemanticOpId]:
        """Operations in the plan that have not yet been issued."""

        return frozenset(
            action.op_id for action in self._plan.actions if action.op_id not in self._issued
        )
