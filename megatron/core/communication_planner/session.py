# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Lifecycle facade for eager graph discovery, profiling, and plan compilation."""

from __future__ import annotations

from .graph import (
    DependencyKind,
    OperationGraph,
    OperationGraphBuilder,
    OperationSpec,
    SemanticOpId,
)
from .planner import RuntimeCommunicationPlanner, RuntimePlan
from .telemetry import CudaEventRecorder, OperationSample, TelemetryStore, TimelineMarker


class RuntimePlanningSession:
    """Coordinate the first runtime-planning lifecycle.

    Operation registration is open during eager discovery and becomes immutable
    after :meth:`freeze_graph`. CUDA samples may then be collected into the
    bounded telemetry store and compiled into successive plan epochs.

    Args:
        recorder: CUDA-event recorder or a test implementation.
        telemetry: Optional bounded telemetry store.
        planner: Optional plan compiler.
    """

    def __init__(
        self,
        recorder: CudaEventRecorder,
        telemetry: TelemetryStore | None = None,
        planner: RuntimeCommunicationPlanner | None = None,
    ) -> None:
        self._builder = OperationGraphBuilder()
        self._graph: OperationGraph | None = None
        self.recorder = recorder
        self.telemetry = telemetry or TelemetryStore()
        self.planner = planner or RuntimeCommunicationPlanner()

    def register_operation(self, operation: OperationSpec) -> None:
        """Register one semantic operation during eager discovery."""

        self._require_mutable_graph()
        self._builder.add_operation(operation)

    def add_dependency(
        self, src: SemanticOpId, dst: SemanticOpId, kind: DependencyKind = DependencyKind.DATA
    ) -> None:
        """Register one hard dependency during eager discovery."""

        self._require_mutable_graph()
        self._builder.add_dependency(src, dst, kind)

    def freeze_graph(self) -> OperationGraph:
        """Validate and freeze the graph used by all subsequent plan epochs."""

        if self._graph is None:
            self._graph = self._builder.build()
        return self._graph

    @property
    def graph(self) -> OperationGraph:
        """Return the frozen graph, raising if discovery is incomplete."""

        if self._graph is None:
            raise RuntimeError("freeze_graph must be called before accessing the graph")
        return self._graph

    def begin_iteration(self, iteration: int, stream: object | None = None) -> None:
        """Begin timing one eager iteration."""

        self.recorder.begin_iteration(iteration, stream)

    def record(
        self, op_id: SemanticOpId, marker: TimelineMarker, stream: object | None = None
    ) -> None:
        """Record one device marker for a registered logical operation."""

        self.recorder.record(op_id, marker, stream)

    def alias_marker(
        self,
        op_id: SemanticOpId,
        marker: TimelineMarker,
        source_op_id: SemanticOpId,
        source_marker: TimelineMarker,
    ) -> None:
        """Reuse an earlier operation marker as a semantic trigger timestamp."""

        self.recorder.alias_marker(op_id, marker, source_op_id, source_marker)

    def alias_origin(self, op_id: SemanticOpId, marker: TimelineMarker) -> None:
        """Use the iteration origin as a semantic trigger timestamp."""

        self.recorder.alias_origin(op_id, marker)

    def end_iteration(self, stream: object | None = None) -> None:
        """Close the current eager iteration."""

        self.recorder.end_iteration(stream)

    def collect_completed(self) -> tuple[tuple[int, tuple[OperationSample, ...]], ...]:
        """Collect ready CUDA timelines and add them to the telemetry store."""

        graph = self.graph
        completed = self.recorder.collect_completed()
        for _, samples in completed:
            self.telemetry.add_iteration(samples, graph)
        return completed

    def compile(self, epoch: int) -> RuntimePlan:
        """Compile a new plan epoch from the retained runtime telemetry."""

        return self.planner.compile(self.graph, self.telemetry, epoch=epoch)

    def _require_mutable_graph(self) -> None:
        if self._graph is not None:
            raise RuntimeError("The semantic graph is frozen for this runtime signature")
