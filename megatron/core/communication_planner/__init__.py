# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Runtime communication graph, profiling, and planning primitives.

The package is intentionally independent from GTP launch code. The first
version supports shadow profiling and deterministic plan compilation; a later
integration can execute the plan from GTP and EP semantic hook sites.
"""

from .graph import (
    Dependency,
    DependencyKind,
    OperationGraph,
    OperationGraphBuilder,
    OperationKind,
    OperationSpec,
    Phase,
    SemanticOpId,
    SymmetricBufferSpec,
    Trigger,
    TriggerKind,
)
from .planner import (
    DeadlineMiss,
    PlanDiagnostics,
    PlannerConfig,
    RuntimeCommunicationPlanner,
    RuntimePlan,
    RuntimePlanExecutor,
    ScheduledAction,
)
from .session import RuntimePlanningSession
from .telemetry import (
    CudaEventRecorder,
    MissingTelemetryError,
    OperationSample,
    OperationStatistics,
    TelemetryStore,
    TimelineMarker,
)

__all__ = [
    "Dependency",
    "DependencyKind",
    "OperationGraph",
    "OperationGraphBuilder",
    "OperationKind",
    "OperationSpec",
    "Phase",
    "SemanticOpId",
    "SymmetricBufferSpec",
    "Trigger",
    "TriggerKind",
    "CudaEventRecorder",
    "MissingTelemetryError",
    "OperationSample",
    "OperationStatistics",
    "TelemetryStore",
    "TimelineMarker",
    "DeadlineMiss",
    "PlanDiagnostics",
    "PlannerConfig",
    "RuntimeCommunicationPlanner",
    "RuntimePlan",
    "RuntimePlanExecutor",
    "ScheduledAction",
    "RuntimePlanningSession",
]
