# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Semantic operation graph for runtime communication planning.

The graph contains correctness constraints only. Resource conflicts are carried
by :class:`OperationSpec.resources` and resolved by the planner, so observing
one eager launch order does not accidentally make that order immutable.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class Phase(str, Enum):
    """Logical training phase."""

    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"


class OperationKind(str, Enum):
    """Logical operation classes understood by the first planner version."""

    COMPUTE = "compute"
    GTP_DENSE_AG = "gtp_dense_ag"
    GTP_DENSE_RS = "gtp_dense_rs"
    GTP_EXPERT_AG = "gtp_expert_ag"
    GTP_EXPERT_RS = "gtp_expert_rs"
    EP_DISPATCH = "ep_dispatch"
    EP_COMBINE = "ep_combine"
    DP_AG = "dp_ag"
    DP_RS = "dp_rs"

    @property
    def is_communication(self) -> bool:
        """Whether the operation is schedulable communication."""

        return self is not OperationKind.COMPUTE


class TriggerKind(str, Enum):
    """Semantic points at which a compiled plan may issue work."""

    WINDOW_START = "window_start"
    OP_START = "op_start"
    OP_END = "op_end"
    OP_CONSUMER_READY = "op_consumer_ready"


class DependencyKind(str, Enum):
    """Hard dependency classes retained in the semantic graph."""

    DATA = "data"
    COMMUNICATOR_ORDER = "communicator_order"
    BUFFER_REUSE = "buffer_reuse"
    CONTROL = "control"


@dataclass(frozen=True, order=True)
class SemanticOpId:
    """Rank-stable identity for one logical operation.

    IDs deliberately contain model semantics rather than tensor addresses or
    Python object IDs, allowing every participant to build the same plan.

    Args:
        scope: Stable model path, for example ``decoder.layers.10.attn.qkv``.
        phase: Logical training phase.
        role: Operation role, for example ``fwd_ag`` or ``wgrad``.
        microbatch: Microbatch index within the iteration template.
        occurrence: Repeated occurrence of the same role within the scope.
    """

    scope: str
    phase: Phase
    role: str
    microbatch: int = 0
    occurrence: int = 0

    def __post_init__(self) -> None:
        if not self.scope:
            raise ValueError("SemanticOpId.scope must be non-empty")
        if not self.role:
            raise ValueError("SemanticOpId.role must be non-empty")
        if self.microbatch < 0:
            raise ValueError("SemanticOpId.microbatch must be non-negative")
        if self.occurrence < 0:
            raise ValueError("SemanticOpId.occurrence must be non-negative")

    @property
    def stable_key(self) -> str:
        """Deterministic string used in plan artifacts and hashes."""

        return ":".join(
            (self.phase.value, f"mb{self.microbatch}", self.scope, self.role, str(self.occurrence))
        )

    def __str__(self) -> str:
        return self.stable_key


@dataclass(frozen=True)
class Trigger:
    """Semantic runtime trigger.

    Window-start triggers have no ``op_id``. Operation triggers fire at the
    start, end, or natural consumer-ready point of the referenced logical
    operation.
    """

    kind: TriggerKind
    phase: Phase
    microbatch: int
    op_id: SemanticOpId | None = None

    def __post_init__(self) -> None:
        if self.microbatch < 0:
            raise ValueError("Trigger.microbatch must be non-negative")
        if self.kind is TriggerKind.WINDOW_START:
            if self.op_id is not None:
                raise ValueError("WINDOW_START must not reference an operation")
            return
        if self.op_id is None:
            raise ValueError(f"{self.kind.value} must reference an operation")
        if self.op_id.phase is not self.phase or self.op_id.microbatch != self.microbatch:
            raise ValueError("Trigger phase/microbatch must match its operation")

    @classmethod
    def window_start(cls, phase: Phase, microbatch: int = 0) -> Trigger:
        """Create a phase-window start trigger."""

        return cls(kind=TriggerKind.WINDOW_START, phase=phase, microbatch=microbatch)

    @classmethod
    def op_start(cls, op_id: SemanticOpId) -> Trigger:
        """Create an operation-start trigger."""

        return cls(
            kind=TriggerKind.OP_START, phase=op_id.phase, microbatch=op_id.microbatch, op_id=op_id
        )

    @classmethod
    def op_end(cls, op_id: SemanticOpId) -> Trigger:
        """Create an operation-end trigger."""

        return cls(
            kind=TriggerKind.OP_END, phase=op_id.phase, microbatch=op_id.microbatch, op_id=op_id
        )

    @classmethod
    def consumer_ready(cls, op_id: SemanticOpId) -> Trigger:
        """Create a trigger for the point where an operation's consumer needs its output."""

        return cls(
            kind=TriggerKind.OP_CONSUMER_READY,
            phase=op_id.phase,
            microbatch=op_id.microbatch,
            op_id=op_id,
        )

    @property
    def stable_key(self) -> str:
        """Deterministic trigger key used in serialized plans."""

        if self.kind is TriggerKind.WINDOW_START:
            return f"{self.phase.value}:mb{self.microbatch}:window_start"
        assert self.op_id is not None
        return f"{self.op_id.stable_key}:{self.kind.value}"


@dataclass(frozen=True)
class SymmetricBufferSpec:
    """Stable symmetric-memory placement for a communication operation."""

    arena: str
    slot: int
    offset_bytes: int
    capacity_bytes: int
    generation: int = 0

    def __post_init__(self) -> None:
        if not self.arena:
            raise ValueError("SymmetricBufferSpec.arena must be non-empty")
        if min(self.slot, self.offset_bytes, self.capacity_bytes, self.generation) < 0:
            raise ValueError("Symmetric buffer fields must be non-negative")

    @property
    def resource_key(self) -> str:
        """Resource identifier used to prevent unsafe slot reuse."""

        return f"symmetric_buffer:{self.arena}:{self.slot}"


@dataclass(frozen=True)
class ReusableBufferSpec:
    """Stable logical slot in a non-symmetric reusable buffer arena."""

    arena: str
    slot: int
    capacity_bytes: int
    generation: int = 0

    def __post_init__(self) -> None:
        if not self.arena:
            raise ValueError("ReusableBufferSpec.arena must be non-empty")
        if min(self.slot, self.capacity_bytes, self.generation) < 0:
            raise ValueError("Reusable buffer fields must be non-negative")

    @property
    def resource_key(self) -> str:
        """Resource identifier used to prevent unsafe slot reuse."""

        return f"reusable_buffer:{self.arena}:{self.slot}"


@dataclass(frozen=True)
class OperationSpec:
    """Description of one logical compute or communication operation.

    Args:
        op_id: Rank-stable logical identity.
        kind: Compute or communication class.
        resources: Exclusive resource classes used during execution. Sharing a
            resource does not create a graph edge; the planner selects an order.
        bytes: Logical communication payload. Dynamic operations may use zero
            here and provide the measured size in a later plan signature.
        communicator_id: Rank-stable process-group or communication-domain ID.
        sequence: Canonical order within ``communicator_id``.
        ready_trigger: Earliest semantic point at which communication may issue.
        deadline_trigger: Consumer point by which communication should finish.
        release_trigger: Point after which an output/symmetric buffer may be reused.
        symmetric_buffer: Optional symmetric-memory placement.
        reusable_buffers: Stable ordinary cache slots used by this operation.
        priority: Tie breaker; smaller values have higher priority.
    """

    op_id: SemanticOpId
    kind: OperationKind
    resources: frozenset[str] = frozenset()
    bytes: int = 0
    communicator_id: str | None = None
    sequence: int | None = None
    ready_trigger: Trigger | None = None
    deadline_trigger: Trigger | None = None
    release_trigger: Trigger | None = None
    symmetric_buffer: SymmetricBufferSpec | None = None
    reusable_buffers: tuple[ReusableBufferSpec, ...] = ()
    priority: int = 0

    def __post_init__(self) -> None:
        if self.bytes < 0:
            raise ValueError("OperationSpec.bytes must be non-negative")
        if any(not resource for resource in self.resources):
            raise ValueError("OperationSpec.resources must not contain empty names")
        if (self.communicator_id is None) != (self.sequence is None):
            raise ValueError("communicator_id and sequence must be specified together")
        if self.sequence is not None and self.sequence < 0:
            raise ValueError("OperationSpec.sequence must be non-negative")
        reusable_keys = [buffer.resource_key for buffer in self.reusable_buffers]
        if len(reusable_keys) != len(set(reusable_keys)):
            raise ValueError("OperationSpec.reusable_buffers must be unique")
        if (
            self.symmetric_buffer is not None or self.reusable_buffers
        ) and self.release_trigger is None:
            raise ValueError("Buffered communication requires a release_trigger")
        if self.kind.is_communication and self.ready_trigger is None:
            raise ValueError("Communication operations require a ready_trigger")
        if not self.kind.is_communication:
            disallowed = (
                self.communicator_id,
                self.sequence,
                self.ready_trigger,
                self.deadline_trigger,
                self.release_trigger,
                self.symmetric_buffer,
            )
            if (
                any(value is not None for value in disallowed)
                or self.reusable_buffers
                or self.bytes
            ):
                raise ValueError("Compute operations cannot carry communication-only fields")


@dataclass(frozen=True)
class Dependency:
    """One hard ordering constraint in the semantic graph."""

    src: SemanticOpId
    dst: SemanticOpId
    kind: DependencyKind = DependencyKind.DATA

    def __post_init__(self) -> None:
        if self.src == self.dst:
            raise ValueError("A dependency cannot be a self-edge")


class OperationGraph:
    """Validated immutable semantic operation DAG."""

    def __init__(
        self, operations: Mapping[SemanticOpId, OperationSpec], dependencies: Iterable[Dependency]
    ) -> None:
        self._operations = MappingProxyType(dict(operations))
        self._dependencies = tuple(
            sorted(
                set(dependencies),
                key=lambda edge: (edge.src.stable_key, edge.dst.stable_key, edge.kind.value),
            )
        )
        self._successors: dict[SemanticOpId, set[SemanticOpId]] = defaultdict(set)
        self._predecessors: dict[SemanticOpId, set[SemanticOpId]] = defaultdict(set)
        for edge in self._dependencies:
            if edge.src not in self._operations or edge.dst not in self._operations:
                raise ValueError(f"Dependency references an unknown operation: {edge}")
            self._successors[edge.src].add(edge.dst)
            self._predecessors[edge.dst].add(edge.src)
        self.topological_order()

    @property
    def operations(self) -> Mapping[SemanticOpId, OperationSpec]:
        """All operation specifications keyed by semantic identity."""

        return self._operations

    @property
    def dependencies(self) -> tuple[Dependency, ...]:
        """All hard graph dependencies."""

        return self._dependencies

    def predecessors(self, op_id: SemanticOpId) -> frozenset[SemanticOpId]:
        """Return direct predecessors of ``op_id``."""

        self._require_operation(op_id)
        return frozenset(self._predecessors[op_id])

    def successors(self, op_id: SemanticOpId) -> frozenset[SemanticOpId]:
        """Return direct successors of ``op_id``."""

        self._require_operation(op_id)
        return frozenset(self._successors[op_id])

    def topological_order(self) -> tuple[SemanticOpId, ...]:
        """Return a deterministic topological order or raise on a cycle."""

        indegree = {op_id: len(self._predecessors[op_id]) for op_id in self._operations}
        ready = [(op_id.stable_key, op_id) for op_id, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        result = []
        while ready:
            _, op_id = heapq.heappop(ready)
            result.append(op_id)
            for successor in sorted(self._successors[op_id], key=lambda item: item.stable_key):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    heapq.heappush(ready, (successor.stable_key, successor))
        if len(result) != len(self._operations):
            cyclic = sorted((op_id.stable_key for op_id, degree in indegree.items() if degree > 0))
            raise ValueError(f"Operation graph contains a cycle involving: {cyclic}")
        return tuple(result)

    @property
    def fingerprint(self) -> str:
        """Content hash used to compare graph identity across ranks."""

        def trigger_data(trigger: Trigger | None) -> str | None:
            return trigger.stable_key if trigger is not None else None

        operations = []
        for op_id in sorted(self._operations, key=lambda item: item.stable_key):
            spec = self._operations[op_id]
            buffer = spec.symmetric_buffer
            operations.append(
                {
                    "id": op_id.stable_key,
                    "kind": spec.kind.value,
                    "resources": sorted(spec.resources),
                    "bytes": spec.bytes,
                    "communicator": spec.communicator_id,
                    "sequence": spec.sequence,
                    "ready": trigger_data(spec.ready_trigger),
                    "deadline": trigger_data(spec.deadline_trigger),
                    "release": trigger_data(spec.release_trigger),
                    "buffer": (
                        {
                            "arena": buffer.arena,
                            "slot": buffer.slot,
                            "offset": buffer.offset_bytes,
                            "capacity": buffer.capacity_bytes,
                            "generation": buffer.generation,
                        }
                        if buffer is not None
                        else None
                    ),
                    "reusable_buffers": [
                        {
                            "arena": buffer.arena,
                            "slot": buffer.slot,
                            "capacity": buffer.capacity_bytes,
                            "generation": buffer.generation,
                        }
                        for buffer in spec.reusable_buffers
                    ],
                    "priority": spec.priority,
                }
            )
        dependencies = [
            {"src": edge.src.stable_key, "dst": edge.dst.stable_key, "kind": edge.kind.value}
            for edge in self._dependencies
        ]
        payload = json.dumps(
            {"operations": operations, "dependencies": dependencies},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _require_operation(self, op_id: SemanticOpId) -> None:
        if op_id not in self._operations:
            raise KeyError(f"Unknown operation: {op_id}")


class OperationGraphBuilder:
    """Incremental builder used while observing eager execution."""

    def __init__(self) -> None:
        self._operations: dict[SemanticOpId, OperationSpec] = {}
        self._dependencies: set[Dependency] = set()

    def add_operation(self, operation: OperationSpec) -> OperationGraphBuilder:
        """Add one operation, rejecting incompatible duplicate registration."""

        prior = self._operations.get(operation.op_id)
        if prior is not None and prior != operation:
            raise ValueError(f"Operation {operation.op_id} was registered with different metadata")
        self._operations[operation.op_id] = operation
        return self

    def add_dependency(
        self, src: SemanticOpId, dst: SemanticOpId, kind: DependencyKind = DependencyKind.DATA
    ) -> OperationGraphBuilder:
        """Add a hard dependency without requiring insertion order."""

        self._dependencies.add(Dependency(src=src, dst=dst, kind=kind))
        return self

    def build(self) -> OperationGraph:
        """Validate operations and derive semantic/communicator edges."""

        dependencies = set(self._dependencies)
        communicator_ops: dict[str, list[OperationSpec]] = defaultdict(list)

        for operation in self._operations.values():
            for trigger in (
                operation.ready_trigger,
                operation.deadline_trigger,
                operation.release_trigger,
            ):
                if trigger is not None and trigger.op_id is not None:
                    if trigger.op_id not in self._operations:
                        raise ValueError(
                            f"Operation {operation.op_id} references "
                            f"unknown trigger {trigger.op_id}"
                        )

            ready = operation.ready_trigger
            if ready is not None and ready.kind is TriggerKind.OP_END:
                assert ready.op_id is not None
                dependencies.add(
                    Dependency(src=ready.op_id, dst=operation.op_id, kind=DependencyKind.DATA)
                )

            deadline = operation.deadline_trigger
            if deadline is not None and deadline.kind is TriggerKind.OP_START:
                assert deadline.op_id is not None
                dependencies.add(
                    Dependency(src=operation.op_id, dst=deadline.op_id, kind=DependencyKind.DATA)
                )

            if operation.communicator_id is not None:
                communicator_ops[operation.communicator_id].append(operation)

        for communicator_id, operations in communicator_ops.items():
            by_sequence = sorted(operations, key=lambda operation: operation.sequence)
            sequence_values = [operation.sequence for operation in by_sequence]
            if len(sequence_values) != len(set(sequence_values)):
                raise ValueError(f"Duplicate sequence in communicator {communicator_id}")
            for previous, current in zip(by_sequence, by_sequence[1:]):
                dependencies.add(
                    Dependency(
                        src=previous.op_id,
                        dst=current.op_id,
                        kind=DependencyKind.COMMUNICATOR_ORDER,
                    )
                )

        return OperationGraph(self._operations, dependencies)
