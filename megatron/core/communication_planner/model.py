# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compact GTP execution model built from eager CUDA profiling.

The model intentionally represents only the fixed GTP scheduling problem:
ordered module compute, the AG required by each module, and the RS made ready
by backward compute. RS is a side branch and never orders the next backward
module.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any


class GTPPhase(str, Enum):
    """Execution phase relevant to GTP materialization."""

    FORWARD = "forward"
    BACKWARD = "backward"
    RECOMPUTE = "recompute"


class GTPWorkKind(str, Enum):
    """Profiled work represented by the compact model."""

    COMPUTE = "compute"
    AG = "ag"
    RS = "rs"


@dataclass(frozen=True, order=True)
class GTPProfileKey:
    """Stable identity for one profiled GTP operation occurrence."""

    scope: str
    phase: GTPPhase
    kind: GTPWorkKind
    occurrence: int = 0

    def __post_init__(self) -> None:
        if not self.scope:
            raise ValueError("scope must be non-empty")
        if self.occurrence < 0:
            raise ValueError("occurrence must be non-negative")
        if self.kind is GTPWorkKind.RS and self.phase is not GTPPhase.BACKWARD:
            raise ValueError("RS is only valid in the backward phase")

    @property
    def stable_key(self) -> str:
        """Deterministic string used in runtime artifacts."""

        return f"{self.phase.value}:{self.scope}:{self.kind.value}:{self.occurrence}"


@dataclass(frozen=True)
class GTPCudaSample:
    """One CUDA-timeline duration sample."""

    key: GTPProfileKey
    iteration: int
    start_us: float
    end_us: float

    def __post_init__(self) -> None:
        values = (self.start_us, self.end_us)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("CUDA timestamps must be finite and non-negative")
        if self.iteration < 0:
            raise ValueError("iteration must be non-negative")
        if self.end_us < self.start_us:
            raise ValueError("end_us must not precede start_us")

    @property
    def duration_us(self) -> float:
        """CUDA service duration."""

        return self.end_us - self.start_us


@dataclass(frozen=True)
class GTPTimingStatistics:
    """Bounded timing summary for one operation occurrence."""

    count: int
    p50_us: float
    p95_us: float
    max_us: float


@dataclass(frozen=True, order=True)
class GTPDependency:
    """One dependency in the compact execution model."""

    src: GTPProfileKey
    dst: GTPProfileKey
    kind: str


def _quantile(values: Iterable[float], percentile: float) -> float:
    data = sorted(values)
    if not data:
        raise ValueError("at least one value is required")
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


class GTPExecutionModel:
    """Immutable compact graph and CUDA timing summaries."""

    def __init__(
        self,
        *,
        phase_orders: Mapping[GTPPhase, Iterable[GTPProfileKey]],
        samples: Iterable[GTPCudaSample],
        parameter_chains: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        orders = {}
        for phase, order in phase_orders.items():
            materialized = tuple(order)
            if materialized:
                orders[phase] = materialized
        for phase, order in orders.items():
            if any(key.phase is not phase or key.kind is not GTPWorkKind.COMPUTE for key in order):
                raise ValueError(f"{phase.value} order must contain matching compute keys")
            if len(order) != len(set(order)):
                raise ValueError(f"{phase.value} order contains duplicate operation keys")
        self.phase_orders: Mapping[GTPPhase, tuple[GTPProfileKey, ...]] = MappingProxyType(
            orders
        )

        grouped: dict[GTPProfileKey, list[float]] = defaultdict(list)
        for sample in samples:
            grouped[sample.key].append(sample.duration_us)
        self.statistics: Mapping[GTPProfileKey, GTPTimingStatistics] = MappingProxyType(
            {
                key: GTPTimingStatistics(
                    count=len(values),
                    p50_us=_quantile(values, 0.50),
                    p95_us=_quantile(values, 0.95),
                    max_us=max(values),
                )
                for key, values in sorted(grouped.items())
            }
        )
        self.parameter_chains: Mapping[str, tuple[str, ...]] = MappingProxyType(
            {
                name: tuple(scopes)
                for name, scopes in sorted((parameter_chains or {}).items())
            }
        )
        self.dependencies = self._build_dependencies()

    def _build_dependencies(self) -> tuple[GTPDependency, ...]:
        dependencies = set()
        available = set(self.statistics)
        for phase, order in self.phase_orders.items():
            for previous, current in zip(order, order[1:]):
                dependencies.add(GTPDependency(previous, current, "compute_order"))
            for compute in order:
                ag = GTPProfileKey(
                    scope=compute.scope,
                    phase=phase,
                    kind=GTPWorkKind.AG,
                    occurrence=compute.occurrence,
                )
                if ag in available:
                    dependencies.add(GTPDependency(ag, compute, "ag_before_compute"))
                if phase is GTPPhase.BACKWARD:
                    rs = GTPProfileKey(
                        scope=compute.scope,
                        phase=phase,
                        kind=GTPWorkKind.RS,
                        occurrence=compute.occurrence,
                    )
                    if rs in available:
                        dependencies.add(GTPDependency(compute, rs, "compute_before_rs"))
        return tuple(
            sorted(
                dependencies,
                key=lambda edge: (edge.src.stable_key, edge.dst.stable_key, edge.kind),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the compact model as a deterministic JSON-compatible object."""

        return {
            "format_version": 1,
            "phase_orders": {
                phase.value: [key.stable_key for key in order]
                for phase, order in self.phase_orders.items()
            },
            "parameter_chains": {
                name: list(scopes) for name, scopes in self.parameter_chains.items()
            },
            "operations": [
                {
                    "id": key.stable_key,
                    "scope": key.scope,
                    "phase": key.phase.value,
                    "kind": key.kind.value,
                    "occurrence": key.occurrence,
                    "cuda_duration": {
                        "count": stats.count,
                        "p50_us": stats.p50_us,
                        "p95_us": stats.p95_us,
                        "max_us": stats.max_us,
                    },
                }
                for key, stats in self.statistics.items()
            ],
            "dependencies": [
                {
                    "src": dependency.src.stable_key,
                    "dst": dependency.dst.stable_key,
                    "kind": dependency.kind,
                }
                for dependency in self.dependencies
            ],
        }
