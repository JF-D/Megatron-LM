# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compact GTP execution model built from eager CUDA profiling.

The model represents ordered GTP consumers, their exposed current-weight wait,
prefetch-issue gap, AG and RS, and the non-overlapping coarse compute element
between consecutive consumers. RS is a side branch and never orders the next
backward module.
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
    COMPUTE_ELEMENT = "compute_element"
    CONSUMER_WAIT = "consumer_wait"
    PREFETCH_ISSUE_GAP = "prefetch_issue_gap"
    AG = "ag"
    RS = "rs"


class GTPCommDomain(str, Enum):
    """Communication domain used by one GTP operation."""

    GTP = "gtp"
    EGTP = "egtp"


@dataclass(frozen=True)
class GTPModuleInfo:
    """Coarse model module containing a GTP parameter."""

    scope: str
    symbol: str
    layer_number: int | None = None

    def __post_init__(self) -> None:
        if not self.scope:
            raise ValueError("module scope must be non-empty")
        if not self.symbol:
            raise ValueError("module symbol must be non-empty")
        if self.layer_number is not None and self.layer_number < 1:
            raise ValueError("layer_number must be positive")


@dataclass(frozen=True, order=True)
class GTPProfileKey:
    """Stable identity for one profiled GTP operation occurrence."""

    scope: str
    phase: GTPPhase
    kind: GTPWorkKind
    occurrence: int = 0
    domain: GTPCommDomain = GTPCommDomain.GTP

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

        return (
            f"{self.phase.value}:{self.domain.value}:{self.scope}:"
            f"{self.kind.value}:{self.occurrence}"
        )


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
        parameter_modules: Mapping[str, GTPModuleInfo] | None = None,
        compute_element_targets: Mapping[GTPProfileKey, GTPProfileKey] | None = None,
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
        self.parameter_modules: Mapping[str, GTPModuleInfo] = MappingProxyType(
            dict(sorted((parameter_modules or {}).items()))
        )
        targets = dict(compute_element_targets or {})
        for element, target in targets.items():
            if element.kind is not GTPWorkKind.COMPUTE_ELEMENT:
                raise ValueError("compute element target source must be a compute element")
            if target.kind is not GTPWorkKind.CONSUMER_WAIT:
                raise ValueError("compute element target must be a consumer-wait operation")
        self.compute_element_targets: Mapping[GTPProfileKey, GTPProfileKey] = (
            MappingProxyType(targets)
        )
        self.dependencies = self._build_dependencies()

    def _build_dependencies(self) -> tuple[GTPDependency, ...]:
        dependencies = set()
        available = set(self.statistics)
        for phase, order in self.phase_orders.items():
            for previous, current in zip(order, order[1:]):
                dependencies.add(GTPDependency(previous, current, "compute_order"))
            for compute in order:
                consumer_wait = GTPProfileKey(
                    scope=compute.scope,
                    phase=phase,
                    kind=GTPWorkKind.CONSUMER_WAIT,
                    occurrence=compute.occurrence,
                    domain=compute.domain,
                )
                issue_gap = GTPProfileKey(
                    scope=compute.scope,
                    phase=phase,
                    kind=GTPWorkKind.PREFETCH_ISSUE_GAP,
                    occurrence=compute.occurrence,
                    domain=compute.domain,
                )
                if consumer_wait in available and issue_gap in available:
                    dependencies.add(
                        GTPDependency(
                            consumer_wait,
                            issue_gap,
                            "consumer_wait_before_prefetch_issue",
                        )
                    )
                if issue_gap in available:
                    dependencies.add(
                        GTPDependency(
                            issue_gap,
                            compute,
                            "prefetch_issue_before_compute",
                        )
                    )
                ag = GTPProfileKey(
                    scope=compute.scope,
                    phase=phase,
                    kind=GTPWorkKind.AG,
                    occurrence=compute.occurrence,
                    domain=compute.domain,
                )
                if ag in available:
                    dependencies.add(GTPDependency(ag, compute, "ag_before_compute"))
                if phase is GTPPhase.BACKWARD:
                    rs = GTPProfileKey(
                        scope=compute.scope,
                        phase=phase,
                        kind=GTPWorkKind.RS,
                        occurrence=compute.occurrence,
                        domain=compute.domain,
                    )
                    if rs in available:
                        dependencies.add(GTPDependency(compute, rs, "compute_before_rs"))
        for element, target in self.compute_element_targets.items():
            if element in available and target in available:
                dependencies.add(
                    GTPDependency(element, target, "compute_before_consumer_wait")
                )
        return tuple(
            sorted(
                dependencies,
                key=lambda edge: (edge.src.stable_key, edge.dst.stable_key, edge.kind),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the compact model as a deterministic JSON-compatible object."""

        operations = []
        for key, stats in self.statistics.items():
            operation = {
                "id": key.stable_key,
                "scope": key.scope,
                "phase": key.phase.value,
                "kind": key.kind.value,
                "occurrence": key.occurrence,
                "communication_domain": key.domain.value,
                "cuda_duration": {
                    "count": stats.count,
                    "p50_us": stats.p50_us,
                    "p95_us": stats.p95_us,
                    "max_us": stats.max_us,
                },
            }
            module = self.parameter_modules.get(key.scope)
            if module is not None:
                operation["module"] = _module_info_to_dict(module)
            target = self.compute_element_targets.get(key)
            if target is not None:
                target_module = self.parameter_modules.get(target.scope)
                operation["next_consumer"] = {
                    "id": target.stable_key,
                    "scope": target.scope,
                    "module": (
                        _module_info_to_dict(target_module)
                        if target_module is not None
                        else None
                    ),
                    "crosses_module_boundary": (
                        module is not None
                        and target_module is not None
                        and module.scope != target_module.scope
                    ),
                }
            operations.append(operation)

        return {
            "format_version": 4,
            "phase_orders": {
                phase.value: [key.stable_key for key in order]
                for phase, order in self.phase_orders.items()
            },
            "parameter_chains": {
                name: list(scopes) for name, scopes in self.parameter_chains.items()
            },
            "parameter_modules": {
                scope: _module_info_to_dict(module)
                for scope, module in self.parameter_modules.items()
            },
            "compute_element_order": [
                key.stable_key for key in self.compute_element_targets
            ],
            "operations": operations,
            "dependencies": [
                {
                    "src": dependency.src.stable_key,
                    "dst": dependency.dst.stable_key,
                    "kind": dependency.kind,
                }
                for dependency in self.dependencies
            ],
        }


def _module_info_to_dict(module: GTPModuleInfo) -> dict[str, Any]:
    return {
        "scope": module.scope,
        "symbol": module.symbol,
        "layer_number": module.layer_number,
    }
