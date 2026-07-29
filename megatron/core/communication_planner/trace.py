# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Chrome Trace export for the compact GTP CUDA execution model."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .model import (
    GTPCudaSample,
    GTPCommDomain,
    GTPDependency,
    GTPModuleInfo,
    GTPPhase,
    GTPProfileKey,
    GTPWorkKind,
)


_LANE_LAYOUT = {
    (GTPPhase.FORWARD, GTPWorkKind.COMPUTE_ELEMENT, None): (
        100,
        "Forward coarse schedule elements",
    ),
    (GTPPhase.RECOMPUTE, GTPWorkKind.COMPUTE_ELEMENT, None): (
        200,
        "Recompute coarse schedule elements",
    ),
    (GTPPhase.BACKWARD, GTPWorkKind.COMPUTE_ELEMENT, None): (
        300,
        "Backward coarse schedule elements",
    ),
    (GTPPhase.FORWARD, GTPWorkKind.COMPUTE, None): (
        400,
        "Forward GTP consumer intervals",
    ),
    (GTPPhase.RECOMPUTE, GTPWorkKind.COMPUTE, None): (
        500,
        "Recompute GTP consumer intervals",
    ),
    (GTPPhase.BACKWARD, GTPWorkKind.COMPUTE, None): (
        600,
        "Backward GTP consumer intervals",
    ),
    (GTPPhase.FORWARD, GTPWorkKind.AG, GTPCommDomain.GTP): (2000, "Forward GTP AG"),
    (GTPPhase.FORWARD, GTPWorkKind.AG, GTPCommDomain.EGTP): (
        2100,
        "Forward EGTP AG",
    ),
    (GTPPhase.RECOMPUTE, GTPWorkKind.AG, GTPCommDomain.GTP): (
        2200,
        "Recompute GTP AG",
    ),
    (GTPPhase.RECOMPUTE, GTPWorkKind.AG, GTPCommDomain.EGTP): (
        2300,
        "Recompute EGTP AG",
    ),
    (GTPPhase.BACKWARD, GTPWorkKind.AG, GTPCommDomain.GTP): (
        2400,
        "Backward GTP AG",
    ),
    (GTPPhase.BACKWARD, GTPWorkKind.AG, GTPCommDomain.EGTP): (
        2500,
        "Backward EGTP AG",
    ),
    (GTPPhase.BACKWARD, GTPWorkKind.RS, GTPCommDomain.GTP): (
        2600,
        "Backward GTP RS",
    ),
    (GTPPhase.BACKWARD, GTPWorkKind.RS, GTPCommDomain.EGTP): (
        2700,
        "Backward EGTP RS",
    ),
}
_FLOW_DEPENDENCIES = frozenset({"ag_before_compute", "compute_before_rs"})
_COLORS = {
    GTPWorkKind.COMPUTE: "thread_state_running",
    GTPWorkKind.COMPUTE_ELEMENT: "thread_state_running",
    GTPWorkKind.MATERIALIZE: "thread_state_iowait",
    GTPWorkKind.AG: "rail_response",
    GTPWorkKind.RS: "rail_animation",
}


def _metadata(name: str, pid: int, tid: int, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "ph": "M", "pid": pid, "tid": tid, "args": args}


def _pack_lanes(
    samples: Iterable[GTPCudaSample],
) -> tuple[dict[GTPProfileKey, int], list[tuple[int, str]]]:
    grouped: dict[
        tuple[GTPPhase, GTPWorkKind, GTPCommDomain | None], list[GTPCudaSample]
    ] = defaultdict(list)
    for sample in samples:
        if sample.key.kind in (
            GTPWorkKind.COMPUTE_ELEMENT,
            GTPWorkKind.MATERIALIZE,
        ):
            group = (
                sample.key.phase,
                GTPWorkKind.COMPUTE_ELEMENT,
                None,
            )
        elif sample.key.kind is GTPWorkKind.COMPUTE:
            group = (sample.key.phase, sample.key.kind, None)
        else:
            group = (sample.key.phase, sample.key.kind, sample.key.domain)
        grouped[group].append(sample)

    lane_for_key = {}
    lane_names = []
    for group in sorted(grouped, key=lambda item: _LANE_LAYOUT[item][0]):
        base_tid, label = _LANE_LAYOUT[group]
        lane_ends: list[float] = []
        for sample in sorted(
            grouped[group], key=lambda item: (item.start_us, item.key.stable_key)
        ):
            lane = next(
                (
                    index
                    for index, lane_end in enumerate(lane_ends)
                    if lane_end <= sample.start_us
                ),
                len(lane_ends),
            )
            if lane == len(lane_ends):
                lane_ends.append(sample.end_us)
                suffix = f" {lane + 1}" if lane else ""
                lane_names.append((base_tid + lane, f"{label}{suffix}"))
            else:
                lane_ends[lane] = sample.end_us
            lane_for_key[sample.key] = base_tid + lane
    return lane_for_key, lane_names


def build_gtp_chrome_trace(
    samples: Iterable[GTPCudaSample],
    dependencies: Iterable[GTPDependency],
    *,
    rank: int,
    parameter_modules: Mapping[str, GTPModuleInfo] | None = None,
    compute_element_targets: Mapping[GTPProfileKey, GTPProfileKey] | None = None,
) -> dict[str, Any]:
    """Build a compact Chrome Trace from raw GTP CUDA-event samples."""

    parameter_modules = parameter_modules or {}
    compute_element_targets = compute_element_targets or {}
    samples_by_iteration: dict[int, list[GTPCudaSample]] = defaultdict(list)
    for sample in samples:
        samples_by_iteration[sample.iteration].append(sample)
    flow_dependencies = [
        dependency
        for dependency in dependencies
        if dependency.kind in _FLOW_DEPENDENCIES
    ]

    events = []
    flow_id = 1
    for process_index, (iteration, iteration_samples) in enumerate(
        sorted(samples_by_iteration.items())
    ):
        pid = process_index + 1
        sample_by_key = {sample.key: sample for sample in iteration_samples}
        lane_for_key, lane_names = _pack_lanes(iteration_samples)
        events.extend(
            (
                _metadata(
                    "process_name",
                    pid,
                    0,
                    {"name": f"rank {rank} / iteration {iteration}"},
                ),
                _metadata("process_sort_index", pid, 0, {"sort_index": process_index}),
            )
        )
        for tid, name in lane_names:
            events.append(_metadata("thread_name", pid, tid, {"name": name}))
            events.append(_metadata("thread_sort_index", pid, tid, {"sort_index": tid}))

        for sample in sorted(
            iteration_samples, key=lambda item: (item.start_us, item.key.stable_key)
        ):
            key = sample.key
            target = compute_element_targets.get(key)
            source_module = parameter_modules.get(key.scope)
            target_module = parameter_modules.get(target.scope) if target is not None else None
            name = f"{key.scope} · {key.kind.value}"
            if target is not None:
                name = (
                    f"{_module_label(source_module)} · {_short_scope(key.scope)} → "
                    f"{_module_label(target_module)} · {_short_scope(target.scope)}"
                )
            events.append(
                {
                    "name": name,
                    "cat": (
                        f"gtp,{key.domain.value},{key.phase.value},{key.kind.value}"
                    ),
                    "ph": "X",
                    "ts": sample.start_us,
                    "dur": sample.duration_us,
                    "pid": pid,
                    "tid": lane_for_key[key],
                    "cname": _COLORS[key.kind],
                    "args": {
                        "id": key.stable_key,
                        "scope": key.scope,
                        "phase": key.phase.value,
                        "kind": key.kind.value,
                        "occurrence": key.occurrence,
                        "communication_domain": key.domain.value,
                        "module": (
                            _module_info_to_dict(source_module)
                            if source_module is not None
                            else None
                        ),
                        "next_consumer": (
                            {
                                "id": target.stable_key,
                                "scope": target.scope,
                                "module": (
                                    _module_info_to_dict(target_module)
                                    if target_module is not None
                                    else None
                                ),
                                "crosses_module_boundary": (
                                    source_module is not None
                                    and target_module is not None
                                    and source_module.scope != target_module.scope
                                ),
                            }
                            if target is not None
                            else None
                        ),
                        "cuda_duration_us": sample.duration_us,
                        "interval_note": (
                            "Direct GTP consumer interval"
                            if key.kind is GTPWorkKind.COMPUTE
                            else (
                                "Non-overlapping compute window until the next GTP consumer"
                                if key.kind is GTPWorkKind.COMPUTE_ELEMENT
                                else (
                                    "Exposed current-weight materialization interval"
                                    if key.kind is GTPWorkKind.MATERIALIZE
                                    else "CUDA service interval on the GTP communication stream"
                                )
                            )
                        ),
                    },
                }
            )

        for dependency in flow_dependencies:
            src = sample_by_key.get(dependency.src)
            dst = sample_by_key.get(dependency.dst)
            if src is None or dst is None:
                continue
            gap_us = dst.start_us - src.end_us
            events.extend(
                (
                    {
                        "name": dependency.kind,
                        "cat": "gtp_dependency",
                        "ph": "s",
                        "ts": src.end_us,
                        "pid": pid,
                        "tid": lane_for_key[src.key],
                        "id": flow_id,
                        "args": {"observed_gap_us": gap_us},
                    },
                    {
                        "name": dependency.kind,
                        "cat": "gtp_dependency",
                        "ph": "f",
                        "ts": dst.start_us,
                        "pid": pid,
                        "tid": lane_for_key[dst.key],
                        "id": flow_id,
                        "bp": "e",
                        "args": {"observed_gap_us": gap_us},
                    },
                )
            )
            flow_id += 1

    return {
        "displayTimeUnit": "ms",
        "traceEvents": events,
        "otherData": {
            "format": "compact GTP CUDA execution trace",
            "timestamp_unit": "microseconds",
            "rank": rank,
            "iterations": sorted(samples_by_iteration),
            "operation_kinds": [
                "compute_element",
                "materialize",
                "compute",
                "ag",
                "rs",
            ],
            "communication_domains": ["gtp", "egtp"],
            "dependency_kinds": sorted(_FLOW_DEPENDENCIES),
            "compute_element_semantics": (
                "current consumer compute-start to next consumer-ready; "
                "includes coarse parameterless work and excludes next materialization"
            ),
        },
    }


def _module_label(module: GTPModuleInfo | None) -> str:
    if module is None:
        return "?"
    if module.layer_number is None:
        return module.symbol
    return f"{module.symbol}{module.layer_number}"


def _short_scope(scope: str) -> str:
    return ".".join(scope.split(".")[-3:])


def _module_info_to_dict(module: GTPModuleInfo) -> dict[str, Any]:
    return {
        "scope": module.scope,
        "symbol": module.symbol,
        "layer_number": module.layer_number,
    }
