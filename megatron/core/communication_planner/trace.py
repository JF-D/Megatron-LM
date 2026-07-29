# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Chrome Trace export for the compact GTP CUDA execution model."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .model import GTPCudaSample, GTPDependency, GTPPhase, GTPProfileKey, GTPWorkKind


_LANE_LAYOUT = {
    (GTPPhase.FORWARD, GTPWorkKind.COMPUTE): (100, "Forward GTP consumer intervals"),
    (GTPPhase.RECOMPUTE, GTPWorkKind.COMPUTE): (120, "Recompute GTP consumer intervals"),
    (GTPPhase.BACKWARD, GTPWorkKind.COMPUTE): (140, "Backward GTP consumer intervals"),
    (GTPPhase.FORWARD, GTPWorkKind.AG): (200, "Forward GTP AG"),
    (GTPPhase.RECOMPUTE, GTPWorkKind.AG): (220, "Recompute GTP AG"),
    (GTPPhase.BACKWARD, GTPWorkKind.AG): (240, "Backward GTP AG"),
    (GTPPhase.BACKWARD, GTPWorkKind.RS): (300, "Backward GTP RS"),
}
_FLOW_DEPENDENCIES = frozenset({"ag_before_compute", "compute_before_rs"})
_COLORS = {
    GTPWorkKind.COMPUTE: "thread_state_running",
    GTPWorkKind.AG: "rail_response",
    GTPWorkKind.RS: "rail_animation",
}


def _metadata(name: str, pid: int, tid: int, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "ph": "M", "pid": pid, "tid": tid, "args": args}


def _pack_lanes(
    samples: Iterable[GTPCudaSample],
) -> tuple[dict[GTPProfileKey, int], list[tuple[int, str]]]:
    grouped: dict[tuple[GTPPhase, GTPWorkKind], list[GTPCudaSample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.key.phase, sample.key.kind)].append(sample)

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
) -> dict[str, Any]:
    """Build a compact Chrome Trace from raw GTP CUDA-event samples."""

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
            events.append(
                {
                    "name": f"{key.scope} · {key.kind.value}",
                    "cat": f"gtp,{key.phase.value},{key.kind.value}",
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
                        "cuda_duration_us": sample.duration_us,
                        "interval_note": (
                            "Gaps between GTP consumer intervals are unmodeled, not GPU idle"
                            if key.kind is GTPWorkKind.COMPUTE
                            else "CUDA service interval on the GTP communication stream"
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
            "operation_kinds": ["compute", "ag", "rs"],
            "dependency_kinds": sorted(_FLOW_DEPENDENCIES),
            "compute_gap_semantics": "unmodeled work; never interpret as GPU idle",
        },
    }
