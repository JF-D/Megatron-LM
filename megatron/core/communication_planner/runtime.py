# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Default-off runtime integration for communication-planner shadow profiling.

This module deliberately does not own or launch production collectives. Hook
sites publish stable semantic operations and CUDA-event markers here while the
existing GTP and EP launch paths continue unchanged. Plan enforcement remains
disabled until rank-consensus and buffer-lifetime validation pass.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from .graph import (
    Dependency,
    DependencyKind,
    OperationKind,
    OperationSpec,
    Phase,
    ReusableBufferSpec,
    SemanticOpId,
    Trigger,
    TriggerKind,
)
from .planner import RuntimePlan
from .session import RuntimePlanningSession
from .telemetry import (
    CudaEventRecorder,
    OperationSample,
    TelemetryStore,
    TimelineMarker,
)

logger = logging.getLogger(__name__)


class RuntimePlannerMode(str, Enum):
    """User-visible runtime planner modes."""

    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


@dataclass(frozen=True)
class RuntimePlannerConfig:
    """Configuration for runtime discovery and bounded shadow profiling."""

    mode: RuntimePlannerMode = RuntimePlannerMode.OFF
    warmup_iters: int = 2
    profile_iters: int = 4
    replan_interval: int = 0
    log_dir: str | None = None
    dump_plan: bool = False
    validate_ranks: str = "0"
    runtime_signature: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.warmup_iters < 1:
            raise ValueError("runtime planner warmup_iters must be at least one")
        if self.profile_iters < 1:
            raise ValueError("runtime planner profile_iters must be at least one")
        if self.replan_interval < 0:
            raise ValueError("runtime planner replan_interval must be non-negative")
        if self.mode is not RuntimePlannerMode.OFF and not self.log_dir:
            raise ValueError("runtime planner log_dir is required outside off mode")


@dataclass
class RuntimeCollectiveToken:
    """Opaque token returned to one instrumented collective call."""

    op_id: SemanticOpId
    nvtx_range_id: int | None = None
    completion_recorded: bool = False
    keepalive: tuple[object, ...] = ()


class _DisabledRuntimePlanner:
    """No-op object used by the production hot path in planner-off mode."""

    enabled = False
    active = False
    hooks_enabled = False
    mode = RuntimePlannerMode.OFF

    def begin_iteration(self, iteration: int, stream: object | None = None) -> None:
        del iteration, stream

    def end_iteration(self, stream: object | None = None) -> None:
        del stream

    def finalize(self) -> None:
        pass

    def set_microbatch(self, microbatch: int) -> None:
        del microbatch

    def tag_model(self, model: object) -> None:
        del model

    def param_bucket_ready(
        self,
        scope: str,
        parameter_scopes: tuple[str, ...],
        stream: object | None = None,
    ) -> None:
        del scope, parameter_scopes, stream

    def gtp_consumer_ready(
        self, scope: str, direction: str, stream: object | None = None
    ) -> None:
        del scope, direction, stream

    def gtp_consumer_resume(
        self, scope: str, direction: str, stream: object | None = None
    ) -> None:
        del scope, direction, stream

    def gtp_ag_ready(
        self,
        scope: str,
        *,
        expert: bool,
        direction: str,
        communicator_size: int,
        payload_bytes: int,
        parameter_scopes: tuple[str, ...] = (),
        reusable_buffers: tuple[tuple[str, int, int, int], ...] = (),
        stream: object | None = None,
    ) -> RuntimeCollectiveToken | None:
        del (
            scope,
            expert,
            direction,
            communicator_size,
            payload_bytes,
            parameter_scopes,
            reusable_buffers,
            stream,
        )
        return None

    def gtp_rs_ready(
        self,
        scope: str,
        *,
        expert: bool,
        communicator_size: int,
        payload_bytes: int,
        reusable_buffers: tuple[tuple[str, int, int, int], ...] = (),
        stream: object | None = None,
    ) -> RuntimeCollectiveToken | None:
        del scope, expert, communicator_size, payload_bytes, reusable_buffers, stream
        return None

    def collective_start(
        self, token: RuntimeCollectiveToken | None, stream: object | None = None
    ) -> None:
        del token, stream

    def collective_end(
        self, token: RuntimeCollectiveToken | None, stream: object | None = None
    ) -> None:
        del token, stream

    def collective_completion(
        self,
        token: RuntimeCollectiveToken | None,
        work: object | None,
        *,
        keepalive: tuple[object, ...] = (),
    ) -> None:
        del token, work, keepalive

    def gtp_rs_consumer_ready(self, scope: str, stream: object | None = None) -> None:
        del scope, stream

    def gtp_rs_consumer_resume(self, scope: str, stream: object | None = None) -> None:
        del scope, stream

    def gtp_rs_finalize_end(self, scope: str, stream: object | None = None) -> None:
        del scope, stream

    def ep_start(
        self,
        scope: str,
        *,
        phase: Phase,
        kind: OperationKind,
        communicator_size: int,
        payload_bytes: int,
        stream: object | None = None,
    ) -> RuntimeCollectiveToken | None:
        del scope, phase, kind, communicator_size, payload_bytes, stream
        return None

    def ep_end(
        self, token: RuntimeCollectiveToken | None, stream: object | None = None
    ) -> None:
        del token, stream


class RuntimeCommunicationPlannerRuntime:
    """Observe eager execution without changing communication launch policy."""

    enabled = True

    def __init__(
        self,
        config: RuntimePlannerConfig,
        *,
        event_factory=None,
        completion_stream_factory=None,
        stream_context=None,
    ) -> None:
        if config.mode is RuntimePlannerMode.OFF:
            raise ValueError("Use the disabled runtime for planner-off mode")
        self.config = config
        self.mode = config.mode
        self.effective_mode = RuntimePlannerMode.SHADOW
        self.session = RuntimePlanningSession(
            CudaEventRecorder(event_factory),
            telemetry=TelemetryStore(max_samples_per_operation=config.profile_iters),
        )
        self._completion_stream_factory = (
            completion_stream_factory or (lambda: torch.cuda.Stream())
        )
        self._stream_context = stream_context or torch.cuda.stream
        self._completion_stream = None
        self._artifact_dir = Path(config.log_dir).expanduser().resolve()
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._errors: list[str] = []
        self._disabled = False
        self._rank = self._distributed_rank()
        self._world_size = self._distributed_world_size()
        self._dump_this_rank = self._rank_selected(config.validate_ranks)
        self._runtime_signature_key = json.dumps(
            self._topology_equivalence_signature(config.runtime_signature),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        self._control_group = None
        self._control_group_backend: str | None = None

        self._active_iteration: int | None = None
        self._observed_iterations = 0
        self._discovering = False
        self._iteration_operations: set[SemanticOpId] = set()
        self._graph_frozen = False
        self._recording = False
        self._profile_iteration_active = False
        self._iteration_nvtx_open = False
        self._profile_started = 0
        self._profile_completed: set[int] = set()
        self._microbatch = 0
        self._communicator_sequences: dict[str, int] = {}
        self._active_compute: dict[tuple[Phase, int], tuple[SemanticOpId, SemanticOpId]] = {}
        self._param_scope_to_ready: dict[str, SemanticOpId] = {}
        self._param_bucket_ready_order: dict[SemanticOpId, int] = {}
        self._param_bucket_ready_seen: set[SemanticOpId] = set()
        self._next_param_bucket_ready_order = 0
        self._param_readiness_hook_handles: list[object] = []
        self._gtp_ready_sources: dict[SemanticOpId, str] = {}
        self._missing_forward_param_ready_scopes: set[str] = set()
        self._rs_tokens: dict[tuple[str, int], SemanticOpId] = {}
        self._rs_finalize_started: set[SemanticOpId] = set()
        self._ep_symmetric_metadata_missing: set[SemanticOpId] = set()
        self._dynamic_payload_bytes: dict[SemanticOpId, list[int]] = {}
        self._dynamic_payload_iterations: set[tuple[int, SemanticOpId]] = set()

        self._plan: RuntimePlan | None = None
        self._plan_build_us: float | None = None
        self._plan_compile_us: float | None = None
        self._plan_validation_us: float | None = None
        self._telemetry_consensus_us: float | None = None
        self._profile_readiness_us = 0.0
        self._profile_readiness_rounds = 0
        self._plan_consensus: bool | None = None
        self._consensus_group_ranks: tuple[int, ...] = ()
        self._consensus_samples_per_operation: int | None = None
        self._graph_build_us: float | None = None
        self._setup_overhead_us = 0.0
        self._hook_overhead_us = 0.0
        self._control_plane_overhead_us = 0.0
        self._max_pending_iterations = 0
        self._skipped_profile_iterations = 0
        self._fallback_reason: str | None = None
        self._last_status_signature: str | None = None
        if config.mode is RuntimePlannerMode.ENFORCE:
            self._fallback_reason = (
                "enforcement requested before rank-consensus and symmetric-buffer validation; "
                "running collectively safe shadow fallback"
            )
        if config.replan_interval:
            self._errors.append(
                "replan_interval is recorded but periodic replanning is not enabled in this "
                "first shadow integration"
            )
        if self._world_size > 1:
            try:
                self._control_group = torch.distributed.new_group(
                    ranks=list(range(self._world_size))
                )
                self._control_group_backend = str(
                    torch.distributed.get_backend(self._control_group)
                )
            except Exception as exc:
                self._disable(
                    "planner control-group creation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        self._dump_status()

    def tag_model(self, model: object) -> None:
        """Attach stable model paths and observe DDP parameter-ready points."""

        started = time.perf_counter()
        chunks = list(model) if isinstance(model, (list, tuple)) else [model]
        include_chunk = len(chunks) > 1
        for chunk_index, chunk in enumerate(chunks):
            if not hasattr(chunk, "named_modules"):
                continue
            parameter_scopes = {}
            for name, parameter in chunk.named_parameters():
                prefix = f"model_chunk_{chunk_index}." if include_chunk else ""
                scope = self._normalize_scope(f"{prefix}{name}")
                parameter_scopes[parameter] = scope
                parameter._runtime_comm_planner_scope = scope
            self._tag_param_bucket_readiness(
                chunk, chunk_index, parameter_scopes, include_chunk
            )
            for name, module in chunk.named_modules():
                prefix = f"model_chunk_{chunk_index}." if include_chunk else ""
                module_scope = f"{prefix}{name or module.__class__.__name__}"
                candidates = (
                    (module, module_scope),
                    (
                        getattr(module, "token_dispatcher", None),
                        f"{module_scope}.token_dispatcher",
                    ),
                )
                for candidate, scope in candidates:
                    manager = getattr(candidate, "_comm_manager", None)
                    if manager is not None:
                        manager.runtime_comm_planner_scope = self._normalize_scope(scope)
        self._add_setup_overhead(started)

    def param_bucket_ready(
        self,
        scope: str,
        parameter_scopes: tuple[str, ...],
        stream: object | None = None,
    ) -> None:
        """Mark a DDP bucket whose synchronized parameters are safe for GTP AG."""

        started = time.perf_counter()
        if not self._observing:
            return
        bucket_id = self._semantic_id(
            self._normalize_scope(scope), Phase.FORWARD, "dp_param_ready"
        )
        self._register_compute(bucket_id)
        if bucket_id not in self._param_bucket_ready_seen:
            self._record_point(bucket_id, stream)
            self._param_bucket_ready_seen.add(bucket_id)
            self._param_bucket_ready_order[bucket_id] = (
                self._next_param_bucket_ready_order
            )
            self._next_param_bucket_ready_order += 1
        for parameter_scope in parameter_scopes:
            self._param_scope_to_ready[
                self._normalize_scope(parameter_scope)
            ] = bucket_id
        self._add_overhead(started)

    def begin_iteration(self, iteration: int, stream: object | None = None) -> None:
        """Begin discovery or one bounded profile iteration."""

        started = time.perf_counter()
        if self._active_iteration is not None:
            self._disable(f"iteration {self._active_iteration} was not closed")
            return
        self._active_iteration = iteration
        self._microbatch = 0
        self._iteration_operations = set()
        self._communicator_sequences = {}
        self._active_compute = {}
        self._param_scope_to_ready = {}
        self._param_bucket_ready_order = {}
        self._param_bucket_ready_seen = set()
        self._next_param_bucket_ready_order = 0
        self._rs_tokens = {}
        self._rs_finalize_started = set()

        self._add_overhead(started)
        control_started = time.perf_counter()
        self._collect_completed()
        self._add_control_plane_overhead(control_started)
        started = time.perf_counter()
        # Earlier warmup iterations exercise lazy initialization without
        # contaminating the steady-state graph. The final warmup iteration is
        # the single discovery template, which is then checked during every
        # profiled iteration.
        self._discovering = (
            not self._graph_frozen
            and self._observed_iterations == self.config.warmup_iters - 1
        )
        can_profile = (
            self._graph_frozen
            and self._profile_started < self.config.profile_iters
            and self.session.recorder.pending_iterations < self.config.profile_iters
            and not self._disabled
        )
        if (
            self._graph_frozen
            and self._profile_started < self.config.profile_iters
            and not can_profile
            and not self._disabled
        ):
            self._skipped_profile_iterations += 1
        self._recording = bool(can_profile)
        if self._recording:
            self.session.begin_iteration(iteration, stream)
            self._profile_iteration_active = True
            self._profile_started += 1
        self._iteration_nvtx_open = self._discovering or self._recording
        if self._iteration_nvtx_open:
            self._nvtx_push(
                "runtime_comm_planner::"
                + ("discover" if self._discovering else "profile")
                + f"::iter{iteration}"
            )
        self._add_overhead(started)

    def end_iteration(self, stream: object | None = None) -> None:
        """Close the active template/profile iteration and query prior samples."""

        started = time.perf_counter()
        if self._active_iteration is None:
            return
        self._close_active_computes(stream)
        if self._profile_iteration_active:
            self.session.end_iteration(stream)
            self._profile_iteration_active = False
        if self._recording:
            expected = frozenset(self.session.graph.operations)
            observed = frozenset(self._iteration_operations)
            if observed != expected:
                missing = sorted(op.stable_key for op in expected.difference(observed))
                extra = sorted(op.stable_key for op in observed.difference(expected))
                self._disable(
                    f"semantic graph changed during profile; "
                    f"missing={missing[:8]}, extra={extra[:8]}"
                )
        if self._iteration_nvtx_open:
            self._nvtx_pop()
            self._iteration_nvtx_open = False

        self._observed_iterations += 1
        self._active_iteration = None
        self._discovering = False
        self._recording = False
        self._add_overhead(started)

        control_started = time.perf_counter()
        if (
            not self._graph_frozen
            and self._observed_iterations >= self.config.warmup_iters
            and not self._disabled
        ):
            graph_started = time.perf_counter()
            try:
                self.session.freeze_graph()
                self._graph_frozen = True
                self._graph_build_us = (time.perf_counter() - graph_started) * 1.0e6
                self._dump_graph()
            except Exception as exc:  # Shadow mode must preserve the production launch policy.
                self._disable(f"graph freeze failed: {type(exc).__name__}: {exc}")

        self._collect_completed()
        self._maybe_compile()
        self._dump_status()
        self._add_control_plane_overhead(control_started)

    def finalize(self) -> None:
        """Write final bounded diagnostics after the training loop exits."""

        started = time.perf_counter()
        self._collect_completed()
        self._maybe_compile()
        self._dump_status(force=True)
        for handle in self._param_readiness_hook_handles:
            handle.remove()
        self._param_readiness_hook_handles.clear()
        self._add_control_plane_overhead(started)

    def set_microbatch(self, microbatch: int) -> None:
        """Publish the schedule's stable microbatch index."""

        if microbatch < 0:
            self._disable(f"invalid microbatch index {microbatch}")
            return
        if self._active_iteration is not None and microbatch != self._microbatch:
            for phase, active_microbatch in list(self._active_compute):
                if active_microbatch == self._microbatch:
                    self._finish_active_compute(
                        phase, active_microbatch, torch.cuda.current_stream()
                    )
        self._microbatch = microbatch

    def gtp_consumer_ready(
        self, scope: str, direction: str, stream: object | None = None
    ) -> None:
        """Mark a weight consumer reaching its AG dependency."""

        started = time.perf_counter()
        if not self._observing:
            return
        phase, role = self._gtp_phase_role(direction)
        self._finish_active_compute(phase, self._microbatch, stream)
        ag_id, compute_id, _ = self._gtp_ag_ids(scope, phase, role)
        self._register_compute(compute_id)
        self._record(ag_id, TimelineMarker.CONSUMER_READY, stream)
        self._mark(f"runtime_comm_planner::consumer_ready::{ag_id.stable_key}")
        self._add_overhead(started)

    def gtp_consumer_resume(
        self, scope: str, direction: str, stream: object | None = None
    ) -> None:
        """Mark AG availability and the beginning of dependent computation."""

        started = time.perf_counter()
        if not self._observing:
            return
        phase, role = self._gtp_phase_role(direction)
        ag_id, compute_id, _ = self._gtp_ag_ids(scope, phase, role)
        self._record(ag_id, TimelineMarker.CONSUMER_RESUME, stream)
        self._record(compute_id, TimelineMarker.START, stream)
        self._active_compute[(phase, self._microbatch)] = (compute_id, ag_id)
        self._mark(f"runtime_comm_planner::consumer_resume::{ag_id.stable_key}")
        self._add_overhead(started)

    def gtp_ag_ready(
        self,
        scope: str,
        *,
        expert: bool,
        direction: str,
        communicator_size: int,
        payload_bytes: int,
        parameter_scopes: tuple[str, ...] = (),
        reusable_buffers: tuple[tuple[str, int, int, int], ...] = (),
        stream: object | None = None,
    ) -> RuntimeCollectiveToken | None:
        """Register a dense/expert GTP AG and mark its input ready."""

        started = time.perf_counter()
        if not self._observing:
            return None
        phase, role = self._gtp_phase_role(direction)
        ag_id, compute_id, producer_id = self._gtp_ag_ids(scope, phase, role)
        communicator = self._communicator_id("expert_gtp" if expert else "dense_gtp", communicator_size)
        sequence = self._next_sequence(communicator)
        self._register_compute(compute_id)
        ready_trigger, ready_dependencies = self._gtp_ag_ready_trigger(
            ag_id=ag_id,
            producer_id=producer_id,
            phase=phase,
            direction=direction,
            parameter_scopes=parameter_scopes,
            stream=stream,
        )
        self._register_operation(
            OperationSpec(
                op_id=ag_id,
                kind=OperationKind.GTP_EXPERT_AG if expert else OperationKind.GTP_DENSE_AG,
                resources=frozenset(
                    {
                        "cross_domain_fabric",
                        "comm_sm",
                        "expert_gtp" if expert else "dense_gtp",
                    }
                ),
                bytes=max(0, int(payload_bytes)),
                communicator_id=communicator,
                sequence=sequence,
                ready_trigger=ready_trigger,
                deadline_trigger=Trigger.consumer_ready(ag_id),
                release_trigger=Trigger.op_end(compute_id),
                reusable_buffers=self._reusable_buffer_specs(reusable_buffers),
                priority=0,
            )
        )
        for ready_dependency in ready_dependencies:
            self._register_dependency(ready_dependency, ag_id)
        self._register_dependency(ag_id, compute_id)
        self._record_ready_trigger(ag_id, ready_trigger)
        self._mark(f"runtime_comm_planner::ready::{ag_id.stable_key}")
        self._add_overhead(started)
        return RuntimeCollectiveToken(ag_id)

    def gtp_rs_ready(
        self,
        scope: str,
        *,
        expert: bool,
        communicator_size: int,
        payload_bytes: int,
        reusable_buffers: tuple[tuple[str, int, int, int], ...] = (),
        stream: object | None = None,
    ) -> RuntimeCollectiveToken | None:
        """Register a dense/expert wgrad RS after its producer compute."""

        started = time.perf_counter()
        if not self._observing:
            return None
        phase = Phase.BACKWARD
        normalized_scope = self._normalize_scope(scope)
        compute_id = self._semantic_id(normalized_scope, phase, "bwd_compute")
        compute_was_active = self._finish_compute_id(compute_id, stream)
        finalize_id = self._semantic_id(normalized_scope, phase, "wgrad_finalize")
        rs_id = self._semantic_id(normalized_scope, phase, "wgrad_rs")
        communicator = self._communicator_id("expert_gtp" if expert else "dense_gtp", communicator_size)
        sequence = self._next_sequence(communicator)
        self._register_compute(compute_id)
        self._register_compute(finalize_id)
        self._register_operation(
            OperationSpec(
                op_id=rs_id,
                kind=OperationKind.GTP_EXPERT_RS if expert else OperationKind.GTP_DENSE_RS,
                resources=frozenset(
                    {
                        "cross_domain_fabric",
                        "comm_sm",
                        "expert_gtp" if expert else "dense_gtp",
                    }
                ),
                bytes=max(0, int(payload_bytes)),
                communicator_id=communicator,
                sequence=sequence,
                ready_trigger=Trigger.op_end(compute_id),
                deadline_trigger=Trigger.consumer_ready(rs_id),
                release_trigger=Trigger.op_end(finalize_id),
                reusable_buffers=self._reusable_buffer_specs(reusable_buffers),
                priority=0,
            )
        )
        self._register_dependency(rs_id, finalize_id)
        self._rs_tokens[(normalized_scope, self._microbatch)] = rs_id
        if not compute_was_active:
            # Embedding gradients and other direct RS inputs have no preceding
            # materialized-weight consumer interval. Their producer becomes
            # ready at this callsite, so represent it as a point operation.
            self._record_point(compute_id, stream)
        self._record(rs_id, TimelineMarker.READY, stream)
        self._mark(f"runtime_comm_planner::ready::{rs_id.stable_key}")
        self._add_overhead(started)
        return RuntimeCollectiveToken(rs_id)

    def collective_start(
        self, token: RuntimeCollectiveToken | None, stream: object | None = None
    ) -> None:
        """Mark actual collective service start on its execution stream."""

        if token is None or not self._observing:
            return
        started = time.perf_counter()
        self._record(token.op_id, TimelineMarker.START, stream)
        token.nvtx_range_id = self._nvtx_range_start(
            f"runtime_comm_planner::op::{token.op_id.stable_key}"
        )
        self._add_overhead(started)

    def collective_end(
        self, token: RuntimeCollectiveToken | None, stream: object | None = None
    ) -> None:
        """Mark the production stream draining one collective."""

        if token is None:
            return
        started = time.perf_counter()
        if self._observing:
            marker = (
                TimelineMarker.DRAIN
                if token.completion_recorded
                else TimelineMarker.END
            )
            self._record(token.op_id, marker, stream)
        if token.nvtx_range_id is not None:
            self._nvtx_range_end(token.nvtx_range_id)
            token.nvtx_range_id = None
        # WorkNCCL::wait() releases ProcessGroupNCCL's allocator-safety
        # stash. Keep the exact collective tensors alive until the normal
        # production drain has established its stream dependency.
        token.keepalive = ()
        if self._observing:
            self._add_overhead(started)

    def collective_completion(
        self,
        token: RuntimeCollectiveToken | None,
        work: object | None,
        *,
        keepalive: tuple[object, ...] = (),
    ) -> None:
        """Record asynchronous NCCL completion without changing production waits."""

        if token is None or work is None or not self._recording:
            return
        started = time.perf_counter()
        token.keepalive = tuple(keepalive)
        try:
            if self._completion_stream is None:
                self._completion_stream = self._completion_stream_factory()
            with self._stream_context(self._completion_stream):
                # CUDA Work.wait() only makes this stream wait for NCCL. The
                # existing production stream still performs its own wait later.
                work.wait()
                self._record(
                    token.op_id,
                    TimelineMarker.END,
                    self._completion_stream,
                )
            token.completion_recorded = True
        except Exception as exc:
            token.keepalive = ()
            self._disable(
                "collective completion probe failed: "
                f"{type(exc).__name__}: {exc}"
            )
        self._add_overhead(started)

    def gtp_rs_consumer_ready(self, scope: str, stream: object | None = None) -> None:
        """Mark the point that gradient finalization needs an RS output."""

        started = time.perf_counter()
        rs_id, finalize_id = self._rs_finalize_ids(scope)
        if rs_id is None:
            return
        self._register_compute(finalize_id)
        self._record(rs_id, TimelineMarker.CONSUMER_READY, stream)
        self._record(finalize_id, TimelineMarker.START, stream)
        self._rs_finalize_started.add(finalize_id)
        self._mark(f"runtime_comm_planner::consumer_ready::{rs_id.stable_key}")
        self._add_overhead(started)

    def gtp_rs_consumer_resume(self, scope: str, stream: object | None = None) -> None:
        """Mark the stream resuming after its RS dependency."""

        started = time.perf_counter()
        rs_id, _ = self._rs_finalize_ids(scope)
        if rs_id is None:
            return
        self._record(rs_id, TimelineMarker.CONSUMER_RESUME, stream)
        self._mark(f"runtime_comm_planner::consumer_resume::{rs_id.stable_key}")
        self._add_overhead(started)

    def gtp_rs_finalize_end(self, scope: str, stream: object | None = None) -> None:
        """Mark main-gradient accumulation complete and release the RS output."""

        started = time.perf_counter()
        rs_id, finalize_id = self._rs_finalize_ids(scope)
        if rs_id is None or finalize_id not in self._rs_finalize_started:
            return
        self._record(finalize_id, TimelineMarker.END, stream)
        self._record(rs_id, TimelineMarker.RELEASE, stream)
        self._rs_finalize_started.discard(finalize_id)
        self._mark(f"runtime_comm_planner::release::{rs_id.stable_key}")
        self._add_overhead(started)

    def ep_start(
        self,
        scope: str,
        *,
        phase: Phase,
        kind: OperationKind,
        communicator_size: int,
        payload_bytes: int,
        stream: object | None = None,
    ) -> RuntimeCollectiveToken | None:
        """Register and start one HybridEP dispatch/combine operation."""

        started = time.perf_counter()
        if not self._observing:
            return None
        if kind not in (OperationKind.EP_DISPATCH, OperationKind.EP_COMBINE):
            self._disable(f"unsupported EP operation kind {kind}")
            return None
        normalized_scope = self._normalize_scope(scope or "unresolved_hybridep")
        role = kind.value
        op_id = self._semantic_id(normalized_scope, phase, role)
        producer_id = self._semantic_id(normalized_scope, phase, f"{role}_input")
        consumer_id = self._semantic_id(normalized_scope, phase, f"{role}_consumer")
        self._record_dynamic_payload(op_id, payload_bytes)
        communicator = self._communicator_id("expert_parallel", communicator_size)
        sequence = self._next_sequence(communicator)
        self._register_compute(producer_id)
        self._register_compute(consumer_id)
        self._register_operation(
            OperationSpec(
                op_id=op_id,
                kind=kind,
                resources=frozenset({"cross_domain_fabric", "comm_sm", "ep_all_to_all"}),
                # Routed-token counts vary by rank and iteration. Keep dynamic
                # payloads out of immutable graph identity and emit bounded
                # observed sizes in diagnostics/telemetry instead.
                bytes=0,
                communicator_id=communicator,
                sequence=sequence,
                ready_trigger=Trigger.op_end(producer_id),
                deadline_trigger=Trigger.consumer_ready(op_id),
                release_trigger=Trigger.op_end(consumer_id),
                priority=1,
            )
        )
        self._register_dependency(op_id, consumer_id)
        self._ep_symmetric_metadata_missing.add(op_id)
        self._record_point(producer_id, stream)
        self._record(op_id, TimelineMarker.READY, stream)
        self._record(op_id, TimelineMarker.CONSUMER_READY, stream)
        self._record(op_id, TimelineMarker.START, stream)
        token = RuntimeCollectiveToken(op_id)
        token.nvtx_range_id = self._nvtx_range_start(
            f"runtime_comm_planner::op::{op_id.stable_key}"
        )
        self._add_overhead(started)
        return token

    def ep_end(
        self, token: RuntimeCollectiveToken | None, stream: object | None = None
    ) -> None:
        """Complete one HybridEP operation and its immediate consumer gate."""

        if token is None or not self._observing:
            return
        started = time.perf_counter()
        op_id = token.op_id
        consumer_id = self._semantic_id(op_id.scope, op_id.phase, f"{op_id.role}_consumer")
        self._record(op_id, TimelineMarker.END, stream)
        self._record(op_id, TimelineMarker.CONSUMER_RESUME, stream)
        self._record_point(consumer_id, stream)
        self._record(op_id, TimelineMarker.RELEASE, stream)
        if token.nvtx_range_id is not None:
            self._nvtx_range_end(token.nvtx_range_id)
            token.nvtx_range_id = None
        self._add_overhead(started)

    @property
    def active(self) -> bool:
        """Whether iteration boundaries are still needed for discovery/profile/compile."""

        return not self._disabled and self._plan is None

    @property
    def hooks_enabled(self) -> bool:
        """Whether semantic hook sites should evaluate and publish metadata."""

        return self._observing

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Current machine-readable runtime diagnostics."""

        graph = self.session.graph if self._graph_frozen else None
        missing_samples = {}
        if graph is not None:
            missing_samples = {
                op_id.stable_key: self.config.profile_iters
                - len(self.session.telemetry.samples(op_id))
                for op_id in graph.operations
                if len(self.session.telemetry.samples(op_id)) < self.config.profile_iters
            }
        return {
            "requested_mode": self.mode.value,
            "effective_mode": self.effective_mode.value,
            "enforcement_active": False,
            "fallback_reason": self._fallback_reason,
            "errors": list(self._errors),
            "rank": self._rank,
            "world_size": self._world_size,
            "hostname": socket.gethostname(),
            "runtime_signature": self.config.runtime_signature,
            "warmup_iterations_observed": min(
                self._observed_iterations, self.config.warmup_iters
            ),
            "profile_iterations_started": self._profile_started,
            "profile_iterations_completed": len(self._profile_completed),
            "skipped_profile_iterations": self._skipped_profile_iterations,
            "pending_event_iterations": self.session.recorder.pending_iterations,
            "max_pending_event_iterations": self._max_pending_iterations,
            "graph_fingerprint": graph.fingerprint if graph is not None else None,
            "graph_operations": len(graph.operations) if graph is not None else 0,
            "graph_dependencies": len(graph.dependencies) if graph is not None else 0,
            "graph_acyclic": graph is not None,
            "graph_build_us": self._graph_build_us,
            "plan_fingerprint": self._plan.fingerprint if self._plan is not None else None,
            "plan_build_us": self._plan_build_us,
            "plan_compile_us": self._plan_compile_us,
            "plan_validation_us": self._plan_validation_us,
            "telemetry_consensus_us": self._telemetry_consensus_us,
            "profile_readiness_us": self._profile_readiness_us,
            "profile_readiness_rounds": self._profile_readiness_rounds,
            "plan_consensus": self._plan_consensus,
            "consensus_group_ranks": list(self._consensus_group_ranks),
            "consensus_samples_per_operation": self._consensus_samples_per_operation,
            "control_group_backend": self._control_group_backend,
            "plan_feasible": (
                self._plan.diagnostics.is_feasible if self._plan is not None else None
            ),
            "missing_telemetry_samples": missing_samples,
            "telemetry_complete": graph is not None and not missing_samples,
            "setup_cpu_overhead_us": self._setup_overhead_us,
            "hook_cpu_overhead_us": self._hook_overhead_us,
            "control_plane_cpu_overhead_us": self._control_plane_overhead_us,
            "total_planner_cpu_overhead_us": (
                self._setup_overhead_us
                + self._hook_overhead_us
                + self._control_plane_overhead_us
            ),
            "dynamic_payload_bytes": self._dynamic_payload_summary(),
            "gtp_ag_ready_sources": {
                op_id.stable_key: source
                for op_id, source in sorted(
                    self._gtp_ready_sources.items(),
                    key=lambda item: item[0].stable_key,
                )
            },
            "missing_forward_param_ready_scopes": sorted(
                self._missing_forward_param_ready_scopes
            ),
            "symmetric_metadata": {
                "status": (
                    "missing_from_hybridep_api"
                    if self._ep_symmetric_metadata_missing
                    else "not_observed"
                ),
                "operations": sorted(
                    op_id.stable_key for op_id in self._ep_symmetric_metadata_missing
                ),
            },
            "enforcement_ready": False,
        }

    @property
    def _observing(self) -> bool:
        return (
            self._active_iteration is not None
            and not self._disabled
            and (
                self._discovering
                or self._recording
            )
        )

    def _register_compute(self, op_id: SemanticOpId) -> None:
        self._register_operation(OperationSpec(op_id=op_id, kind=OperationKind.COMPUTE))

    def _register_operation(self, operation: OperationSpec) -> None:
        if self._disabled:
            return
        self._iteration_operations.add(operation.op_id)
        try:
            if self._graph_frozen:
                prior = self.session.graph.operations.get(operation.op_id)
                if prior != operation:
                    self._disable(
                        f"operation metadata changed for {operation.op_id.stable_key}: "
                        f"expected={prior!r}, observed={operation!r}"
                    )
            else:
                self.session.register_operation(operation)
        except Exception as exc:
            self._disable(
                f"operation registration failed for {operation.op_id.stable_key}: "
                f"{type(exc).__name__}: {exc}"
            )

    def _register_dependency(
        self,
        src: SemanticOpId,
        dst: SemanticOpId,
        kind: DependencyKind = DependencyKind.DATA,
    ) -> None:
        if self._disabled:
            return
        dependency = Dependency(src=src, dst=dst, kind=kind)
        try:
            if self._graph_frozen:
                if dependency not in self.session.graph.dependencies:
                    self._disable(
                        "semantic dependency changed during profile: "
                        f"{src.stable_key} -> {dst.stable_key} ({kind.value})"
                    )
            else:
                self.session.add_dependency(src, dst, kind)
        except Exception as exc:
            self._disable(
                f"dependency registration failed for {src.stable_key} -> "
                f"{dst.stable_key}: {type(exc).__name__}: {exc}"
            )

    def _record(
        self,
        op_id: SemanticOpId,
        marker: TimelineMarker,
        stream: object | None,
    ) -> None:
        if not self._recording or self._disabled:
            return
        try:
            self.session.record(op_id, marker, stream)
        except Exception as exc:
            self._disable(
                f"telemetry marker failed for {op_id.stable_key}/{marker.value}: "
                f"{type(exc).__name__}: {exc}"
            )

    def _record_point(self, op_id: SemanticOpId, stream: object | None) -> None:
        self._record(op_id, TimelineMarker.START, stream)
        self._record(op_id, TimelineMarker.END, stream)

    def _record_ready_trigger(self, op_id: SemanticOpId, trigger: Trigger) -> None:
        if not self._recording or self._disabled:
            return
        try:
            if trigger.kind is TriggerKind.WINDOW_START:
                self.session.alias_origin(op_id, TimelineMarker.READY)
            elif trigger.kind is TriggerKind.OP_END:
                assert trigger.op_id is not None
                self.session.alias_marker(
                    op_id,
                    TimelineMarker.READY,
                    trigger.op_id,
                    TimelineMarker.END,
                )
            else:
                raise ValueError(
                    f"unsupported GTP readiness trigger {trigger.kind.value}"
                )
        except Exception as exc:
            self._disable(
                f"ready-trigger telemetry failed for {op_id.stable_key}: "
                f"{type(exc).__name__}: {exc}"
            )

    def _finish_active_compute(
        self, phase: Phase, microbatch: int, stream: object | None
    ) -> None:
        active = self._active_compute.pop((phase, microbatch), None)
        if active is None:
            return
        compute_id, ag_id = active
        self._record(compute_id, TimelineMarker.END, stream)
        self._record(ag_id, TimelineMarker.RELEASE, stream)

    def _finish_compute_id(self, compute_id: SemanticOpId, stream: object | None) -> bool:
        key = (compute_id.phase, compute_id.microbatch)
        active = self._active_compute.get(key)
        if active is None or active[0] != compute_id:
            return False
        self._finish_active_compute(compute_id.phase, compute_id.microbatch, stream)
        return True

    def _close_active_computes(self, stream: object | None) -> None:
        for phase, microbatch in list(self._active_compute):
            self._finish_active_compute(phase, microbatch, stream)

    def _gtp_ag_ids(
        self, scope: str, phase: Phase, role: str
    ) -> tuple[SemanticOpId, SemanticOpId, SemanticOpId]:
        normalized_scope = self._normalize_scope(scope)
        return (
            self._semantic_id(normalized_scope, phase, f"{role}_ag"),
            self._semantic_id(normalized_scope, phase, f"{role}_compute"),
            self._semantic_id(normalized_scope, phase, f"{role}_ag_input"),
        )

    def _rs_finalize_ids(self, scope: str) -> tuple[SemanticOpId | None, SemanticOpId]:
        normalized_scope = self._normalize_scope(scope)
        rs_id = self._rs_tokens.get((normalized_scope, self._microbatch))
        finalize_id = self._semantic_id(normalized_scope, Phase.BACKWARD, "wgrad_finalize")
        return rs_id, finalize_id

    def _semantic_id(self, scope: str, phase: Phase, role: str) -> SemanticOpId:
        return SemanticOpId(
            scope=scope,
            phase=phase,
            role=role,
            microbatch=self._microbatch,
        )

    def _gtp_ag_ready_trigger(
        self,
        *,
        ag_id: SemanticOpId,
        producer_id: SemanticOpId,
        phase: Phase,
        direction: str,
        parameter_scopes: tuple[str, ...],
        stream: object | None,
    ) -> tuple[Trigger, tuple[SemanticOpId, ...]]:
        if direction in ("backward", "recompute"):
            self._gtp_ready_sources[ag_id] = "backward_window"
            return Trigger.window_start(phase, self._microbatch), ()

        normalized_scopes = tuple(
            self._normalize_scope(scope) for scope in parameter_scopes
        )
        ready_ids = {
            self._param_scope_to_ready[scope]
            for scope in normalized_scopes
            if scope in self._param_scope_to_ready
        }
        missing = {
            scope
            for scope in normalized_scopes
            if scope not in self._param_scope_to_ready
        }
        if normalized_scopes and not missing and ready_ids:
            last_ready = max(
                ready_ids,
                key=lambda op_id: self._param_bucket_ready_order[op_id],
            )
            self._gtp_ready_sources[ag_id] = "ddp_param_bucket"
            return (
                Trigger.op_end(last_ready),
                tuple(sorted(ready_ids, key=lambda op_id: op_id.stable_key)),
            )

        self._missing_forward_param_ready_scopes.update(missing or normalized_scopes)
        self._register_compute(producer_id)
        self._record_point(producer_id, stream)
        self._gtp_ready_sources[ag_id] = "launch_site_fallback"
        return Trigger.op_end(producer_id), (producer_id,)

    def _tag_param_bucket_readiness(
        self,
        chunk: object,
        chunk_index: int,
        parameter_scopes: dict[object, str],
        include_chunk: bool,
    ) -> None:
        param_to_bucket = getattr(chunk, "param_to_bucket_group", None)
        wrapped_module = getattr(chunk, "module", None)
        if not param_to_bucket or wrapped_module is None:
            return

        bucket_groups = []
        seen = set()
        for bucket_group in param_to_bucket.values():
            if id(bucket_group) in seen:
                continue
            seen.add(id(bucket_group))
            scopes = tuple(
                sorted(
                    parameter_scopes[param]
                    for param in getattr(bucket_group, "params", ())
                    if param in parameter_scopes
                    and getattr(param, "is_gtp_weight_remat", False)
                )
            )
            if not scopes:
                continue
            bucket_groups.append((scopes, bucket_group))

        for bucket_index, (scopes, bucket_group) in enumerate(
            sorted(bucket_groups, key=lambda item: item[0])
        ):
            digest = hashlib.sha256("\n".join(scopes).encode("utf-8")).hexdigest()[:12]
            chunk_prefix = f"model_chunk_{chunk_index}." if include_chunk else ""
            bucket_group._runtime_comm_planner_scope = (
                f"{chunk_prefix}ddp_param_bucket_{bucket_index}_{digest}"
            )
            bucket_group._runtime_comm_planner_param_scopes = scopes

        def readiness_hook(module, *unused):
            del unused
            observed = set()
            for parameter in module.parameters(recurse=False):
                bucket_group = param_to_bucket.get(parameter)
                if bucket_group is None or id(bucket_group) in observed:
                    continue
                observed.add(id(bucket_group))
                scope = getattr(
                    bucket_group, "_runtime_comm_planner_scope", None
                )
                scopes = getattr(
                    bucket_group,
                    "_runtime_comm_planner_param_scopes",
                    (),
                )
                if scope and scopes:
                    self.param_bucket_ready(
                        scope, scopes, torch.cuda.current_stream()
                    )

        for module in wrapped_module.modules():
            self._param_readiness_hook_handles.append(
                module.register_forward_pre_hook(readiness_hook)
            )

    @staticmethod
    def _gtp_phase_role(direction: str) -> tuple[Phase, str]:
        if direction == "forward":
            return Phase.FORWARD, "fwd"
        if direction == "backward":
            return Phase.BACKWARD, "bwd"
        if direction == "recompute":
            return Phase.BACKWARD, "recompute"
        raise ValueError(f"unsupported GTP direction {direction!r}")

    @staticmethod
    def _normalize_scope(scope: str) -> str:
        normalized = scope.strip(".")
        for prefix in ("module.module.", "module."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        return normalized or "unresolved_operation"

    @staticmethod
    def _communicator_id(domain: str, size: int) -> str:
        return f"{domain}:size{size}"

    @staticmethod
    def _reusable_buffer_specs(
        buffers: tuple[tuple[str, int, int, int], ...],
    ) -> tuple[ReusableBufferSpec, ...]:
        return tuple(
            ReusableBufferSpec(
                arena=arena,
                slot=slot,
                capacity_bytes=capacity_bytes,
                generation=generation,
            )
            for arena, slot, capacity_bytes, generation in buffers
        )

    @staticmethod
    def _topology_equivalence_signature(
        runtime_signature: dict[str, Any]
    ) -> dict[str, Any]:
        """Remove rank-local coordinates while retaining pipeline-stage identity."""

        signature = dict(runtime_signature)
        coordinates = dict(signature.pop("parallel_coordinates", {}) or {})
        signature.pop("communicator_memberships", None)
        signature["pipeline_parallel_rank"] = coordinates.get(
            "pipeline_parallel_rank", 0
        )
        return signature

    def _next_sequence(self, communicator: str) -> int:
        sequence = self._communicator_sequences.get(communicator, 0)
        self._communicator_sequences[communicator] = sequence + 1
        return sequence

    def _record_dynamic_payload(self, op_id: SemanticOpId, payload_bytes: int) -> None:
        if self._active_iteration is None:
            return
        key = (self._active_iteration, op_id)
        if key in self._dynamic_payload_iterations:
            return
        self._dynamic_payload_iterations.add(key)
        samples = self._dynamic_payload_bytes.setdefault(op_id, [])
        if len(samples) < self.config.profile_iters + 1:
            samples.append(max(0, int(payload_bytes)))

    def _dynamic_payload_summary(
        self, op_id: SemanticOpId | None = None
    ) -> dict[str, Any] | None:
        def summarize(samples: list[int]) -> dict[str, Any]:
            return {
                "samples": list(samples),
                "count": len(samples),
                "min": min(samples),
                "max": max(samples),
            }

        if op_id is not None:
            samples = self._dynamic_payload_bytes.get(op_id)
            return summarize(samples) if samples else None
        return {
            item.stable_key: summarize(samples)
            for item, samples in sorted(
                self._dynamic_payload_bytes.items(), key=lambda pair: pair[0].stable_key
            )
            if samples
        }

    def _collect_completed(self) -> None:
        if not self._graph_frozen or self._disabled:
            return
        try:
            self._max_pending_iterations = max(
                self._max_pending_iterations, self.session.recorder.pending_iterations
            )
            completed = self.session.collect_completed()
            self._profile_completed.update(iteration for iteration, _ in completed)
            self._max_pending_iterations = max(
                self._max_pending_iterations, self.session.recorder.pending_iterations
            )
        except Exception as exc:
            self._disable(f"telemetry collection failed: {type(exc).__name__}: {exc}")

    def _maybe_compile(self) -> None:
        if self._plan is not None or self._plan_consensus is False:
            return
        if not self._profiles_ready_collectively():
            return
        graph = self.session.graph
        missing = [
            op_id
            for op_id in graph.operations
            if len(self.session.telemetry.samples(op_id)) < self.config.profile_iters
        ]
        if missing:
            self._disable(
                "insufficient telemetry for operations: "
                + ", ".join(op_id.stable_key for op_id in missing[:8])
            )
            return

        build_started = time.perf_counter()
        self._nvtx_push("runtime_comm_planner::compile")
        candidate: RuntimePlan | None = None
        local_error: str | None = None
        try:
            consensus_started = time.perf_counter()
            payloads = self._all_gather_object(
                {
                    "rank": self._rank,
                    "graph_fingerprint": graph.fingerprint,
                    "runtime_signature": self._runtime_signature_key,
                    "samples": self.session.telemetry.all_samples(),
                }
            )
            consensus_telemetry = self._build_consensus_telemetry(payloads)
            self._telemetry_consensus_us = (
                time.perf_counter() - consensus_started
            ) * 1.0e6

            compile_started = time.perf_counter()
            candidate = self.session.planner.compile(
                graph, consensus_telemetry, epoch=0
            )
            self._plan_compile_us = (time.perf_counter() - compile_started) * 1.0e6
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"

        try:
            validation_started = time.perf_counter()
            statuses = self._all_gather_object(
                {
                    "rank": self._rank,
                    "graph_fingerprint": graph.fingerprint,
                    "runtime_signature": self._runtime_signature_key,
                    "plan_fingerprint": (
                        candidate.fingerprint if candidate is not None else None
                    ),
                    "error": local_error,
                }
            )
            self._plan_validation_us = (
                time.perf_counter() - validation_started
            ) * 1.0e6
            equivalent_statuses = self._equivalent_payloads(statuses)
            errors = [
                f"rank {status['rank']}: {status['error']}"
                for status in equivalent_statuses
                if status["error"] is not None
            ]
            if errors:
                raise RuntimeError(
                    "topology-equivalent plan compilation failed: " + "; ".join(errors)
                )
            fingerprints = {
                status["plan_fingerprint"] for status in equivalent_statuses
            }
            if candidate is None or fingerprints != {candidate.fingerprint}:
                raise RuntimeError(
                    "topology-equivalent ranks produced different plan fingerprints: "
                    + ", ".join(sorted(str(item) for item in fingerprints))
                )

            self._plan_consensus = True
            self._plan = candidate
            self._dump_telemetry()
            self._dump_plan()
        except Exception as exc:
            self._plan_consensus = False
            self._disable(
                f"plan consensus failed: {type(exc).__name__}: {exc}"
            )
        finally:
            self._plan_build_us = (time.perf_counter() - build_started) * 1.0e6
            self._nvtx_pop()

    def _profiles_ready_collectively(self) -> bool:
        """Make every rank enter telemetry exchange at the same iteration boundary."""

        local_ready = (
            not self._disabled
            and self._graph_frozen
            and self._profile_started >= self.config.profile_iters
            and len(self._profile_completed) >= self.config.profile_iters
            and self.session.recorder.pending_iterations == 0
        )
        if self._world_size == 1:
            return local_ready
        if (
            self._observed_iterations
            < self.config.warmup_iters + self.config.profile_iters
        ):
            return False

        started = time.perf_counter()
        self._profile_readiness_rounds += 1
        try:
            statuses = self._all_gather_object(
                {
                    "rank": self._rank,
                    "runtime_signature": self._runtime_signature_key,
                    "graph_fingerprint": (
                        self.session.graph.fingerprint
                        if self._graph_frozen
                        else None
                    ),
                    "ready": local_ready,
                    "disabled": self._disabled,
                    "error": self._fallback_reason if self._disabled else None,
                    "profile_started": self._profile_started,
                    "profile_completed": len(self._profile_completed),
                    "pending_iterations": self.session.recorder.pending_iterations,
                }
            )
        except Exception as exc:
            self._plan_consensus = False
            self._disable(
                "profile-readiness consensus failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return False
        finally:
            self._profile_readiness_us += (
                time.perf_counter() - started
            ) * 1.0e6

        ranks = [int(status["rank"]) for status in statuses]
        if sorted(ranks) != list(range(self._world_size)):
            self._plan_consensus = False
            self._disable(
                "profile-readiness consensus returned invalid rank coverage: "
                f"{sorted(ranks)}"
            )
            return False

        failed = [
            status
            for status in statuses
            if status["disabled"] or status["error"] is not None
        ]
        if failed:
            details = "; ".join(
                f"rank {status['rank']}: {status['error'] or 'disabled'}"
                for status in failed
            )
            self._plan_consensus = False
            self._disable(
                "profile-readiness consensus observed a rank-local failure: "
                + details
            )
            return False
        if not all(status["ready"] for status in statuses):
            return False

        fingerprints_by_signature: dict[str, set[str | None]] = {}
        for status in statuses:
            fingerprints_by_signature.setdefault(
                status["runtime_signature"], set()
            ).add(status["graph_fingerprint"])
        mismatches = [
            (signature, fingerprints)
            for signature, fingerprints in fingerprints_by_signature.items()
            if len(fingerprints) != 1 or None in fingerprints
        ]
        if mismatches:
            details = "; ".join(
                f"{signature}: {sorted(str(item) for item in fingerprints)}"
                for signature, fingerprints in mismatches
            )
            self._plan_consensus = False
            self._disable(
                "topology-equivalent ranks produced different graph fingerprints: "
                + details
            )
            return False
        return True

    def _build_consensus_telemetry(
        self, payloads: list[dict[str, Any]]
    ) -> TelemetryStore:
        graph = self.session.graph
        expected_operations = frozenset(graph.operations)
        equivalent = self._equivalent_payloads(payloads)
        if not equivalent:
            raise RuntimeError("no topology-equivalent telemetry payloads were gathered")

        rank_samples: dict[int, tuple[OperationSample, ...]] = {}
        for payload in equivalent:
            rank = int(payload["rank"])
            if rank in rank_samples:
                raise RuntimeError(f"duplicate telemetry payload for rank {rank}")
            samples = tuple(payload["samples"])
            if any(not isinstance(sample, OperationSample) for sample in samples):
                raise TypeError(f"rank {rank} supplied an invalid telemetry sample")
            counts = Counter(sample.op_id for sample in samples)
            if frozenset(counts) != expected_operations:
                missing = sorted(
                    op_id.stable_key
                    for op_id in expected_operations.difference(counts)
                )
                extra = sorted(
                    op_id.stable_key
                    for op_id in counts.keys() - expected_operations
                )
                raise RuntimeError(
                    f"rank {rank} telemetry operation mismatch: "
                    f"missing={missing[:8]}, extra={extra[:8]}"
                )
            invalid_counts = {
                op_id.stable_key: count
                for op_id, count in counts.items()
                if count != self.config.profile_iters
            }
            if invalid_counts:
                raise RuntimeError(
                    f"rank {rank} telemetry sample-count mismatch: "
                    f"{dict(sorted(invalid_counts.items())[:8])}"
                )
            rank_samples[rank] = samples

        self._consensus_group_ranks = tuple(sorted(rank_samples))
        self._consensus_samples_per_operation = (
            len(rank_samples) * self.config.profile_iters
        )
        return TelemetryStore.merge_rank_samples(rank_samples)

    def _equivalent_payloads(
        self, payloads: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        graph_fingerprint = self.session.graph.fingerprint
        return sorted(
            (
                payload
                for payload in payloads
                if payload["graph_fingerprint"] == graph_fingerprint
                and payload["runtime_signature"] == self._runtime_signature_key
            ),
            key=lambda payload: int(payload["rank"]),
        )

    def _all_gather_object(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if self._world_size == 1:
            return [payload]
        if self._control_group is None:
            raise RuntimeError("planner control process group is unavailable")
        gathered: list[dict[str, Any] | None] = [None] * self._world_size
        torch.distributed.all_gather_object(
            gathered, payload, group=self._control_group
        )
        if any(item is None for item in gathered):
            raise RuntimeError("planner control collective returned an empty payload")
        return [item for item in gathered if item is not None]

    def _disable(self, reason: str) -> None:
        if not self._disabled:
            self._disabled = True
            self._fallback_reason = reason
            logger.warning("Runtime communication planner disabled: %s", reason)
        self._recording = False
        self._errors.append(reason)
        self._dump_status()

    def _dump_graph(self) -> None:
        if not self.config.dump_plan or not self._dump_this_rank:
            return
        graph = self.session.graph
        operations = []
        for op_id in sorted(graph.operations, key=lambda item: item.stable_key):
            spec = graph.operations[op_id]
            operations.append(
                {
                    "id": op_id.stable_key,
                    "kind": spec.kind.value,
                    "resources": sorted(spec.resources),
                    "bytes": spec.bytes,
                    "dynamic_payload_bytes": self._dynamic_payload_summary(op_id),
                    "communicator_id": spec.communicator_id,
                    "sequence": spec.sequence,
                    "ready_trigger": self._trigger_key(spec.ready_trigger),
                    "deadline_trigger": self._trigger_key(spec.deadline_trigger),
                    "release_trigger": self._trigger_key(spec.release_trigger),
                    "symmetric_buffer": (
                        {
                            "arena": spec.symmetric_buffer.arena,
                            "slot": spec.symmetric_buffer.slot,
                            "offset_bytes": spec.symmetric_buffer.offset_bytes,
                            "capacity_bytes": spec.symmetric_buffer.capacity_bytes,
                            "generation": spec.symmetric_buffer.generation,
                        }
                        if spec.symmetric_buffer is not None
                        else None
                    ),
                    "reusable_buffers": [
                        {
                            "arena": buffer.arena,
                            "slot": buffer.slot,
                            "capacity_bytes": buffer.capacity_bytes,
                            "generation": buffer.generation,
                        }
                        for buffer in spec.reusable_buffers
                    ],
                    "priority": spec.priority,
                }
            )
        self._write_json(
            "graph",
            {
                "fingerprint": graph.fingerprint,
                "operations": operations,
                "dependencies": [
                    {
                        "src": edge.src.stable_key,
                        "dst": edge.dst.stable_key,
                        "kind": edge.kind.value,
                    }
                    for edge in graph.dependencies
                ],
            },
        )

    def _dump_telemetry(self) -> None:
        if not self.config.dump_plan or not self._dump_this_rank:
            return
        rows = []
        for op_id in sorted(self.session.graph.operations, key=lambda item: item.stable_key):
            for sample in self.session.telemetry.samples(op_id):
                rows.append(
                    {
                        "op_id": op_id.stable_key,
                        "iteration": sample.iteration,
                        "ready_us": sample.ready_us,
                        "start_us": sample.start_us,
                        "end_us": sample.end_us,
                        "service_us": sample.service_us,
                        "queue_delay_us": sample.queue_delay_us,
                        "drain_us": sample.drain_us,
                        "drain_delay_us": sample.drain_delay_us,
                        "consumer_ready_us": sample.consumer_ready_us,
                        "consumer_resume_us": sample.consumer_resume_us,
                        "exposed_wait_us": sample.exposed_wait_us,
                        "deadline_slack_us": sample.deadline_slack_us,
                        "release_us": sample.release_us,
                        "lifetime_us": sample.lifetime_us,
                        "overlap_kinds": sorted(kind.value for kind in sample.overlap_kinds),
                    }
                )
        self._write_json(
            "telemetry",
            {
                "samples": rows,
                "dynamic_payload_bytes": self._dynamic_payload_summary(),
            },
        )

    def _dump_plan(self) -> None:
        if not self.config.dump_plan or not self._dump_this_rank or self._plan is None:
            return
        self._write_json(
            "plan",
            {
                "epoch": self._plan.epoch,
                "graph_fingerprint": self._plan.graph_fingerprint,
                "fingerprint": self._plan.fingerprint,
                "actions": [
                    {
                        "op_id": action.op_id.stable_key,
                        "kind": action.kind.value,
                        "issue_trigger": action.issue_trigger.stable_key,
                        "wait_for": [trigger.stable_key for trigger in action.wait_for],
                        "planned_start_us": action.planned_start_us,
                        "planned_end_us": action.planned_end_us,
                        "ready_us": action.ready_us,
                        "deadline_us": action.deadline_us,
                        "predicted_slack_us": action.predicted_slack_us,
                        "resources": sorted(action.resources),
                        "communicator_id": action.communicator_id,
                        "sequence": action.sequence,
                        "reusable_buffers": [
                            {
                                "arena": buffer.arena,
                                "slot": buffer.slot,
                                "capacity_bytes": buffer.capacity_bytes,
                                "generation": buffer.generation,
                            }
                            for buffer in action.reusable_buffers
                        ],
                        "buffer_release_us": action.buffer_release_us,
                    }
                    for action in self._plan.actions
                ],
                "diagnostics": {
                    "feasible": self._plan.diagnostics.is_feasible,
                    "serialized_resources": list(
                        self._plan.diagnostics.serialized_resources
                    ),
                    "deadline_misses": [
                        {
                            "op_id": miss.op_id.stable_key,
                            "deadline_us": miss.deadline_us,
                            "completion_us": miss.completion_us,
                            "lateness_us": miss.lateness_us,
                        }
                        for miss in self._plan.diagnostics.deadline_misses
                    ],
                },
            },
        )

    def _dump_status(self, *, force: bool = False) -> None:
        if not self._dump_this_rank:
            return
        diagnostics = self.diagnostics
        signature_payload = {
            key: value
            for key, value in diagnostics.items()
            if key
            not in {
                "setup_cpu_overhead_us",
                "hook_cpu_overhead_us",
                "control_plane_cpu_overhead_us",
                "total_planner_cpu_overhead_us",
                "warmup_iterations_observed",
            }
        }
        signature = json.dumps(
            signature_payload, sort_keys=True, separators=(",", ":"), default=str
        )
        if not force and signature == self._last_status_signature:
            return
        self._write_json("diagnostics", diagnostics)
        self._last_status_signature = signature

    def _write_json(self, stem: str, payload: dict[str, Any]) -> None:
        path = self._artifact_dir / f"rank{self._rank:05d}_{stem}.json"
        temporary = path.with_suffix(".json.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as output:
                json.dump(payload, output, indent=2, sort_keys=True)
                output.write("\n")
            os.replace(temporary, path)
        except OSError as exc:
            logger.warning("Could not write runtime planner artifact %s: %s", path, exc)

    def _rank_selected(self, value: str) -> bool:
        normalized = (value or "0").strip().lower()
        if normalized == "all":
            return True
        try:
            selected = {int(item) for item in normalized.split(",") if item.strip()}
        except ValueError:
            self._errors = [f"invalid validate_ranks value {value!r}"]
            return self._rank == 0
        return self._rank in selected

    @staticmethod
    def _distributed_rank() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    @staticmethod
    def _distributed_world_size() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1

    @staticmethod
    def _trigger_key(trigger: Trigger | None) -> str | None:
        return trigger.stable_key if trigger is not None else None

    def _add_overhead(self, started: float) -> None:
        self._hook_overhead_us += (time.perf_counter() - started) * 1.0e6

    def _add_setup_overhead(self, started: float) -> None:
        self._setup_overhead_us += (time.perf_counter() - started) * 1.0e6

    def _add_control_plane_overhead(self, started: float) -> None:
        self._control_plane_overhead_us += (
            time.perf_counter() - started
        ) * 1.0e6

    @staticmethod
    def _nvtx_push(label: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(label)

    @staticmethod
    def _nvtx_pop() -> None:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

    @staticmethod
    def _nvtx_range_start(label: str) -> int | None:
        if torch.cuda.is_available():
            return torch.cuda.nvtx.range_start(label)
        return None

    @staticmethod
    def _nvtx_range_end(range_id: int) -> None:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_end(range_id)

    @staticmethod
    def _mark(label: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.nvtx.mark(label)


_DISABLED_RUNTIME = _DisabledRuntimePlanner()
_RUNTIME: RuntimeCommunicationPlannerRuntime | _DisabledRuntimePlanner = _DISABLED_RUNTIME


def configure_runtime_comm_planner(
    config: RuntimePlannerConfig,
    model: object | None = None,
) -> RuntimeCommunicationPlannerRuntime | _DisabledRuntimePlanner:
    """Install the process-global runtime facade after distributed/model setup."""

    global _RUNTIME
    if config.mode is RuntimePlannerMode.OFF:
        _RUNTIME = _DISABLED_RUNTIME
    else:
        _RUNTIME = RuntimeCommunicationPlannerRuntime(config)
        if model is not None:
            _RUNTIME.tag_model(model)
    return _RUNTIME


def get_runtime_comm_planner() -> RuntimeCommunicationPlannerRuntime | _DisabledRuntimePlanner:
    """Return the process-global runtime facade."""

    return _RUNTIME


def reset_runtime_comm_planner() -> None:
    """Restore the no-op runtime; primarily used by focused tests."""

    global _RUNTIME
    _RUNTIME = _DISABLED_RUNTIME
