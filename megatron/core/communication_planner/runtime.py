# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Eager CUDA profiler for the compact GTP execution model."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .model import (
    GTPCudaSample,
    GTPCommDomain,
    GTPExecutionModel,
    GTPModuleInfo,
    GTPPhase,
    GTPProfileKey,
    GTPWorkKind,
)
from .trace import build_gtp_chrome_trace


class _CudaMarker(str, Enum):
    START = "start"
    END = "end"


class _RuntimeProfileState(str, Enum):
    DISCOVERY = "discovery"
    RECORDING = "recording"
    DRAINING = "draining"
    COMPLETE = "complete"


@dataclass(frozen=True)
class GTPProfileToken:
    """Handle connecting one runtime hook sequence to one profile operation."""

    key: GTPProfileKey


@dataclass
class _CudaIteration:
    iteration: int
    origin: object
    markers: dict[GTPProfileKey, dict[_CudaMarker, object]]


class GTPCudaEventRecorder:
    """Record timing-enabled CUDA events without synchronizing the training path."""

    def __init__(self, event_factory: Callable[[], object] | None = None) -> None:
        if event_factory is None:
            import torch

            event_factory = lambda: torch.cuda.Event(enable_timing=True)
        self._event_factory = event_factory
        self._active: _CudaIteration | None = None
        self._pending: deque[_CudaIteration] = deque()

    def begin_iteration(self, iteration: int, stream: object | None = None) -> None:
        """Start a profiled iteration with a common CUDA timing origin."""

        if self._active is not None:
            raise RuntimeError("A CUDA-profile iteration is already active")
        self._active = _CudaIteration(
            iteration=iteration,
            origin=self._new_event(stream),
            markers=defaultdict(dict),
        )

    def record(
        self,
        key: GTPProfileKey,
        marker: _CudaMarker,
        stream: object | None = None,
    ) -> None:
        """Record one operation marker on the stream that executes it."""

        if self._active is None:
            raise RuntimeError("begin_iteration must be called before record")
        markers = self._active.markers[key]
        if marker in markers:
            raise ValueError(f"Duplicate {marker.value} marker for {key.stable_key}")
        markers[marker] = self._new_event(stream)

    def end_iteration(self, stream: object | None = None) -> None:
        """Close and enqueue the current iteration for asynchronous collection."""

        del stream
        if self._active is None:
            raise RuntimeError("No CUDA-profile iteration is active")
        self._pending.append(self._active)
        self._active = None

    def abort_iteration(self) -> None:
        """Discard an incomplete active iteration."""

        self._active = None

    def discard(self, key: GTPProfileKey) -> None:
        """Discard markers for an intentionally unterminated operation."""

        if self._active is not None:
            self._active.markers.pop(key, None)

    def collect_completed(self) -> tuple[tuple[int, tuple[GTPCudaSample, ...]], ...]:
        """Collect completed event sets without calling synchronize."""

        completed = []
        remaining: deque[_CudaIteration] = deque()
        while self._pending:
            iteration = self._pending.popleft()
            events = [iteration.origin]
            events.extend(
                event for markers in iteration.markers.values() for event in markers.values()
            )
            if any(not event.query() for event in events):
                remaining.append(iteration)
                continue
            completed.append((iteration.iteration, self._samples(iteration)))
        self._pending = remaining
        return tuple(completed)

    @property
    def pending_iterations(self) -> int:
        """Number of closed iterations whose CUDA events are not all complete."""

        return len(self._pending)

    def _new_event(self, stream: object | None) -> object:
        event = self._event_factory()
        if stream is None:
            event.record()
        else:
            event.record(stream)
        return event

    @staticmethod
    def _samples(iteration: _CudaIteration) -> tuple[GTPCudaSample, ...]:
        def timestamp(event: object | None) -> float | None:
            if event is None:
                return None
            return float(iteration.origin.elapsed_time(event)) * 1000.0

        samples = []
        for key, markers in iteration.markers.items():
            if _CudaMarker.START not in markers or _CudaMarker.END not in markers:
                raise RuntimeError(f"{key.stable_key} is missing CUDA START or END")
            start_us = timestamp(markers[_CudaMarker.START])
            end_us = timestamp(markers[_CudaMarker.END])
            assert start_us is not None and end_us is not None
            samples.append(
                GTPCudaSample(
                    key=key,
                    iteration=iteration.iteration,
                    start_us=start_us,
                    end_us=end_us,
                )
            )
        return tuple(sorted(samples, key=lambda sample: sample.key.stable_key))


@dataclass(frozen=True)
class GTPRuntimeProfileConfig:
    """Configuration for bounded eager CUDA profiling."""

    warmup_iters: int = 2
    profile_iters: int = 4
    log_dir: Path = Path("gtp_runtime_profile")

    def __post_init__(self) -> None:
        object.__setattr__(self, "log_dir", Path(self.log_dir))
        if self.warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")
        if self.profile_iters <= 0:
            raise ValueError("profile_iters must be positive")


class GTPRuntimeProfiler:
    """Collect the minimal CUDA model needed by a count-based GTP plan."""

    def __init__(
        self,
        config: GTPRuntimeProfileConfig,
        recorder: GTPCudaEventRecorder | None = None,
    ) -> None:
        self.config = config
        self.recorder = recorder or GTPCudaEventRecorder()
        self._state = (
            _RuntimeProfileState.DISCOVERY
            if config.warmup_iters > 0
            else _RuntimeProfileState.RECORDING
        )
        self._iteration_ordinal = 0
        self._profile_started = 0
        self._recording = False
        self._active_iteration: int | None = None
        self._counters: dict[tuple[str, GTPPhase, GTPWorkKind], int] = defaultdict(int)
        self._active_compute: dict[str, list[GTPProfileToken]] = defaultdict(list)
        self._active_compute_element: GTPProfileToken | None = None
        self._pending_consumer_wait: dict[
            tuple[str, GTPPhase], list[GTPProfileToken]
        ] = defaultdict(list)
        self._pending_prefetch_issue_gap: dict[
            tuple[str, GTPPhase], list[GTPProfileToken]
        ] = defaultdict(list)
        self._current_orders: dict[GTPPhase, list[GTPProfileKey]] = defaultdict(list)
        self._reference_orders: dict[GTPPhase, tuple[GTPProfileKey, ...]] | None = None
        self._current_compute_element_targets: dict[
            GTPProfileKey, GTPProfileKey
        ] = {}
        self._reference_compute_element_targets: dict[
            GTPProfileKey, GTPProfileKey
        ] | None = None
        self._samples: list[GTPCudaSample] = []
        self._errors: list[str] = []
        self._hook_handles: list[object] = []
        self._parameters: dict[str, object] = {}
        self._communication_domains: dict[str, GTPCommDomain] = {}
        self._parameter_modules: dict[str, GTPModuleInfo] = {}
        self._dumped = False
        self._artifact_path: Path | None = None

    @property
    def active(self) -> bool:
        """Whether runtime hooks should report operations."""

        return self._active_iteration is not None

    @property
    def recording(self) -> bool:
        """Whether this iteration records timing-enabled CUDA events."""

        return self._recording

    @property
    def complete(self) -> bool:
        """Whether the bounded profiling lifecycle has finished."""

        return self._state is _RuntimeProfileState.COMPLETE

    @property
    def window_closed(self) -> bool:
        """Whether discovery and event recording have permanently stopped."""

        return self._state in {
            _RuntimeProfileState.DRAINING,
            _RuntimeProfileState.COMPLETE,
        }

    @property
    def errors(self) -> tuple[str, ...]:
        """Non-fatal discovery and validation errors."""

        return tuple(self._errors)

    def attach_model(self, model: object) -> int:
        """Attach compute-boundary hooks to modules that directly own GTP parameters."""

        chunks = list(model) if isinstance(model, (list, tuple)) else [model]
        attached_modules = set()
        for chunk_index, chunk in enumerate(chunks):
            if not hasattr(chunk, "named_parameters") or not hasattr(chunk, "named_modules"):
                continue
            prefix = f"model_chunk_{chunk_index}." if len(chunks) > 1 else ""
            parameter_names = {
                id(parameter): f"{prefix}{name}" for name, parameter in chunk.named_parameters()
            }
            named_modules = list(chunk.named_modules())
            module_names = {
                id(module): f"{prefix}{name}" for name, module in named_modules
            }
            module_by_parameter = _hybrid_module_info_by_parameter(
                named_modules,
                module_names,
            )
            for module_name, module in named_modules:
                scopes = []
                for parameter in module.parameters(recurse=False):
                    if not getattr(parameter, "is_distributed_weight", False):
                        continue
                    # A routed GroupedLinear exposes every expert shard as a
                    # parameter, but only expert 0 owns ``weight_list`` and
                    # executes the batched DistributedWeight protocol.
                    if (
                        getattr(parameter, "is_routed_expert", False)
                        and getattr(parameter, "expert_idx", None) != 0
                    ):
                        continue
                    scope = getattr(parameter, "_debug_name", "") or parameter_names.get(
                        id(parameter), ""
                    )
                    if not scope:
                        continue
                    self._parameters[scope] = parameter
                    domain = (
                        GTPCommDomain.EGTP
                        if getattr(parameter, "is_routed_expert", False)
                        else GTPCommDomain.GTP
                    )
                    previous_domain = self._communication_domains.setdefault(scope, domain)
                    if previous_domain is not domain:
                        raise ValueError(
                            f"GTP profile scope {scope!r} belongs to both "
                            f"{previous_domain.value} and {domain.value}"
                        )
                    qualified_module_name = f"{prefix}{module_name}".rstrip(".")
                    module_scope = qualified_module_name or scope.rsplit(".", 1)[0]
                    self._parameter_modules[scope] = module_by_parameter.get(
                        id(parameter),
                        GTPModuleInfo(
                            scope=module_scope or scope,
                            symbol=_infer_module_symbol(scope),
                        ),
                    )
                    scopes.append(scope)
                scopes = sorted(set(scopes))
                if not scopes or id(module) in attached_modules:
                    continue
                attached_modules.add(id(module))

                def forward_end_hook(unused_module, unused_inputs, unused_output, *, scopes=scopes):
                    del unused_module, unused_inputs, unused_output
                    stream = _current_cuda_stream()
                    for scope in scopes:
                        self.forward_compute_end(scope, stream)

                self._hook_handles.append(module.register_forward_hook(forward_end_hook))
                if getattr(module, "_gtp_runtime_profile_embedding", False):

                    def backward_start_hook(unused_module, unused_grad_output, *, scopes=scopes):
                        del unused_module, unused_grad_output
                        stream = _current_cuda_stream()
                        for scope in scopes:
                            consumer_wait = self.consumer_enter(
                                scope,
                                GTPPhase.BACKWARD,
                                stream,
                            )
                            issue_gap = self.weight_ready(consumer_wait, stream)
                            self.compute_start(
                                scope,
                                GTPPhase.BACKWARD,
                                stream,
                                issue_gap_token=issue_gap,
                            )

                    self._hook_handles.append(
                        module.register_full_backward_pre_hook(backward_start_hook)
                    )
        return len(attached_modules)

    def begin_iteration(self, iteration: int, stream: object | None = None) -> bool:
        """Begin one iteration inside the bounded discovery/profile window."""

        self.collect_completed()
        if self.window_closed:
            return False
        if self._active_iteration is not None:
            raise RuntimeError(f"Iteration {self._active_iteration} is still active")
        self._active_iteration = iteration
        self._counters = defaultdict(int)
        self._active_compute = defaultdict(list)
        self._active_compute_element = None
        self._pending_consumer_wait = defaultdict(list)
        self._pending_prefetch_issue_gap = defaultdict(list)
        self._current_orders = defaultdict(list)
        self._current_compute_element_targets = {}
        self._recording = self._state is _RuntimeProfileState.RECORDING
        if self._recording:
            self.recorder.begin_iteration(iteration, stream)
            self._profile_started += 1
        return True

    @contextmanager
    def profile_iteration(
        self, iteration: int, stream: object | None = None
    ) -> Iterator[bool]:
        """Scope profiler activation to one training iteration."""

        active = self.begin_iteration(iteration, stream)
        try:
            yield active
        finally:
            if active:
                self.end_iteration(stream)

    def end_iteration(self, stream: object | None = None) -> None:
        """Close one iteration and asynchronously collect prior CUDA samples."""

        if self._active_iteration is None:
            return
        unfinished_consumer_wait = [
            token.key.stable_key
            for tokens in self._pending_consumer_wait.values()
            for token in tokens
        ]
        unfinished_issue_gap = [
            token.key.stable_key
            for tokens in self._pending_prefetch_issue_gap.values()
            for token in tokens
        ]
        if self._active_compute or unfinished_consumer_wait or unfinished_issue_gap:
            unfinished = sorted(
                token.key.stable_key
                for tokens in self._active_compute.values()
                for token in tokens
            )
            unfinished.extend(sorted(unfinished_consumer_wait))
            unfinished.extend(sorted(unfinished_issue_gap))
            self._errors.append(
                f"iteration {self._active_iteration} has unfinished compute operations: "
                f"{unfinished[:8]}"
            )
            if self._recording:
                self.recorder.abort_iteration()
        else:
            if self._recording and self._active_compute_element is not None:
                # A compute element is defined only between two consumers. The final
                # consumer in an iteration deliberately has no right-hand boundary.
                self.recorder.discard(self._active_compute_element.key)
            if self._recording:
                self.recorder.end_iteration(stream)

        orders = {
            phase: tuple(order)
            for phase, order in self._current_orders.items()
            if order
        }
        consumer_count = sum(len(order) for order in orders.values())
        expected_elements = max(consumer_count - 1, 0)
        if len(self._current_compute_element_targets) != expected_elements:
            self._errors.append(
                f"iteration {self._active_iteration} captured "
                f"{len(self._current_compute_element_targets)} compute elements "
                f"for {consumer_count} consumers; expected {expected_elements}"
            )
        if self._reference_orders is None:
            self._reference_orders = orders
        elif orders != self._reference_orders:
            self._errors.append(
                f"iteration {self._active_iteration} changed the GTP execution order"
            )
        if self._reference_compute_element_targets is None:
            self._reference_compute_element_targets = dict(
                self._current_compute_element_targets
            )
        elif (
            self._current_compute_element_targets
            != self._reference_compute_element_targets
        ):
            self._errors.append(
                f"iteration {self._active_iteration} changed the GTP compute elements"
            )

        self._active_iteration = None
        self._recording = False
        self._iteration_ordinal += 1
        self._advance_window()
        self.collect_completed()

    def collect_completed(self) -> tuple[tuple[int, tuple[GTPCudaSample, ...]], ...]:
        """Collect completed CUDA samples and retain them for model construction."""

        completed = self.recorder.collect_completed()
        for _, samples in completed:
            self._samples.extend(samples)
        if (
            self._state is _RuntimeProfileState.DRAINING
            and not self.recorder.pending_iterations
        ):
            self._artifact_path = self._dump_model()
            self._state = _RuntimeProfileState.COMPLETE
        return completed

    def finalize(self, *, synchronize: bool = True) -> Path | None:
        """Collect final samples, optionally synchronizing once after training."""

        if self._active_iteration is not None:
            raise RuntimeError(
                f"Cannot finalize while iteration {self._active_iteration} is active"
            )
        self._close_window()
        if synchronize and self.recorder.pending_iterations:
            import torch

            torch.cuda.synchronize()
        self.collect_completed()
        return self._artifact_path

    def ag_ready(
        self,
        scope: str,
        phase: GTPPhase,
    ) -> GTPProfileToken | None:
        """Create the AG operation semantically required by one consumer."""

        return self._new_token(scope, phase, GTPWorkKind.AG)

    def consumer_enter(
        self,
        scope: str,
        phase: GTPPhase,
        stream: object | None = None,
    ) -> GTPProfileToken | None:
        """Start the exposed wait for one consumer's current weight."""

        token = self._new_token(scope, phase, GTPWorkKind.CONSUMER_WAIT)
        if token is None:
            return None
        if self._active_compute_element is not None:
            element = self._active_compute_element
            self._current_compute_element_targets[element.key] = token.key
            if self._recording:
                self.recorder.record(element.key, _CudaMarker.END, stream)
        self._active_compute_element = None
        self._pending_consumer_wait[(scope, phase)].append(token)
        if self._recording:
            self.recorder.record(token.key, _CudaMarker.START, stream)
        return token

    def weight_ready(
        self,
        consumer_wait_token: GTPProfileToken | None,
        stream: object | None = None,
    ) -> GTPProfileToken | None:
        """Close current-weight wait and start the future-prefetch issue gap."""

        if consumer_wait_token is None:
            return None
        self._remove_pending_consumer_wait(consumer_wait_token)
        if self._recording:
            self.recorder.record(
                consumer_wait_token.key,
                _CudaMarker.END,
                stream,
            )
        issue_gap = self._new_token(
            consumer_wait_token.key.scope,
            consumer_wait_token.key.phase,
            GTPWorkKind.PREFETCH_ISSUE_GAP,
        )
        if issue_gap is None:
            return None
        if issue_gap.key.occurrence != consumer_wait_token.key.occurrence:
            raise RuntimeError(
                "GTP consumer-wait and prefetch-issue occurrences diverged for "
                f"{consumer_wait_token.key.scope}"
            )
        key = (issue_gap.key.scope, issue_gap.key.phase)
        self._pending_prefetch_issue_gap[key].append(issue_gap)
        if self._recording:
            self.recorder.record(issue_gap.key, _CudaMarker.START, stream)
        return issue_gap

    def communication_start(
        self, token: GTPProfileToken | None, stream: object | None = None
    ) -> None:
        """Record AG/RS device-service start on its communication stream."""

        if token is not None and self._recording:
            self.recorder.record(token.key, _CudaMarker.START, stream)

    def communication_end(
        self, token: GTPProfileToken | None, stream: object | None = None
    ) -> None:
        """Record AG/RS device-service end on its communication stream."""

        if token is not None and self._recording:
            self.recorder.record(token.key, _CudaMarker.END, stream)

    def compute_start(
        self,
        scope: str,
        phase: GTPPhase,
        stream: object | None = None,
        *,
        issue_gap_token: GTPProfileToken | None = None,
    ) -> GTPProfileToken | None:
        """Close prefetch issue and start dependent module compute."""

        if self._active_iteration is None:
            return None
        if issue_gap_token is None:
            pending = self._pending_prefetch_issue_gap.get((scope, phase))
            issue_gap_token = pending[0] if pending else None
        if issue_gap_token is None:
            pending_wait = self._pending_consumer_wait.get((scope, phase))
            consumer_wait_token = (
                pending_wait[0]
                if pending_wait
                else self.consumer_enter(scope, phase, stream)
            )
            issue_gap_token = self.weight_ready(consumer_wait_token, stream)
        if issue_gap_token is None:
            return None
        self._remove_pending_prefetch_issue_gap(issue_gap_token)
        if self._recording:
            self.recorder.record(
                issue_gap_token.key,
                _CudaMarker.END,
                stream,
            )

        token = self._new_token(scope, phase, GTPWorkKind.COMPUTE)
        if token is None:
            return None
        element = GTPProfileToken(
            GTPProfileKey(
                scope=scope,
                phase=phase,
                kind=GTPWorkKind.COMPUTE_ELEMENT,
                occurrence=token.key.occurrence,
                domain=token.key.domain,
            )
        )
        self._current_orders[phase].append(token.key)
        self._active_compute[scope].append(token)
        if self._recording:
            self.recorder.record(token.key, _CudaMarker.START, stream)
            self.recorder.record(element.key, _CudaMarker.START, stream)
        self._active_compute_element = element
        return token

    def forward_compute_end(self, scope: str, stream: object | None = None) -> None:
        """Close forward or recompute work at the owning module's forward hook."""

        token = self._pop_active_compute(
            scope,
            allowed_phases=frozenset({GTPPhase.FORWARD, GTPPhase.RECOMPUTE}),
        )
        if token is None:
            return
        if self._recording:
            self.recorder.record(token.key, _CudaMarker.END, stream)

    def rs_ready(
        self, scope: str, stream: object | None = None
    ) -> GTPProfileToken | None:
        """Close backward compute and create its asynchronous RS side branch."""

        token = self._pop_active_compute(
            scope, allowed_phases=frozenset({GTPPhase.BACKWARD})
        )
        if token is None:
            self._errors.append(
                f"{scope} reached RS without a backward compute-start marker"
            )
            token = self.compute_start(scope, GTPPhase.BACKWARD, stream)
            if token is None:
                return None
            token = self._pop_active_compute(
                scope, allowed_phases=frozenset({GTPPhase.BACKWARD})
            )
            assert token is not None
        if self._recording:
            self.recorder.record(token.key, _CudaMarker.END, stream)
        rs_key = GTPProfileKey(
            scope=scope,
            phase=GTPPhase.BACKWARD,
            kind=GTPWorkKind.RS,
            occurrence=token.key.occurrence,
            domain=token.key.domain,
        )
        rs_token = GTPProfileToken(rs_key)
        return rs_token

    def build_model(self) -> GTPExecutionModel:
        """Build the compact execution graph from collected CUDA samples."""

        if not self._reference_orders:
            raise RuntimeError("No GTP execution order was discovered")
        return GTPExecutionModel(
            phase_orders=self._reference_orders,
            samples=self._samples,
            parameter_chains=self._parameter_chains(),
            parameter_modules=self._parameter_modules,
            compute_element_targets=self._reference_compute_element_targets,
        )

    def _new_token(
        self, scope: str, phase: GTPPhase, kind: GTPWorkKind
    ) -> GTPProfileToken | None:
        if self._active_iteration is None:
            return None
        counter_key = (scope, phase, kind)
        occurrence = self._counters[counter_key]
        self._counters[counter_key] += 1
        return GTPProfileToken(
            GTPProfileKey(
                scope=scope,
                phase=phase,
                kind=kind,
                occurrence=occurrence,
                domain=self._communication_domains.get(scope, GTPCommDomain.GTP),
            )
        )

    def _remove_pending_consumer_wait(self, token: GTPProfileToken) -> None:
        key = (token.key.scope, token.key.phase)
        tokens = self._pending_consumer_wait.get(key)
        if not tokens or token not in tokens:
            raise RuntimeError(
                f"Unknown GTP consumer-wait token {token.key.stable_key}"
            )
        tokens.remove(token)
        if not tokens:
            del self._pending_consumer_wait[key]

    def _remove_pending_prefetch_issue_gap(self, token: GTPProfileToken) -> None:
        key = (token.key.scope, token.key.phase)
        tokens = self._pending_prefetch_issue_gap.get(key)
        if not tokens or token not in tokens:
            raise RuntimeError(
                f"Unknown GTP prefetch-issue token {token.key.stable_key}"
            )
        tokens.remove(token)
        if not tokens:
            del self._pending_prefetch_issue_gap[key]

    def _pop_active_compute(
        self,
        scope: str,
        allowed_phases: frozenset[GTPPhase],
    ) -> GTPProfileToken | None:
        tokens = self._active_compute.get(scope)
        if not tokens:
            return None
        index = next(
            (
                index
                for index in range(len(tokens) - 1, -1, -1)
                if tokens[index].key.phase in allowed_phases
            ),
            None,
        )
        if index is None:
            return None
        token = tokens.pop(index)
        if not tokens:
            del self._active_compute[scope]
        return token

    def _parameter_chains(self) -> dict[str, tuple[str, ...]]:
        scope_by_parameter = {id(parameter): scope for scope, parameter in self._parameters.items()}
        visited = set()
        chains = {}
        heads = [
            (scope, parameter)
            for scope, parameter in self._parameters.items()
            if getattr(parameter, "prev_w", None) is None
        ]
        for head_scope, head in sorted(heads):
            scopes = []
            current = head
            while current is not None and id(current) not in visited:
                visited.add(id(current))
                scope = scope_by_parameter.get(id(current))
                if scope is None:
                    break
                scopes.append(scope)
                current = getattr(current, "next_w", None)
            chain_id = getattr(head, "chain_id", "gtp")
            chains[f"{chain_id}:{head_scope}"] = tuple(scopes)
        return chains

    def _advance_window(self) -> None:
        """Advance the bounded profiler after one observed training iteration."""

        total_iterations = self.config.warmup_iters + self.config.profile_iters
        if self._iteration_ordinal >= total_iterations:
            self._close_window()
        elif self._iteration_ordinal >= self.config.warmup_iters:
            self._state = _RuntimeProfileState.RECORDING

    def _close_window(self) -> None:
        """Stop semantic instrumentation while completed CUDA events drain."""

        if self._state in {
            _RuntimeProfileState.DRAINING,
            _RuntimeProfileState.COMPLETE,
        }:
            return
        self._state = _RuntimeProfileState.DRAINING
        self._remove_hooks()

    def _dump_model(self) -> Path | None:
        if self._dumped:
            return self._artifact_path
        if not self._samples or not self._reference_orders:
            return None
        model = self.build_model()
        payload = model.to_dict()
        chain_errors = self._chain_validation_errors(model.parameter_chains)
        self._errors.extend(error for error in chain_errors if error not in self._errors)
        payload["diagnostics"] = {
            "errors": list(self._errors),
            "warmup_iterations_configured": self.config.warmup_iters,
            "profile_iterations_configured": self.config.profile_iters,
            "iterations_observed": self._iteration_ordinal,
            "profile_iterations_started": self._profile_started,
            "sample_count": len(self._samples),
            "consumer_count": sum(
                len(order) for order in self._reference_orders.values()
            ),
            "compute_element_count": len(
                self._reference_compute_element_targets or {}
            ),
            "timing_source": "cuda_events",
            "communication_timing": "work_completion_fenced_on_comm_stream",
            "opportunity_intervals": [
                GTPWorkKind.CONSUMER_WAIT.value,
                GTPWorkKind.PREFETCH_ISSUE_GAP.value,
            ],
        }
        rank = _distributed_rank()
        path = self.config.log_dir / f"rank{rank:05d}_gtp_execution_model.json"
        trace_path = self.config.log_dir / f"rank{rank:05d}_gtp_execution_trace.json"
        payload["diagnostics"]["trace_file"] = trace_path.name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        with trace_path.open("w", encoding="utf-8") as stream:
            json.dump(
                build_gtp_chrome_trace(
                    self._samples,
                    model.dependencies,
                    rank=rank,
                    parameter_modules=model.parameter_modules,
                    compute_element_targets=model.compute_element_targets,
                ),
                stream,
                indent=2,
            )
            stream.write("\n")
        self._dumped = True
        self._artifact_path = path
        return path

    def _remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _chain_validation_errors(
        self, chains: Mapping[str, tuple[str, ...]]
    ) -> list[str]:
        if not chains:
            return ["No populated GTP prev_w/next_w chain was discovered"]

        errors = []
        forward_order = self._reference_orders.get(GTPPhase.FORWARD, ())
        backward_order = self._reference_orders.get(GTPPhase.BACKWARD, ())
        for chain_name, chain in chains.items():
            chain_set = set(chain)
            observed_forward = _first_scope_occurrences(forward_order, chain_set)
            if observed_forward != chain:
                errors.append(
                    f"{chain_name} differs from observed forward order: "
                    f"chain={chain}, observed={observed_forward}"
                )
            observed_backward = _first_scope_occurrences(backward_order, chain_set)
            if observed_backward and observed_backward != tuple(reversed(chain)):
                errors.append(
                    f"{chain_name} differs from observed backward order: "
                    f"chain={tuple(reversed(chain))}, observed={observed_backward}"
                )
        return errors


_RUNTIME_PROFILER: GTPRuntimeProfiler | None = None


def configure_gtp_runtime_profiler(
    config: GTPRuntimeProfileConfig,
    *,
    model: object | None = None,
    recorder: GTPCudaEventRecorder | None = None,
) -> GTPRuntimeProfiler:
    """Configure the process-global bounded GTP profiler."""

    global _RUNTIME_PROFILER
    profiler = GTPRuntimeProfiler(config, recorder)
    if model is not None:
        attached_modules = profiler.attach_model(model)
        if attached_modules == 0:
            raise RuntimeError("GTP runtime profiling found no GTP parameter-owning modules")
    _RUNTIME_PROFILER = profiler
    return profiler


def get_gtp_runtime_profiler() -> GTPRuntimeProfiler | None:
    """Return the configured profiler, or ``None`` when profiling is disabled."""

    return _RUNTIME_PROFILER


def get_active_gtp_runtime_profiler() -> GTPRuntimeProfiler | None:
    """Return the profiler only while a bounded semantic iteration is active."""

    profiler = _RUNTIME_PROFILER
    return profiler if profiler is not None and profiler.active else None


def reset_gtp_runtime_profiler() -> None:
    """Clear the process-global profiler. Intended for tests and model teardown."""

    global _RUNTIME_PROFILER
    if _RUNTIME_PROFILER is not None:
        _RUNTIME_PROFILER._remove_hooks()
    _RUNTIME_PROFILER = None


def _current_cuda_stream() -> object:
    import torch

    return torch.cuda.current_stream()


def _distributed_rank() -> int:
    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except (AttributeError, ImportError, RuntimeError):
        pass
    return 0


def _hybrid_module_info_by_parameter(
    named_modules: list[tuple[str, object]],
    module_names: Mapping[int, str],
) -> dict[int, GTPModuleInfo]:
    """Map parameters to the enclosing hybrid M/*/E layer."""

    result = {}
    for _, container in named_modules:
        layer_types = getattr(container, "layer_type_list", None)
        layers = getattr(container, "layers", None)
        if layer_types is None or layers is None:
            continue
        materialized_layers = list(layers)
        if len(layer_types) != len(materialized_layers):
            continue
        for symbol, layer in zip(layer_types, materialized_layers):
            module_scope = module_names.get(id(layer), "")
            if not module_scope or not hasattr(layer, "parameters"):
                continue
            layer_number = getattr(layer, "layer_number", None)
            info = GTPModuleInfo(
                scope=module_scope,
                symbol=str(symbol),
                layer_number=int(layer_number) if layer_number is not None else None,
            )
            for parameter in layer.parameters():
                result.setdefault(id(parameter), info)
    return result


def _infer_module_symbol(scope: str) -> str:
    if "embedding" in scope:
        return "embedding"
    if "output_layer" in scope:
        return "output"
    return "other"


def _first_scope_occurrences(
    order: tuple[GTPProfileKey, ...], scopes: set[str]
) -> tuple[str, ...]:
    seen = set()
    result = []
    for key in order:
        if key.scope in scopes and key.scope not in seen:
            seen.add(key.scope)
            result.append(key.scope)
    return tuple(result)
