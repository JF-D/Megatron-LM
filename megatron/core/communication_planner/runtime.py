# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Eager CUDA profiler for the compact GTP execution model."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .model import (
    GTPCudaSample,
    GTPExecutionModel,
    GTPPhase,
    GTPProfileKey,
    GTPWorkKind,
)


class _CudaMarker(str, Enum):
    START = "start"
    END = "end"


@dataclass(frozen=True)
class GTPProfileToken:
    """Handle connecting one runtime hook sequence to one profile operation."""

    key: GTPProfileKey


@dataclass
class _CudaIteration:
    iteration: int
    origin: object
    end: object | None
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
            end=None,
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

        if self._active is None:
            raise RuntimeError("No CUDA-profile iteration is active")
        self._active.end = self._new_event(stream)
        self._pending.append(self._active)
        self._active = None

    def abort_iteration(self) -> None:
        """Discard an incomplete active iteration."""

        self._active = None

    def collect_completed(self) -> tuple[tuple[int, tuple[GTPCudaSample, ...]], ...]:
        """Collect completed event sets without calling synchronize."""

        completed = []
        remaining: deque[_CudaIteration] = deque()
        while self._pending:
            iteration = self._pending.popleft()
            events = [iteration.origin, iteration.end]
            events.extend(
                event for markers in iteration.markers.values() for event in markers.values()
            )
            if any(event is None or not event.query() for event in events):
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
        self._iteration_ordinal = 0
        self._profile_started = 0
        self._recording = False
        self._active_iteration: int | None = None
        self._counters: dict[tuple[str, GTPPhase, GTPWorkKind], int] = defaultdict(int)
        self._active_compute: dict[str, list[GTPProfileToken]] = defaultdict(list)
        self._current_orders: dict[GTPPhase, list[GTPProfileKey]] = defaultdict(list)
        self._reference_orders: dict[GTPPhase, tuple[GTPProfileKey, ...]] | None = None
        self._samples: list[GTPCudaSample] = []
        self._errors: list[str] = []
        self._hook_handles: list[object] = []
        self._parameters: dict[str, object] = {}
        self._dumped = False

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
        """Whether the bounded model artifact has been written."""

        return self._dumped

    @property
    def errors(self) -> tuple[str, ...]:
        """Non-fatal discovery and validation errors."""

        return tuple(self._errors)

    def attach_model(self, model: object) -> int:
        """Attach forward-end hooks to modules that directly own GTP parameters."""

        chunks = list(model) if isinstance(model, (list, tuple)) else [model]
        attached_modules = set()
        for chunk_index, chunk in enumerate(chunks):
            if not hasattr(chunk, "named_parameters") or not hasattr(chunk, "named_modules"):
                continue
            prefix = f"model_chunk_{chunk_index}." if len(chunks) > 1 else ""
            parameter_names = {
                id(parameter): f"{prefix}{name}" for name, parameter in chunk.named_parameters()
            }
            for _, module in chunk.named_modules():
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
                    scopes.append(scope)
                scopes = sorted(set(scopes))
                if not scopes or id(module) in attached_modules:
                    continue
                attached_modules.add(id(module))

                def forward_end_hook(unused_module, unused_inputs, unused_output, *, scopes=scopes):
                    del unused_module, unused_inputs, unused_output
                    stream = _current_cuda_stream()
                    for scope in scopes:
                        self.compute_end(scope, stream)

                self._hook_handles.append(module.register_forward_hook(forward_end_hook))
        return len(attached_modules)

    def begin_iteration(self, iteration: int, stream: object | None = None) -> None:
        """Begin discovery or one bounded CUDA-profile iteration."""

        if self._dumped:
            return
        if self._active_iteration is not None:
            raise RuntimeError(f"Iteration {self._active_iteration} is still active")
        self._active_iteration = iteration
        self._counters = defaultdict(int)
        self._active_compute = defaultdict(list)
        self._current_orders = defaultdict(list)
        self._recording = (
            self._iteration_ordinal >= self.config.warmup_iters
            and self._profile_started < self.config.profile_iters
        )
        if self._recording:
            self.recorder.begin_iteration(iteration, stream)
            self._profile_started += 1

    def end_iteration(self, stream: object | None = None) -> None:
        """Close one iteration and asynchronously collect prior CUDA samples."""

        if self._active_iteration is None:
            return
        if self._active_compute:
            unfinished = sorted(
                token.key.stable_key
                for tokens in self._active_compute.values()
                for token in tokens
            )
            self._errors.append(
                f"iteration {self._active_iteration} has unfinished compute operations: "
                f"{unfinished[:8]}"
            )
            if self._recording:
                self.recorder.abort_iteration()
        elif self._recording:
            self.recorder.end_iteration(stream)

        orders = {
            phase: tuple(order)
            for phase, order in self._current_orders.items()
            if order
        }
        if self._reference_orders is None:
            self._reference_orders = orders
        elif orders != self._reference_orders:
            self._errors.append(
                f"iteration {self._active_iteration} changed the GTP execution order"
            )

        self._active_iteration = None
        self._recording = False
        self._iteration_ordinal += 1
        self.collect_completed()

    def collect_completed(self) -> tuple[tuple[int, tuple[GTPCudaSample, ...]], ...]:
        """Collect completed CUDA samples and retain them for model construction."""

        completed = self.recorder.collect_completed()
        for _, samples in completed:
            self._samples.extend(samples)
        if self._profile_started >= self.config.profile_iters and not self.recorder.pending_iterations:
            self._dump_model()
        return completed

    def finalize(self, *, synchronize: bool = True) -> Path | None:
        """Collect final samples, optionally synchronizing once after training."""

        if synchronize and self.recorder.pending_iterations:
            import torch

            torch.cuda.synchronize()
        self.collect_completed()
        path = self._dump_model()
        self._remove_hooks()
        return path

    def ag_ready(
        self,
        scope: str,
        phase: GTPPhase,
    ) -> GTPProfileToken | None:
        """Create the AG operation semantically required by one consumer."""

        return self._new_token(scope, phase, GTPWorkKind.AG)

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
    ) -> GTPProfileToken | None:
        """Record dependent module compute start after its AG is available."""

        token = self._new_token(scope, phase, GTPWorkKind.COMPUTE)
        if token is None:
            return None
        self._current_orders[phase].append(token.key)
        self._active_compute[scope].append(token)
        if self._recording:
            self.recorder.record(token.key, _CudaMarker.START, stream)
        return token

    def compute_end(self, scope: str, stream: object | None = None) -> None:
        """Close the most recently started compute operation for ``scope``."""

        token = self._pop_active_compute(scope)
        if token is None:
            return
        if self._recording:
            self.recorder.record(token.key, _CudaMarker.END, stream)

    def rs_ready(
        self, scope: str, stream: object | None = None
    ) -> GTPProfileToken | None:
        """Close backward compute and create its asynchronous RS side branch."""

        token = self._pop_active_compute(scope, GTPPhase.BACKWARD)
        if token is None:
            # Embedding and other direct-gradient paths have no backward weight
            # materialization. Preserve their place in the chain as a point compute.
            token = self.compute_start(scope, GTPPhase.BACKWARD, stream)
            if token is None:
                return None
            token = self._pop_active_compute(scope, GTPPhase.BACKWARD)
            assert token is not None
        if self._recording:
            self.recorder.record(token.key, _CudaMarker.END, stream)
        rs_key = GTPProfileKey(
            scope=scope,
            phase=GTPPhase.BACKWARD,
            kind=GTPWorkKind.RS,
            occurrence=token.key.occurrence,
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
        )

    def _new_token(
        self, scope: str, phase: GTPPhase, kind: GTPWorkKind
    ) -> GTPProfileToken | None:
        if self._active_iteration is None:
            return None
        counter_key = (scope, phase, kind)
        occurrence = self._counters[counter_key]
        self._counters[counter_key] += 1
        return GTPProfileToken(GTPProfileKey(scope, phase, kind, occurrence))

    def _pop_active_compute(
        self, scope: str, phase: GTPPhase | None = None
    ) -> GTPProfileToken | None:
        tokens = self._active_compute.get(scope)
        if not tokens:
            return None
        index = next(
            (
                index
                for index in range(len(tokens) - 1, -1, -1)
                if phase is None or tokens[index].key.phase is phase
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

    def _dump_model(self) -> Path | None:
        if self._dumped or not self._samples or not self._reference_orders:
            return None
        model = self.build_model()
        payload = model.to_dict()
        chain_errors = self._chain_validation_errors(model.parameter_chains)
        self._errors.extend(error for error in chain_errors if error not in self._errors)
        payload["diagnostics"] = {
            "errors": list(self._errors),
            "profile_iterations_started": self._profile_started,
            "sample_count": len(self._samples),
            "timing_source": "cuda_events",
        }
        rank = _distributed_rank()
        path = self.config.log_dir / f"rank{rank:05d}_gtp_execution_model.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        self._dumped = True
        self._remove_hooks()
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
