# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CUDA-graph lifecycle support for Generalized Tensor Parallelism (GTP).

This module owns state that exists only for local CUDA-graph capture and replay:

* capture-local ownership of asynchronous GTP communication;
* alternating graph-memory lanes for bounded cross-graph ownership;
* routing graph-owned allocations into the active CUDA-graph memory pool.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class GTPGraphPoolLane:
    """One graph-memory lane reused only after its prior replay drains."""

    index: int
    mempool: object
    ready_event: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    rs_outputs: dict = field(default_factory=dict)
    has_pending_work: bool = False

    def wait_for_reuse(self, stream: torch.cuda.Stream) -> None:
        """Delay new writes until the previous graph and its side streams finish."""
        if self.has_pending_work:
            stream.wait_event(self.ready_event)

    def mark_reusable_after(self, stream: torch.cuda.Stream) -> None:
        """Publish lane availability at the tail of a graph replay."""
        self.ready_event.record(stream)
        self.has_pending_work = True


@dataclass
class GTPCaptureCommState:
    """Asynchronous GTP work issued while capturing one CUDA graph."""

    params: list = field(default_factory=list)
    ag_streams: list = field(default_factory=list)
    rs_streams: list = field(default_factory=list)
    _param_ids: set = field(default_factory=set)
    _ag_stream_ids: set = field(default_factory=set)
    _rs_stream_ids: set = field(default_factory=set)
    _rs_cache_buffer_params: dict = field(default_factory=dict)

    def register_comm(self, param, stream: torch.cuda.Stream, *, reduce_scatter: bool) -> None:
        """Record a parameter and side stream owned by this graph capture."""
        param_id = id(param)
        if param_id not in self._param_ids:
            self._param_ids.add(param_id)
            self.params.append(param)

        stream_id = id(stream)
        streams = self.rs_streams if reduce_scatter else self.ag_streams
        stream_ids = self._rs_stream_ids if reduce_scatter else self._ag_stream_ids
        if stream_id not in stream_ids:
            stream_ids.add(stream_id)
            streams.append(stream)

    def register_rs_cache_buffer(self, buffer: torch.Tensor, param) -> None:
        """Reject two parameters using one lane-local RS output in the same graph."""
        buffer_id = id(buffer)
        param_id = id(param)
        prior_param_id = self._rs_cache_buffer_params.get(buffer_id)
        if prior_param_id is not None and prior_param_id != param_id:
            raise RuntimeError(
                "One CUDA graph uses the same GTP lane-local RS output for multiple parameters"
            )
        self._rs_cache_buffer_params[buffer_id] = param_id


_ACTIVE_CAPTURE_COMM_STATE: Optional[GTPCaptureCommState] = None


def register_capture_comm(param, stream: torch.cuda.Stream, *, reduce_scatter: bool) -> None:
    """Register communication with the active capture, if one exists."""
    if _ACTIVE_CAPTURE_COMM_STATE is not None:
        _ACTIVE_CAPTURE_COMM_STATE.register_comm(param, stream, reduce_scatter=reduce_scatter)


def register_capture_rs_cache_buffer(buffer: torch.Tensor, param) -> None:
    """Register a lane-local RS output with the active capture, if one exists."""
    if _ACTIVE_CAPTURE_COMM_STATE is not None:
        _ACTIVE_CAPTURE_COMM_STATE.register_rs_cache_buffer(buffer, param)


@contextmanager
def track_gtp_capture_comms():
    """Track asynchronous GTP work owned by one CUDA-graph capture."""
    global _ACTIVE_CAPTURE_COMM_STATE

    if _ACTIVE_CAPTURE_COMM_STATE is not None:
        raise RuntimeError("Nested GTP CUDA-graph communication tracking is unsupported")

    state = GTPCaptureCommState()
    _ACTIVE_CAPTURE_COMM_STATE = state
    try:
        yield state
    finally:
        _ACTIVE_CAPTURE_COMM_STATE = None


_CG_MEMPOOL_DEVICE = None
_CG_MEMPOOL = None
_CG_MEMPOOL_LANE = None


def set_cuda_graph_mempool(device, mempool, lane=None) -> None:
    """Register the memory pool and optional lane used by the active graph capture."""
    global _CG_MEMPOOL_DEVICE, _CG_MEMPOOL, _CG_MEMPOOL_LANE
    _CG_MEMPOOL_DEVICE = device
    _CG_MEMPOOL = mempool
    _CG_MEMPOOL_LANE = lane


def get_cuda_graph_mempool_lane() -> Optional[GTPGraphPoolLane]:
    """Return the lane assigned to the graph currently being captured."""
    return _CG_MEMPOOL_LANE


@contextmanager
def cuda_graph_pool_allocation(enabled: bool):
    """Route allocations in this context into the registered CUDA-graph pool."""
    if _CG_MEMPOOL is None or not enabled:
        yield
        return

    torch._C._cuda_beginAllocateCurrentThreadToPool(_CG_MEMPOOL_DEVICE, _CG_MEMPOOL)
    try:
        yield
    finally:
        torch._C._cuda_endAllocateToPool(_CG_MEMPOOL_DEVICE, _CG_MEMPOOL)
