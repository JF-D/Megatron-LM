# Runtime communication planner draft

This package is the shadow-mode foundation for planning GTP, EP, and later DP
communication from eager runtime observations.

It deliberately separates four concerns:

1. `graph.py` — rank-stable semantic operations and correctness dependencies.
2. `telemetry.py` — non-synchronizing CUDA-event samples and bounded statistics.
3. `planner.py` — deterministic resource-aware scheduling and trigger-plan output.
4. `session.py` — the eager discovery/profile/compile lifecycle used by integration hooks.

The existing GTP launch policy is unchanged. A later integration will call the
session from GTP AG/RS and HybridEP dispatch/combine semantic sites, validate
plan hashes across participating ranks, and enable `RuntimePlanExecutor` only
after shadow validation.

## Minimal lifecycle

```python
session = RuntimePlanningSession(CudaEventRecorder())

session.register_operation(
    OperationSpec(
        op_id=ag_id,
        kind=OperationKind.GTP_DENSE_AG,
        ready_trigger=Trigger.window_start(Phase.FORWARD),
        deadline_trigger=Trigger.op_start(gemm_id),
        release_trigger=Trigger.op_end(gemm_id),
        resources=frozenset({"cross_domain_fabric", "comm_sm"}),
        communicator_id="dense_gtp",
        sequence=37,
        symmetric_buffer=SymmetricBufferSpec(
            arena="dense_gtp", slot=2, offset_bytes=128 << 20, capacity_bytes=8 << 20
        ),
    )
)
session.freeze_graph()

session.begin_iteration(iteration, main_stream)
session.record(ag_id, TimelineMarker.READY, main_stream)
session.record(ag_id, TimelineMarker.START, ag_stream)
launch_ag()
session.record(ag_id, TimelineMarker.END, ag_stream)
session.end_iteration(main_stream)

# Called later, after query() says the device events are complete.
session.collect_completed()
plan = session.compile(epoch=1)
```

The first compiler treats a shared resource as exclusive. This gives a
conservative GTP-versus-EP admission plan immediately. The operation,
telemetry, and plan schemas are designed so a learned overlap-cost model can
replace that policy without changing runtime hook sites.
