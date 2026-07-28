# Runtime communication planner draft

This package is the shadow-mode foundation for planning GTP, EP, and later DP
communication from eager runtime observations.

It deliberately separates four concerns:

1. `graph.py` — rank-stable semantic operations and correctness dependencies.
2. `telemetry.py` — non-synchronizing CUDA-event samples and bounded statistics.
3. `planner.py` — deterministic resource-aware scheduling and trigger-plan output.
4. `session.py` — the eager discovery/profile/compile lifecycle used by integration hooks.

The production GTP launch policy is unchanged in `off` and `shadow` modes.
`runtime.py` connects the session to eager GTP AG/RS and HybridEP
dispatch/combine semantic sites, publishes iteration and microbatch context,
and writes rank-local validation artifacts. It does not issue collectives or
change stream selection.

After bounded profiling completes, topology-equivalent ranks exchange their
completed samples on a dedicated planner control process group. Every member
then compiles from the same rank-ordered aggregate and checks the resulting
plan fingerprint. A world-wide readiness exchange first ensures that no rank
enters telemetry collection while another rank still needs to run production
collectives to drain its final CUDA-event sample. This control-plane exchange
happens outside measured iterations and does not call a global CUDA
synchronization. Rank-local timing is retained in telemetry artifacts, but it
is not used independently to produce an enforceable schedule.

## Runtime arguments

- `--runtime-comm-planner-mode {off,shadow,enforce}` (default: `off`)
- `--runtime-comm-planner-warmup-iters N` (default: 2)
- `--runtime-comm-planner-profile-iters N` (default: 4)
- `--runtime-comm-planner-replan-interval N` (default: 0)
- `--runtime-comm-planner-log-dir PATH`
- `--runtime-comm-planner-dump-plan`
- `--runtime-comm-planner-validate-ranks all|RANK[,RANK...]`

`enforce` currently enters an explicit shadow fallback. Enforcement must not
be enabled until topology-equivalent ranks agree on graph and plan
fingerprints and all required symmetric-buffer metadata is available.
Periodic replanning is also not active in this first integration.

The current HybridEP API does not expose its symmetric arena, slot, offset, or
capacity to MCore. Shadow diagnostics report that missing metadata as an
enforcement blocker rather than inventing buffer ownership information.

Ordinary GTP AG and RS cache storage is represented separately as reusable
buffer arenas and slots. The cache uses Python object identity only to discover
which checked-out tickets alias the same physical allocation; artifacts expose
only deterministic logical arena and slot numbers. The planner reserves each
logical slot until the operation's consumer or gradient-finalization release
trigger. Cross-rank graph fingerprint validation is required before these
rank-local first-use ordinals can be trusted for enforcement.

Warmup iterations run with the production launch path unchanged. The final
warmup iteration is used to discover the steady-state semantic graph; earlier
iterations are burn-in for lazy GTP/FP8 initialization. Each profiled iteration
must reproduce that graph. HybridEP routed-token payload sizes are dynamic and
therefore are reported as bounded min/max samples rather than included in the
immutable graph fingerprint.

Communication deadlines use the operation's `consumer_ready` trigger, which is
recorded immediately before the production path waits for an output. The
dependent compute starts only after that wait returns. This keeps the natural
deadline distinct from a delayed compute start while an explicit DAG edge
still records the hard communication-to-consumer data dependency.

Asynchronous GTP completion is observed on a dedicated telemetry CUDA stream.
The stream calls `Work.wait()` immediately after issue and records `END` behind
the NCCL completion event; this does not block the CPU or replace the existing
production-stream wait. The later production wait records `DRAIN` separately.
Consequently, `service_us` measures device completion rather than the entire
prefetch-to-consumer lifetime, while `drain_delay_us` reports how long completed
work remained undrained.

Runtime diagnostics separate CPU cost into `setup_cpu_overhead_us` for model
tagging, `hook_cpu_overhead_us` for per-iteration semantic hooks and event
recording, and `control_plane_cpu_overhead_us` for completed-sample
collection, graph freezing, cross-rank consensus, plan compilation, and
artifact writes. The component-specific graph and plan timing fields are
subsets of the control-plane total. Diagnostic status is rewritten only when
planner state changes and once at training shutdown; it is not emitted on every
steady-state iteration.

## Minimal lifecycle

```python
session = RuntimePlanningSession(CudaEventRecorder())

session.register_operation(
    OperationSpec(
        op_id=ag_id,
        kind=OperationKind.GTP_DENSE_AG,
        ready_trigger=Trigger.window_start(Phase.FORWARD),
        deadline_trigger=Trigger.consumer_ready(ag_id),
        release_trigger=Trigger.op_end(gemm_id),
        resources=frozenset({"cross_domain_fabric", "comm_sm"}),
        communicator_id="dense_gtp",
        sequence=37,
        symmetric_buffer=SymmetricBufferSpec(
            arena="dense_gtp", slot=2, offset_bytes=128 << 20, capacity_bytes=8 << 20
        ),
    )
)
session.add_dependency(ag_id, gemm_id)
session.freeze_graph()

session.begin_iteration(iteration, main_stream)
session.record(ag_id, TimelineMarker.READY, main_stream)
session.record(ag_id, TimelineMarker.START, ag_stream)
work = launch_ag(async_op=True)
# Work.wait() on ag_stream establishes the dependency on NCCL's internal
# execution stream. END must follow that wait, not the host API return.
session.record(ag_id, TimelineMarker.CONSUMER_READY, main_stream)
with torch.cuda.stream(ag_stream):
    work.wait()
    session.record(ag_id, TimelineMarker.END, ag_stream)
session.record(ag_id, TimelineMarker.CONSUMER_RESUME, main_stream)
session.end_iteration(main_stream)

# Called later, after query() says the device events are complete.
session.collect_completed()
plan = session.compile(epoch=1)
```

The first compiler treats a shared resource as exclusive. This gives a
conservative GTP-versus-EP admission plan immediately. The operation,
telemetry, and plan schemas are designed so a learned overlap-cost model can
replace that policy without changing runtime hook sites.
