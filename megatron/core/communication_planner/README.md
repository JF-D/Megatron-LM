# GTP runtime CUDA model prototype

This package builds only the compact execution model needed by GTP's existing
count-based prefetch and reduce-scatter controls. It is not a general
communication scheduler.

## Model

Eager execution discovers the actual GTP parameter order and records
timing-enabled CUDA events on the streams that execute module compute, AG, and
RS:

```text
forward:   AG_i -> F_i -> F_(i+1)

backward:  BWD_AG_i -> B_i -> B_(i-1)
                         |
                         +-> RS_i
```

There is deliberately no `RS_i -> B_(i-1)` edge. RS is an asynchronous side
branch whose eventual consumer is gradient finalization, not the next backward
module.

The existing `GTPShardedParam.prev_w` / `next_w` links remain the executable
parameter chain. The profiler also records the observed forward and backward
orders and validates that they are stable across sampled iterations.

## CUDA boundaries

- AG/RS `START` and `END` events are recorded on their actual GTP communication
  streams around the collective enqueue. An event enqueued after an asynchronous
  collective completes only after the collective's device work.
- Forward compute starts after the gathered weight is available and ends in a
  forward hook on the smallest module that directly owns the GTP parameter.
- Backward compute starts after backward weight materialization and ends when
  the resulting wgrad reaches the RS callsite.

These are GTP consumer intervals, not a complete GPU-occupancy trace. Time
between consecutive intervals is unmodeled work and must not be interpreted as
GPU idle time.

Dependency edges come from these semantic callsites, so the profiler does not
spend extra CUDA events measuring redundant logical-ready timestamps.

The forward end hook is essential: it prevents the last weight, such as
`output_layer.weight`, from remaining active until the end of the training
iteration.

## Usage

Enable the bounded eager profile:

```text
--gtp-runtime-profile
--gtp-runtime-profile-warmup-iters 2
--gtp-runtime-profile-iters 4
--gtp-runtime-profile-log-dir /path/to/profile
```

Each rank writes:

```text
rankXXXXX_gtp_execution_model.json
rankXXXXX_gtp_execution_trace.json
```

The model artifact contains CUDA-duration summaries, observed phase order,
existing parameter chains, and only these dependency kinds:

- `compute_order`
- `ag_before_compute`
- `compute_before_rs`

The Chrome Trace artifact retains the sampled CUDA intervals for manual
boundary inspection. It renders only compute, AG, and RS, and draws only
`ag_before_compute` and `compute_before_rs` flows.

Profiling cost is bounded by the configured sample count. Each observed compute,
AG, or RS uses two timing events; raw CUDA kernels, CPU operators, and generic
resource-conflict nodes are not collected. Completed samples are reduced to
duration statistics, and all module hooks are removed after the artifact is
written.

The next step consumes this model to generate per-`GTPShardedParam` forward
prefetch counts, backward prefetch counts, and RS issue/hold counts.
