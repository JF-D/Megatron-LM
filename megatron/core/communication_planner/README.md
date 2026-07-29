# GTP runtime CUDA model prototype

This package builds only the compact execution model needed by GTP's existing
count-based prefetch and reduce-scatter controls. It is not a general
communication scheduler.

## Model

Eager execution discovers the actual GTP parameter order and records
timing-enabled CUDA events on the streams that execute module compute, AG, and
RS:

```text
consumer i:       AG_i -> consumer_wait_i -> prefetch_issue_gap_i -> compute_i
                                                                  \
                                                                   +-> compute_element_i
                                                                         |
                                                                         +-> consumer_wait_(i+1)

backward:         BWD_AG_i -> B_i
                                |
                                +-> RS_i
```

There is deliberately no `RS_i -> B_(i-1)` edge. RS is an asynchronous side
branch whose eventual consumer is gradient finalization, not the next backward
module.

`compute_element_i` is the non-overlapping CUDA interval from the point where
consumer `i` has issued its planned future prefetches until consumer `i+1`
enters its weight-consumption path. It therefore includes parameterless
activations, mixer/attention cores, routing, residuals, normalization, and
other coarse work between two `GTPShardedParam` consumption points.

`consumer_wait_i` measures only the exposed interval from entering the
consumer to obtaining its current weight. `prefetch_issue_gap_i` begins as soon
as that current weight is ready and ends when planned future prefetches and GTP
bookkeeping have been issued. The latter is a CUDA-visible compute-stream
bubble, even when its root cause is host launch overhead.

The existing `GTPShardedParam.prev_w` / `next_w` links remain the executable
parameter chain. The profiler also records the observed forward and backward
orders and validates that they are stable across sampled iterations.

Every operation is classified as either the dense `GTP` communication domain or
the routed-expert `EGTP` domain when the model hooks are attached. The domain is
part of the operation identity and dependency matching. Chrome Trace therefore
uses separate AG and RS lanes for GTP and EGTP even when their CUDA intervals
overlap.

Hybrid layers are labeled from `HybridStack.layer_type_list`, so parameters and
compute elements carry their enclosing `M`, `*`, or `E` module, module scope,
and layer number. The runtime-discovered parameter order remains authoritative;
the symbols provide coarse boundaries and readable grouping rather than a
hard-coded execution order.

## CUDA boundaries

- During recorded iterations, asynchronous TE/c10d AG and RS work is fenced on
  its dedicated GTP communication stream with `Work.wait()` before the timing
  END event is recorded. This orders END after actual backend completion
  without making the compute stream wait. Symmetric communication should
  provide the equivalent stream-ordered ready event directly.
- Consumer-enter is recorded inside `GTPShardedParam.all_gather_and_prefetch`
  or `all_gather_and_prefetch_bwd` before the current weight is consumed.
- Weight-ready is recorded immediately after the current prefetched weight is
  obtained or an on-demand AG completes, before any future AG is issued.
- Compute starts after future prefetches and GTP bookkeeping have been issued.
  This covers TE, legacy MCore linears such as the output layer, and embedding
  without separate forward callsite instrumentation.
- Forward compute starts after the gathered weight is available and ends in a
  forward hook on the smallest module that directly owns the GTP parameter.
- Backward compute starts after backward weight materialization and ends when
  the resulting wgrad reaches the RS callsite.
- The embedding path is handled explicitly because it bypasses the generic TE
  backward materialization callback: a module backward pre-hook starts its
  direct-gradient compute, and the embedding RS callsite closes it. Embedding
  has no backward AG.

These are scheduling intervals, not a complete GPU-occupancy trace. Coarse
compute elements account for the CUDA critical-path time between consecutive
GTP consumers without naming every kernel.

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
- `consumer_wait_before_prefetch_issue`
- `prefetch_issue_before_compute`
- `compute_before_consumer_wait`
- `compute_before_rs`

The Chrome Trace artifact retains the sampled CUDA intervals for manual
boundary inspection. It renders coarse compute elements, consumer wait,
prefetch-issue gaps, direct GTP consumer compute, AG, and RS. It draws only
`ag_before_compute` and `compute_before_rs` flows to keep the view readable.

Profiling cost is bounded by the configured sample count. Each observed compute,
compute element, opportunity interval, AG, or RS uses two timing events; the
final consumer has no compute-element end and its provisional event is
discarded. Raw CUDA kernels, CPU operators, and generic resource-conflict nodes
are not collected. Completed samples are reduced to duration statistics, and
all module hooks are removed after the artifact is written.

The next step consumes this model to generate per-`GTPShardedParam` forward
prefetch counts, backward prefetch counts, and RS issue/hold counts.
