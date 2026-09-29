# Public API and input domains

These are scaffold contracts, not implementations. Earlier stages remain dependencies of later stages.

## `partition_indices`

Strided indices rank,rank+world_size,... below size. No duplication or padding.
Require size>=0, world_size>=1, 0<=rank<world_size. Return list[int].

Signature: `partition_indices(size, rank, world_size)`

## `shard_bounds`

Balanced contiguous [start,end) ownership. First total%world_size ranks own one extra.
Require total>=0,world_size>=1, valid rank. Empty shards allowed.

Signature: `shard_bounds(total, rank, world_size)`

## `flatten_tensors`

Concatenate flattened nonempty list of same-dtype, same-device tensors into a new 1D
tensor. Preserve autograd connectivity, tensor order and values. Zero-size tensors allowed.
Reject empty list, mixed dtypes or mixed devices. Do not modify inputs.

Signature: `flatten_tensors(tensors)`

## `unflatten`

Return views into rank-one flat with specified shapes, in order. Require exact element
count and all shape dimensions nonnegative. Scalar shape () consumes one element.
Mutating a returned view must mutate the corresponding flat segment.

Signature: `unflatten(flat, shapes)`

## `accumulate_gradients`

Each microbatch is (x,y), nonempty and equal leading size. loss_sum_fn(model(x),y)
returns SUM over examples. Clear grads once, backprop each loss/total_examples, return
total detached loss/total_examples. Do not optimizer.step. Unequal microbatch sizes matter.

Signature: `accumulate_gradients(model, microbatches, loss_sum_fn)`

## `weighted_allreduce`

With initialized process group, return count-weighted global mean, leaving input intact.
local_count>=0; zero count contributes zero even if its local mean is arbitrary.
All ranks must call with equal tensor shape/dtype/device. Global count zero raises ValueError
on all ranks. Counts travel as float64 tensors on the input device. Use SUM collectives.

Signature: `weighted_allreduce(local_mean, local_count)`

## `distributed_step`

One global mean-squared-error update with initialized group; model(x) and y both [N,1].
Unequal local N, including zero, are allowed; total N must be positive. Every rank executes
collectives in identical order. Sum local squared errors and gradients globally, divide by
GLOBAL example count, then step. Return identical global mean loss on every rank.

Signature: `distributed_step(model, optimizer, x, y)`

## `shard_state`

For a flat global momentum tensor return dict start,end,momentum containing only
an owned clone of this rank's balanced contiguous shard. Do not retain global storage.

Signature: `shard_state(momentum, rank, world_size)`

## `sharded_momentum_step`

Initialized group, identical parameters and already globally averaged grads on each rank.
Keep ONLY a local momentum shard in state (initially {}). Update local shard using
velocity=momentum*velocity+grad, param-=lr*velocity; all_gather padded equal-size shards
and restore full replicated parameters. state keys start,end,momentum. No full optimizer
state replication. This is an educational ZeRO-1-style step, not full parameter sharding.

Signature: `sharded_momentum_step(parameters, state, lr, momentum=0.9)`

## `unscale_gradients`

Positive finite scale. If any present gradient is nonfinite, return False and change
nothing. Otherwise divide all present grads by scale and return True. Ignore None grads.
This stage is local; collective_finite later coordinates the skip decision across ranks.

Signature: `unscale_gradients(parameters, scale)`

## `checkpointed`

Call PyTorch non-reentrant activation checkpointing, preserving RNG state.
Forward outputs and backward gradients must match ordinary execution, while the function
executes again during backward when intermediates are required. This is activation
recomputation, not serialization of model weights. Use the framework checkpoint primitive.

Signature: `checkpointed(function, *args)`

## `consolidate_shards`

Reconstruct 1D momentum from dicts start,end,momentum. Input order arbitrary.
Validate gap-free, nonoverlapping coverage [0,total), matching slice lengths and rank-one
tensors BEFORE concatenation. Empty slices allowed. Require at least one shard.
Return new owned tensor; reject malformed layouts with ValueError.

Signature: `consolidate_shards(shards, total)`

## `collective_finite`

Initialized group and nonempty parameter list all on one device. Return bool true
on ALL ranks iff every present gradient on EVERY rank is finite. None is allowed.
Do not change gradients. Reduce a device-local int flag using MIN, so one bad rank
makes the same skip decision everywhere. Call before any optimizer update.

Signature: `collective_finite(parameters)`
