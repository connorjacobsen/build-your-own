"""Instructor reference for real-process distributed training exercises."""

import math
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint


def partition_indices(size, rank, world_size):
    """Strided indices rank,rank+world_size,... below size. No duplication or padding.
    Require size>=0, world_size>=1, 0<=rank<world_size. Return list[int].
    """
    if size < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid partition")
    return list(range(rank, size, world_size))


def shard_bounds(total, rank, world_size):
    """Balanced contiguous [start,end) ownership. First total%world_size ranks own one extra.
    Require total>=0,world_size>=1, valid rank. Empty shards allowed.
    """
    if total < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid shard")
    base, extra = divmod(total, world_size)
    start = rank * base + min(rank, extra)
    return start, start + base + (rank < extra)


def flatten_tensors(tensors):
    """Concatenate flattened nonempty list of same-dtype, same-device tensors into a new 1D
    tensor. Preserve autograd connectivity, tensor order and values. Zero-size tensors allowed.
    Reject empty list, mixed dtypes or mixed devices. Do not modify inputs.
    """
    if not tensors or any(
        t.dtype != tensors[0].dtype or t.device != tensors[0].device for t in tensors
    ):
        raise ValueError("Incompatible tensors")
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten(flat, shapes):
    """Return views into rank-one flat with specified shapes, in order. Require exact element
    count and all shape dimensions nonnegative. Scalar shape () consumes one element.
    Mutating a returned view must mutate the corresponding flat segment.
    """
    sizes = [math.prod(s) for s in shapes]
    if flat.ndim != 1 or any(d < 0 for s in shapes for d in s) or sum(sizes) != flat.numel():
        raise ValueError("Invalid layout")
    result = []
    offset = 0
    for shape, size in zip(shapes, sizes):
        result.append(flat[offset : offset + size].view(shape))
        offset += size
    return result


def accumulate_gradients(model, microbatches, loss_sum_fn):
    """Each microbatch is (x,y), nonempty and equal leading size. loss_sum_fn(model(x),y)
    returns SUM over examples. Clear grads once, backprop each loss/total_examples, return
    total detached loss/total_examples. Do not optimizer.step. Unequal microbatch sizes matter.
    """
    if not microbatches or any(len(x) == 0 or len(x) != len(y) for x, y in microbatches):
        raise ValueError("Invalid microbatches")
    total = sum(len(y) for _, y in microbatches)
    model.zero_grad(set_to_none=True)
    loss = 0.0
    for x, y in microbatches:
        value = loss_sum_fn(model(x), y)
        (value / total).backward()
        loss += float(value.detach())
    return loss / total


def weighted_allreduce(local_mean, local_count):
    """With initialized process group, return count-weighted global mean, leaving input intact.
    local_count>=0; zero count contributes zero even if its local mean is arbitrary.
    All ranks must call with equal tensor shape/dtype/device. Global count zero raises ValueError
    on all ranks. Counts travel as float64 tensors on the input device. Use SUM collectives.
    """
    if not dist.is_initialized() or local_count < 0:
        raise ValueError("Invalid group or count")
    value = local_mean.clone() * local_count if local_count else torch.zeros_like(local_mean)
    count = torch.tensor(float(local_count), dtype=torch.float64, device=value.device)
    dist.all_reduce(value)
    dist.all_reduce(count)
    if count.item() == 0:
        raise ValueError("No global examples")
    return value / count.to(value.dtype)


def distributed_step(model, optimizer, x, y):
    """One global mean-squared-error update with initialized group; model(x) and y both [N,1].
    Unequal local N, including zero, are allowed; total N must be positive. Every rank executes
    collectives in identical order. Sum local squared errors and gradients globally, divide by
    GLOBAL example count, then step. Return identical global mean loss on every rank.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(x)
    local = (output - y).square().sum()
    local.backward()
    count = torch.tensor(float(len(y)), device=x.device, dtype=torch.float64)
    dist.all_reduce(count)
    if count.item() == 0:
        raise ValueError("No examples")
    for p in model.parameters():
        dist.all_reduce(p.grad)
        p.grad.div_(count.to(p.grad.dtype))
    loss = local.detach().clone()
    dist.all_reduce(loss)
    loss.div_(count.to(loss.dtype))
    optimizer.step()
    return float(loss)


def shard_state(momentum, rank, world_size):
    """For a flat global momentum tensor return dict start,end,momentum containing only
    an owned clone of this rank's balanced contiguous shard. Do not retain global storage.
    """
    start, end = shard_bounds(momentum.numel(), rank, world_size)
    return dict(start=start, end=end, momentum=momentum[start:end].clone())


def sharded_momentum_step(parameters, state, lr, momentum=0.9):
    """Initialized group, identical parameters and already globally averaged grads on each rank.
    Keep ONLY a local momentum shard in state (initially {}). Update local shard using
    velocity=momentum*velocity+grad, param-=lr*velocity; all_gather padded equal-size shards
    and restore full replicated parameters. state keys start,end,momentum. No full optimizer
    state replication. This is an educational ZeRO-1-style step, not full parameter sharding.
    """
    flat = flatten_tensors([p.detach() for p in parameters])
    grads = flatten_tensors([p.grad for p in parameters])
    world = dist.get_world_size()
    rank = dist.get_rank()
    start, end = shard_bounds(flat.numel(), rank, world)
    if not state:
        state.update(start=start, end=end, momentum=torch.zeros_like(flat[start:end]))
    state["momentum"].mul_(momentum).add_(grads[start:end])
    piece = flat[start:end] - lr * state["momentum"]
    width = (flat.numel() + world - 1) // world
    padded = torch.zeros(width, device=flat.device, dtype=flat.dtype)
    padded[: end - start] = piece
    gathered = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded)
    full = torch.cat(
        [
            part[
                : shard_bounds(flat.numel(), r, world)[1] - shard_bounds(flat.numel(), r, world)[0]
            ]
            for r, part in enumerate(gathered)
        ]
    )
    with torch.no_grad():
        for p, v in zip(parameters, unflatten(full, [tuple(p.shape) for p in parameters])):
            p.copy_(v)


def unscale_gradients(parameters, scale):
    """Positive finite scale. If any present gradient is nonfinite, return False and change
    nothing. Otherwise divide all present grads by scale and return True. Ignore None grads.
    This stage is local; collective_finite later coordinates the skip decision across ranks.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid scale")
    ps = [p for p in parameters if p.grad is not None]
    if any(not torch.isfinite(p.grad).all() for p in ps):
        return False
    for p in ps:
        p.grad.div_(scale)
    return True


def checkpointed(function, *args):
    """Call PyTorch non-reentrant activation checkpointing, preserving RNG state.
    Forward outputs and backward gradients must match ordinary execution, while the function
    executes again during backward when intermediates are required. This is activation
    recomputation, not serialization of model weights. Use the framework checkpoint primitive.
    """
    return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=True)


def consolidate_shards(shards, total):
    """Reconstruct 1D momentum from dicts start,end,momentum. Input order arbitrary.
    Validate gap-free, nonoverlapping coverage [0,total), matching slice lengths and rank-one
    tensors BEFORE concatenation. Empty slices allowed. Require at least one shard.
    Return new owned tensor; reject malformed layouts with ValueError.
    """
    if not shards or total < 0:
        raise ValueError("Invalid shards")
    ordered = sorted(shards, key=lambda s: (s["start"], s["end"]))
    cursor = 0
    for shard in ordered:
        start, end = shard["start"], shard["end"]
        data = shard["momentum"]
        if (
            start != cursor
            or end < start
            or end > total
            or data.ndim != 1
            or data.numel() != end - start
        ):
            raise ValueError("Invalid shard coverage")
        cursor = end
    if cursor != total:
        raise ValueError("Missing shard")
    return torch.cat([s["momentum"] for s in ordered])


def collective_finite(parameters):
    """Initialized group and nonempty parameter list all on one device. Return bool true
    on ALL ranks iff every present gradient on EVERY rank is finite. None is allowed.
    Do not change gradients. Reduce a device-local int flag using MIN, so one bad rank
    makes the same skip decision everywhere. Call before any optimizer update.
    """
    good = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
    flag = torch.tensor(int(good), dtype=torch.int32, device=parameters[0].device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())
