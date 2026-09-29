# 09. Update sharded momentum and gather weights

State ownership and parameter ownership need not be the same. Here every rank has the full parameters and gradients, but retains momentum only for its slice. Each rank updates its portion, then gathers updated weights so replicas agree. This resembles the basic idea of optimizer-state sharding; it does not yet implement full parameter or gradient sharding.

## Implementation contract

Work in `src/learner/api.py`. Implement **sharded_momentum_step**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`sharded_momentum_step`**

Initialized group, identical parameters and already globally averaged grads on each rank.
Keep ONLY a local momentum shard in state (initially {}). Update local shard using
velocity=momentum*velocity+grad, param-=lr*velocity; all_gather padded equal-size shards
and restore full replicated parameters. state keys start,end,momentum. No full optimizer
state replication. This is an educational ZeRO-1-style step, not full parameter sharding.

Use equal-size padded all_gather payloads; trim each rank according to its true bounds. Momentum starts at zero.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 9 --only
uv run course check 9
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 9 --level 1`.

## Explain and investigate

What memory still scales with total model size on every rank?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
