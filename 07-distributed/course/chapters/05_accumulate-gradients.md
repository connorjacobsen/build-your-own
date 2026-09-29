# 05. Accumulate uneven microbatches

Gradient accumulation can reproduce a larger batch without holding every activation simultaneously. The accumulated objective must use the same denominator as the large batch. Dividing each microbatch mean by the number of microbatches is wrong when their sizes differ. Backpropagating sums divided by the total example count gives the desired weighting.

## Implementation contract

Work in `src/learner/api.py`. Implement **accumulate_gradients**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`accumulate_gradients`**

Each microbatch is (x,y), nonempty and equal leading size. loss_sum_fn(model(x),y)
returns SUM over examples. Clear grads once, backprop each loss/total_examples, return
total detached loss/total_examples. Do not optimizer.step. Unequal microbatch sizes matter.

Clear gradients once before the microbatch loop. Do not step the optimizer inside it.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 5 --only
uv run course check 5
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 5 --level 1`.

## Explain and investigate

Compare microbatch sizes one and four with averaging two mean losses.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
