# 12. Perform one token-weighted update

Unequal sequence lengths expose a common reduction bug. Averaging sequence losses equally gives a token in a short sequence more weight than one in a long sequence. Summing token losses and dividing by the total supervised count matches the intended objective. Clear previous gradients before computing the new step.

## Implementation contract

Work in `src/learner/api.py`. Implement **train_step**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`train_step`**

One update on nonempty sequences, each at least two token IDs long.
Weight all target tokens equally, not all sequences equally. Clear old gradients,
set training mode, backpropagate the aggregate loss, optimizer.step(), return float loss.

Use a supplied torch optimizer; the earlier manual AdamW exercise established its mechanics. Do not detach the loss until after backpropagation.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 12 --only
uv run course check 12
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 12 --level 1`.

## Explain and investigate

Compare a batch of lengths 2 and 20 under sequence and token averaging.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)
