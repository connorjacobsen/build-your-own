# 13. Save resumable state

Weights alone do not describe the state of training. Adam moments, the update counter and random-number state affect the next update. A resumable checkpoint and an inference export serve different purposes. This stage stores trusted local training state; later the release course records content identity and governs which artifacts become active.

## Implementation contract

Work in `src/learner/api.py`. Implement **save_checkpoint**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`save_checkpoint`**

Save dict model, optimizer, step, rng (CPU torch RNG state) using torch.save.
This resume format is distinct from the portable inference export. Parent directory exists.

Save the CPU RNG as a tensor. The caller owns directory creation.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 13 --only
uv run course check 13
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 13 --level 1`.

## Explain and investigate

List additional state needed for dropout, a shuffled data iterator and a learning-rate scheduler.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)
