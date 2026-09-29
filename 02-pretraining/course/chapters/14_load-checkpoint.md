# 14. Resume without changing the trajectory

A strong resume test compares the next update with uninterrupted training. Merely checking that a file loads misses omitted optimizer moments and an incorrect learning rate. Restoring RNG also ensures later stochastic choices follow the saved trajectory. Exact reproducibility here is a CPU teaching guarantee, not a promise across arbitrary GPU libraries and hardware.

## Implementation contract

Work in `src/learner/api.py`. Implement **load_checkpoint**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`load_checkpoint`**

Restore model, optimizer, CPU RNG from our trusted local checkpoint; return saved step.
Use torch.load(weights_only=True,map_location='cpu') and strict parameter matching.

Restore into a newly constructed matching model and optimizer, including saved optimizer hyperparameters.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 14 --only
uv run course check 14
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 14 --level 1`.

## Explain and investigate

Deliberately omit optimizer restoration and compare the next update.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)
