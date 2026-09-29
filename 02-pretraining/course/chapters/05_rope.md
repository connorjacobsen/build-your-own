# 05. Encode relative position by rotation

Rotary position embeddings rotate adjacent coordinate pairs by an angle determined by position and frequency. Rotation preserves each pair’s squared length while changing dot products according to position differences. The exact pairing convention is part of the checkpoint contract; alternating pairs and split-half conventions are not interchangeable.

## Implementation contract

Work in `src/learner/api.py`. Implement **rope**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`rope`**

Rotate adjacent pairs. x: [T,H,D], positions: [T] absolute positions.

x is [T,H,d], positions is [T], frequencies are base**(-arange(0,d,2)/d). Use adjacent pairs and base 10000.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 5 --only
uv run course check 5
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 5 --level 1`.

## Explain and investigate

Verify length preservation for positions 0, 1 and 100.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)
