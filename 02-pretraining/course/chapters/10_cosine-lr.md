# 10. Schedule the learning rate

Warmup and decay encode a choice about how aggressively to change a model over time. Endpoints are part of the API: this course defines the first warmup rate as zero and reaches peak exactly at warmup. Explicit clamping prevents a cosine schedule from rising again after the planned training horizon.

## Implementation contract

Work in `src/learner/api.py`. Implement **cosine_lr**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`cosine_lr`**

0<=warmup<total; step>=0. With warmup>0, linear step/warmup * peak before warmup.
Cosine decay from peak at warmup to floor at total; clamp later steps to floor.
Reject invalid steps, bounds, negative floor or peak<floor.

Validate bounds rather than dividing by zero for an invalid schedule.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 10 --only
uv run course check 10
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 10 --level 1`.

## Explain and investigate

Plot the schedule with and without warmup and label its endpoints.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)
