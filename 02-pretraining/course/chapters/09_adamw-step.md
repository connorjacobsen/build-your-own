# 09. Implement AdamW

Adam tracks a moving mean and uncentered second moment of gradients. Both estimates initially underestimate their steady-state magnitude and need step-dependent bias correction. AdamW applies weight decay directly to parameters, separately from the adaptive gradient update. Confusing it with adding an L2 term to the gradient produces different dynamics.

## Implementation contract

Work in `src/learner/api.py`. Implement **adamw_step**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`adamw_step`**

In-place AdamW over matching lists of tensors. state is initially {}, then contains
step (integer), m and v (tensor lists). Bias-correct both moments. Apply decoupled
decay p *= (1-lr*weight_decay). Do not mutate gradient tensors. No torch optimizer.
All parameters receive a dense gradient in this educational version.

Use an initially empty mutable state dictionary. Dense gradients only; sparse and missing gradients are outside this stage.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 9 --only
uv run course check 9
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 9 --level 1`.

## Explain and investigate

Compare a zero gradient update with nonzero weight decay.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)
