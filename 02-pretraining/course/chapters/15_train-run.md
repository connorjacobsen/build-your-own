# 15. Train and diagnose a tiny model

Overfitting a tiny controlled corpus is a useful systems diagnostic. It shows the loss, gradients and optimizer are connected. It does not show generalization. Repeated local seeds make regressions visible and separate initialization effects from coding changes. A fresh initialization belongs inside a random-state scope so running the exercise does not alter unrelated experiments.

## Implementation contract

Work in `src/learner/api.py`. Implement **train_run**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`train_run`**

Initialize TinyLM under fork_rng, AdamW(lr=lr,weight_decay=.01), then train_step
on all supplied sequences for each update. Return model, list of pre-update losses.
Preserve the caller's CPU torch RNG state. Default config has dim=16, n_layers=1,
n_heads=2,n_kv_heads=1,hidden_dim=32; vocab and max length keep TinyConfig defaults.

Use train_step rather than a second independently coded training loop. The acceptance task tests mechanics; use the capstone for held-out quality.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 15 --only
uv run course check 15
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 15 --level 1`.

## Explain and investigate

Compare memorization loss with loss on a held-out pattern.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)
