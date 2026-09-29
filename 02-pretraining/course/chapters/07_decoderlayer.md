# 07. Build one decoder layer

A pre-normalized decoder alternates attention and a gated feed-forward transformation, with a residual connection around each. SwiGLU multiplies a SiLU gate by a separate up projection before returning to model width. Keeping projection and residual completion as separate methods makes the same weights usable later in packed inference.

## Implementation contract

Work in `src/learner/api.py`. Implement **DecoderLayer**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`DecoderLayer`**



Use bias-free q,k,v,o,gate,up,down linear layers; attn_norm and ffn_norm. project returns rotated q/k and unrotated v. finish applies attention residual then SwiGLU residual.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 7 --only
uv run course check 7
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 7 --level 1`.

## Explain and investigate

Explain why every projection need not have the same output width.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)
