# 08. Assemble a trainable decoder

A model is a parameterized composition, not just a function that emits the right shape. The output head projects the normalized residual stream to vocabulary scores. Matching module names and orientations gives the serving course an exact checkpoint interface. Causality can be tested by changing future tokens and checking that earlier logits remain identical.

## Implementation contract

Work in `src/learner/api.py`. Implement **TinyLM**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`TinyLM`**



State keys: embedding.weight; layers.N.{attn_norm.weight,q.weight,k.weight,v.weight,o.weight,ffn_norm.weight,gate.weight,up.weight,down.weight}; norm.weight; lm_head.weight. Implement dense forward, device and _ids. Cached forward is optional here.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 8 --only
uv run course check 8
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 8 --level 1`.

## Explain and investigate

Count parameters and estimate bytes before allocating a larger config.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)
