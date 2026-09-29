# 06. Implement causal grouped-query attention

Each query compares with permitted keys, normalizes those scores and averages values. Query heads may outnumber KV heads: consecutive query groups share the same KV head. The causal mask uses absolute positions, including a query_start offset. Using a plain triangular mask for a later chunk incorrectly hides part of its history.

## Implementation contract

Work in `src/learner/api.py`. Implement **attention**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`attention`**

Causal GQA with an explicit offset mask; handles cached multi-token chunks.

Scores divide by sqrt(head_dim). Compute softmax in float32. Queries have shape [T,Hq,d], keys/values [S,Hkv,d].

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 6 --only
uv run course check 6
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 6 --level 1`.

## Explain and investigate

Draw the mask for three queries beginning at position two.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)
