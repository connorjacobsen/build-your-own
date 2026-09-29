# 11. Build isolated causal visibility

An attention mask is a statement about allowed information flow. Causality permits keys no later than the query; isolation additionally requires equal valid segment IDs. Padding queries have no valid keys, making naive all-negative-infinity softmax undefined. A model consuming this representation should avoid evaluating padding rows or explicitly handle them.

## Implementation contract

Work in `src/learner/api.py`. Implement **attention_mask**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`attention_mask`**

Return list[list[bool]] allowed[q][k]: k<=q, equal segment IDs and segment >=0.
Padding rows are entirely false; callers must avoid softmax over these rows.

Return Python boolean rows, not a floating additive mask. True means allowed.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 11 --only
uv run course check 11
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 11 --level 1`.

## Explain and investigate

Draw a block diagonal causal mask for two packed documents.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 data curriculum](https://cs336.stanford.edu/)
- [Python Unicode normalization](https://docs.python.org/3/library/unicodedata.html)
