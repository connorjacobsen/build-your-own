# 02. Shift inputs and targets

A causal language model predicts the next token at each input position. The target window therefore extends one position beyond the input window. Off-by-one mistakes can produce a model trained to copy its current input or predict across unintended boundaries. Explicit starting offsets make data selection testable separately from the model.

## Implementation contract

Work in `src/learner/api.py`. Implement **next_token_batch**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`next_token_batch`**

From one 1D token stream, return long X,Y tensors [len(starts),context].
X starts at each specified offset; Y is shifted one position. Reject empty starts,
context<1, negative starts and windows extending past the available next token.

This stage samples within one stream. The data course adds document-isolated packing.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 2 --only
uv run course check 2
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 2 --level 1`.

## Explain and investigate

Trace context=3 over five tokens. Which starts are valid?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)
