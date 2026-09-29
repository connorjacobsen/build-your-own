# 03. Score a continuation at the right offset

Likelihood evaluation conditions on the prompt and scores only the continuation. A token’s probability comes from the logit immediately before it. Including prompt tokens rewards the model for reproducing the question rather than answering it. The final logit predicts an unobserved next token and is outside the score.

## Implementation contract

Work in `src/learner/api.py`. Implement **continuation_logp**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`continuation_logp`**

logits [T,V] correspond to ids [T]. Sum next-token log probabilities for targets
ids[prompt_length:] using rows prompt_length-1 through T-2. Require 1<=prompt_length<T.
Exclude prompt tokens. Return Python float. Caller supplies concatenated prompt+response.

The full prompt must be provided to the model even though its targets are excluded from the metric.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 3 --only
uv run course check 3
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 3 --level 1`.

## Explain and investigate

Draw token positions and corresponding prediction rows for a two-token prompt.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 evaluation curriculum](https://cs336.stanford.edu/)
- [ARENA evaluations](https://www.arena.education/curriculum)
