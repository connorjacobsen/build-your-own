# 04. Make length normalization explicit

Summed log probability tends to favor shorter strings, while average log probability defines a different comparison. Neither choice should be hidden inside a helper. Declaring the protocol allows results to be reproduced and compared. Stable tie handling prevents candidate order from producing undocumented random variation.

## Implementation contract

Work in `src/learner/api.py`. Implement **rank_choices**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`rank_choices`**

Return winning index, maximum summed logp or logp/count when normalize=True.
Equal scores choose earliest index. Require nonempty equal-length inputs, positive
counts and finite scores. This exposes two different evaluation protocols explicitly.

Lengths count scored continuation tokens, not prompt tokens.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 4 --only
uv run course check 4
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 4 --level 1`.

## Explain and investigate

Construct two choices whose ranking flips under normalization.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 evaluation curriculum](https://cs336.stanford.edu/)
- [ARENA evaluations](https://www.arena.education/curriculum)
