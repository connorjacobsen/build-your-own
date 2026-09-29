# 02. Score acceptable references

Some questions have several valid surface forms. Evaluate against the set of acceptable references rather than choosing one arbitrarily. The metric remains exact after normalization; it does not infer synonyms. An empty reference list is malformed evaluation data and must fail rather than make every model incorrect.

## Implementation contract

Work in `src/learner/api.py`. Implement **exact_match**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`exact_match`**

Return float 1.0 if normalized prediction matches any acceptable string, else 0.0.
acceptable must be a nonempty list; empty lists raise ValueError.

Return a numeric score suitable for aggregation, not a generated explanation.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 2 --only
uv run course check 2
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 2 --level 1`.

## Explain and investigate

When would exact match be suitable for model behavior, and when would it be misleading?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 evaluation curriculum](https://cs336.stanford.edu/)
- [ARENA evaluations](https://www.arena.education/curriculum)
