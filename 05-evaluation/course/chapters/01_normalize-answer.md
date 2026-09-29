# 01. Specify answer normalization

A metric embeds decisions about what counts as the same answer. Removing punctuation can help harmless formatting differences but can also merge distinct strings. A clear policy is more useful than a vague claim of semantic correctness. This first stage deliberately uses a narrow deterministic normalizer whose limitations are easy to inspect.

## Implementation contract

Work in `src/learner/api.py`. Implement **normalize_answer**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`normalize_answer`**

casefold, remove ASCII punctuation (not whitespace), then collapse whitespace.
No article removal, numeric coercion or Unicode punctuation removal in this policy.

Do not remove articles or Unicode punctuation. Changing normalization creates a different evaluation protocol.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 1 --only
uv run course check 1
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 1 --level 1`.

## Explain and investigate

Find a punctuation-sensitive answer pair that this metric would confuse.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 evaluation curriculum](https://cs336.stanford.edu/)
- [ARENA evaluations](https://www.arena.education/curriculum)
