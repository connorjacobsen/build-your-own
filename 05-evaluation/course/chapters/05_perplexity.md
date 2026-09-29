# 05. Aggregate perplexity by token count

Perplexity exponentiates average negative log likelihood. Averaging per-document perplexities is not equivalent because exponentiation is nonlinear and documents contain different numbers of tokens. Carrying loss sums and counts through aggregation avoids both mistakes. Comparisons require the same tokenizer and evaluation corpus.

## Implementation contract

Work in `src/learner/api.py`. Implement **perplexity**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`perplexity`**

exp(sum total NLL / sum token counts); lists nonempty, equal length; counts>0;
NLLs finite and nonnegative. Return float (possibly inf for overflow).

Inputs are NLL sums, not already averaged losses.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 5 --only
uv run course check 5
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 5 --level 1`.

## Explain and investigate

Why is perplexity across different token vocabularies difficult to compare?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 evaluation curriculum](https://cs336.stanford.edu/)
- [ARENA evaluations](https://www.arena.education/curriculum)
