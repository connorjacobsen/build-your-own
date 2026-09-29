# 01. Separate matching text from training text

Text equality is a policy decision. Compatibility normalization merges some visually or semantically related spellings, while casefolding handles more than ASCII lowercase. Whitespace normalization makes layout differences irrelevant for matching. These operations can remove useful distinctions, so preserve original content and apply this representation only to matching.

## Implementation contract

Work in `src/learner/api.py`. Implement **normalize**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`normalize`**

NFKC normalize, casefold, collapse all whitespace to single spaces, strip ends.
Use for matching; preserve original text separately for training and provenance.

Normalization must be deterministic and idempotent. Never silently replace original training text.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 1 --only
uv run course check 1
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 1 --level 1`.

## Explain and investigate

Find two strings that this normalization merges but that might matter in a code corpus.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 data curriculum](https://cs336.stanford.edu/)
- [Python Unicode normalization](https://docs.python.org/3/library/unicodedata.html)
