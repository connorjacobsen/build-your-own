# 13. Integrate the preparation pipeline

Pipeline ordering matters. Exact deduplication precedes near-duplicate grouping, and decontamination removes reserved content before split assignment. Retaining original text and adding content IDs preserves both usability and traceability. The final artifact is a real input to the training course, so this integration gate checks the combined behavior rather than isolated helper outputs.

## Implementation contract

Work in `src/learner/api.py`. Implement **prepare_dataset**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`prepare_dataset`**

Integrated preparation: quality_filter defaults, deduplicate, decontaminate defaults,
duplicate_components defaults, split_groups(.2,salt), then write_dataset. Output copied
records gain split and content_sha256 fields. Validate unique input IDs before filtering.
Return manifest. Original text and IDs survive; holdout texts never enter the output.

Validate raw record IDs before filtering so a discarded record cannot hide a duplicate identifier.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 13 --only
uv run course check 13
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 13 --level 1`.

## Explain and investigate

Change one preprocessing rule and describe which experiment metadata must change.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 data curriculum](https://cs336.stanford.edu/)
- [Python Unicode normalization](https://docs.python.org/3/library/unicodedata.html)
