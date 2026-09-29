# 10. Pack without erasing boundaries

Packing saves padding but introduces a semantic risk: independent documents can attend to one another or create cross-document targets. Token IDs alone cannot express which tokens belong together. Segment IDs, position IDs and a target-position loss mask preserve those boundaries. This stage keeps documents whole; truncation or chunking must be an explicit upstream policy.

## Implementation contract

Work in `src/learner/api.py`. Implement **pack_documents**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`pack_documents`**

Greedily pack whole nonempty token-ID lists in input order, never split a document.
Reject block_size<1 or any document longer than it. Return list of dicts: input_ids,
segment_ids (original document index, padding=-1), position_ids (0..len(doc)-1,
padding=0), loss_mask (bool at TARGET position: true except each document first token
and padding). Pad every block to block_size. Empty documents are ignored, but indexes
still refer to the original input. Empty input returns [].

First tokens have loss_mask=false because no earlier token in that document predicts them. EOS is supervised.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 10 --only
uv run course check 10
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 10 --level 1`.

## Explain and investigate

Explain why masking the BOS target does not remove the need for an attention mask.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 data curriculum](https://cs336.stanford.edu/)
- [Python Unicode normalization](https://docs.python.org/3/library/unicodedata.html)
