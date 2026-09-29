# 01. Define the token boundary

The training objective depends on the tokenizer. A byte vocabulary is small, deterministic and handles any UTF-8 text without vocabulary fitting. BOS and EOS are distinct from byte values: empty text still has a beginning and an end. Tokenization choices must accompany exported weights because swapping token identities changes the model even when tensor shapes remain valid.

## Implementation contract

Work in `src/learner/api.py`. Implement **encode_bytes**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`encode_bytes`**

UTF-8 bytes with BOS=256 prepended and EOS=257 appended; return list[int].

Use exactly 258 token IDs. Inference prompts omit EOS, while training documents include it.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 1 --only
uv run course check 1
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 1 --level 1`.

## Explain and investigate

What would happen if EOS were a normal byte value?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)
