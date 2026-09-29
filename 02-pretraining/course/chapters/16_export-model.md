# 16. Export a portable model artifact

Serving needs weights, architecture configuration and tokenizer identity together. Optimizer moments are unnecessary. A digest identifies exact file bytes; it does not by itself prove the quality or provenance of the training data. The shared toylm-v1 format makes it possible to test a training-to-inference handoff with strict parameter loading and logit equivalence.

## Implementation contract

Work in `src/learner/api.py`. Implement **export_model**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`export_model`**

Write portable torch payload: format='toylm-v1', config=vars(model.cfg),
tokenizer={'kind':'utf8-byte','bos_id':256,'eos_id':257,'vocab_size':258},
state_dict=detached CPU cloned tensors, provenance=caller dict of JSON-safe metadata.
Require cfg.vocab_size=258. Return SHA256 of the exact file bytes. No optimizer in export.

The shared artifact contract is documented at ../ARTIFACTS.md from the course root. Do not relabel an incompatible vocabulary.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 16 --only
uv run course check 16
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 16 --level 1`.

## Explain and investigate

Verify exported logits before attempting any serving optimization.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)
