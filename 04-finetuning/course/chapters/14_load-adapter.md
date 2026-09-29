# 14. Restore an adapter atomically

A failed restore should leave a model usable in its previous state. Validate all keys, shapes, scaling and base identity before copying the first tensor. This matters when one early tensor is compatible but a later one is not. A strict boundary turns silent partial loading into an actionable error.

## Implementation contract

Work in `src/learner/api.py`. Implement **load_adapter**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`load_adapter`**

Load onto an already injected matching model. Validate format, base digest, config,
all adapter keys and shapes BEFORE modifying anything; mismatch raises ValueError.
Copy tensors in place under no_grad, converting to destination dtype/device. Base stays fixed.

This is a trusted local torch artifact using weights_only=True. Reject incompatibility with ValueError.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 14 --only
uv run course check 14
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 14 --level 1`.

## Explain and investigate

Corrupt the second tensor and verify that the first tensor was not changed.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)
