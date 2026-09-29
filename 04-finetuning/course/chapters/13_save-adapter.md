# 13. Bind an adapter to its base

Adapter matrices have meaning relative to the base weights used during training. Two models with identical tensor shapes can require different adapters. A base artifact digest gives this relationship an explicit identity. Rank and scaling belong in the artifact too, because the same matrices produce different updates under a different alpha.

## Implementation contract

Work in `src/learner/api.py`. Implement **save_adapter**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`save_adapter`**

Save trusted local torch payload format='toyadapter-v1', base_sha256 (64 hex chars),
state=adapter_state(model), config={module_name:{rank,alpha}}. Return SHA256 file digest.
Reject malformed base digest. The base digest identifies a toylm-v1 artifact externally.

The caller supplies the previously verified base artifact digest. A valid hex string alone is not evidence that the base file exists.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 13 --only
uv run course check 13
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 13 --level 1`.

## Explain and investigate

What should an experiment registry record besides the adapter file?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)
