# 07. Run one supervised update

A fine-tuning step reuses the familiar training loop with a changed supervision policy and trainable parameter set. Testing frozen weights before and after the update catches accidental base training. The first LoRA update can leave A unchanged because B starts at zero; tests should check expected learning mechanics rather than assume every parameter moves immediately.

## Implementation contract

Work in `src/learner/api.py`. Implement **sft_step**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`sft_step`**

One optimizer update using model(input_ids)->[B,T,V] logits and masked_loss.
Set train mode, clear old grads, backpropagate, step; return pre-update float loss.
The teaching model interface needs no attention mask because test fixtures are causal
positionwise networks; capstone wrappers must handle padding and document isolation.

The optimizer should be constructed from freeze_except_adapters. The small test model is positionwise; real transformer masking remains the caller’s responsibility.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 7 --only
uv run course check 7
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 7 --level 1`.

## Explain and investigate

Compare a fully supervised loss with assistant-only loss on the same batch.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)
