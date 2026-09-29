# 11. Optimize a preference pair

The reference model defines a fixed comparison point. Accidentally updating it changes the objective during the run. Disabling its gradients is separate from setting evaluation mode: the latter controls behaviors such as dropout, while the former controls the differentiation graph. A one-step test can verify the preferred response’s relative margin moves in the correct direction.

## Implementation contract

Work in `src/learner/api.py`. Implement **dpo_step**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`dpo_step`**

chosen/rejected are collate-style batches of equal batch size. One policy update
from summed response log probabilities. Run reference in eval mode under no_grad;
clear policy gradients, leave reference parameters unchanged, return float DPO loss.

Equal policy and reference initially imply loss log(2), regardless of the original margin.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 11 --only
uv run course check 11
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 11 --level 1`.

## Explain and investigate

Why can a lower DPO loss coexist with worse performance on an unrelated task?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)
