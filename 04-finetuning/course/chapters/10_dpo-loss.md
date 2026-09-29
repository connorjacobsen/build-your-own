# 10. Derive the DPO objective

DPO increases the policy’s chosen-versus-rejected log-probability margin relative to a fixed reference. Subtracting reference margins accounts for the base model’s existing preferences. A stable log-sigmoid avoids overflow for strongly incorrect preferences. The reference is a constant in differentiation, even if a caller supplies tensors that require gradients.

## Implementation contract

Work in `src/learner/api.py`. Implement **dpo_loss**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`dpo_loss`**

Mean -logsigmoid(beta*((pc-pr)-(rc-rr))). All arguments [B]; beta>0.
Detach reference terms so the frozen baseline receives no gradient. Stable for large margins.

beta scales the reference-relative margin; require it to be positive.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 10 --only
uv run course check 10
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 10 --level 1`.

## Explain and investigate

Derive the sign of each policy gradient when the chosen response is disfavored.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)
