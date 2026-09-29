# 09. Score whole responses

Preference optimization compares response probabilities conditioned on prompts. Summing token log probabilities produces the log probability of the sequence under the autoregressive factorization. Averaging instead changes the objective and its relationship to response length. Prompt tokens are excluded from the sum but still supply context to the model.

## Implementation contract

Work in `src/learner/api.py`. Implement **sequence_logps**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`sequence_logps`**

Return [B] SUM of next-token log probabilities for labels[:,1:] excluding -100.
An unsupervised row contributes zero. Keep gradients; do not length-normalize for DPO.

Unlike masked_loss, return one sum per example and permit all-masked rows.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 9 --only
uv run course check 9
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 9 --level 1`.

## Explain and investigate

Compare two equally probable-per-token responses of different lengths.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)
