# 08. Merge an adapter for inference

The low-rank update is linear and can be folded into the original weight for inference. Doing so removes the extra adapter matrix multiplications. Numerical equivalence should be checked before any benchmark. Returning a fresh layer avoids a double-merge bug in which repeated calls silently apply the same update again.

## Implementation contract

Work in `src/learner/api.py`. Implement **merge_lora**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`merge_lora`**

Return a new ordinary nn.Linear on the same device/dtype with W + alpha/rank * B@A
and the same bias. Preserve the input adapter and all its parameters. Merging twice
from the same original gives the same result; do not add the delta into base in place.

This stage merges one layer; the capstone traverses and replaces selected wrappers.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 8 --only
uv run course check 8
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 8 --level 1`.

## Explain and investigate

How would you serve two adapters if both were merged into the same base object?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)
