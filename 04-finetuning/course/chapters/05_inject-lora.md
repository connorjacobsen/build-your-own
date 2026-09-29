# 05. Inject adapters deliberately

Target selection is an architectural decision. Exact qualified module names avoid accidentally adapting every layer whose name contains a substring. Validate the complete plan before replacement so a misspelled later target cannot leave the model half-modified. This stage replaces existing module objects and preserves all unaffected modules.

## Implementation contract

Work in `src/learner/api.py`. Implement **inject_lora**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`inject_lora`**

In-place replace exact qualified names of nn.Linear modules with LoRALinear.
Validate every name and type before changing any module. Reject duplicates, missing names,
empty targets and non-linear targets. Return the same model. Other modules stay untouched.

Targets refer to named_modules paths, including numeric Sequential children.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 5 --only
uv run course check 5
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 5 --level 1`.

## Explain and investigate

Compare adapting the output head with adapting attention projections.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)
