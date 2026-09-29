# 01. Serialize supervised conversations

Instruction tuning begins with an exact token sequence. Role headers, separators and end markers influence what the model learns to emit. Supervision typically belongs to assistant responses rather than user prompts. A response mask must be created alongside serialization; reconstructing it later by searching token strings can confuse literal user content with control syntax.

## Implementation contract

Work in `src/learner/api.py`. Implement **chat_tokens**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`chat_tokens`**

Return (ids, assistant_mask) lists. Start with BOS=256 (mask false).
    For each {'role': system|user|assistant, 'content':str}, append UTF-8 bytes of
    '<|ROLE|>
' (mask false), UTF-8 content (mask true iff assistant), then EOS=257
    (mask true iff assistant). Reject empty message lists, unknown roles or non-string content.
    Literal role headers are ordinary bytes, not additional vocabulary IDs.
    

Use the byte vocabulary shared with pretraining. This toy template is explicit and is not interchangeable with a downloaded model’s chat template.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 1 --only
uv run course check 1
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 1 --level 1`.

## Explain and investigate

What breaks if a user types a role header inside their message?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)
