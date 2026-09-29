# 05 · From one sequence to a changing batch

**Time:** 90–120 minutes. **Prerequisite:** cached decode. **Deliverable:** packed token metadata and a model of wasted batch work. **Build:** `pack_tokens`.

Batching lets several requests reuse one model invocation. Large matrix multiplications often use hardware more efficiently than many tiny ones. But requests differ in prompt length, output length, and arrival time. A serving system has to represent those differences without mixing one request's attention context with another's.

Start with three distinct ideas:

- **Sequential execution:** finish A, then B, then C.
- **Static batching:** admit a fixed cohort and keep it until that batch completes. Some implementations retain padded inactive slots; others compact them. Static admission still prevents new requests from joining that cohort.
- **Continuous batching:** reconsider membership between execution iterations, allowing completed requests to leave and new ones to join.

Continuous batching is an admission/scheduling policy. Packed execution is a tensor representation. Neither alone guarantees a throughput improvement on every workload.

## Padding waste during prefill

A rectangular batch `[B,max_prompt,D]` is convenient, but padded positions may consume work. Packing concatenates the real tokens into `[sum(lengths),D]`. A cumulative-length vector identifies where each sequence starts and ends. Positions restart at each sequence's logical offset, not at its location in the packed buffer.

```python
sequences = [[1,2,3], [4], [5,6,7,8,9,10]]
lengths = [len(x) for x in sequences]
flat = torch.tensor([token for sequence in sequences for token in sequence])
cu_seqlens = torch.tensor([0] + list(np.cumsum(lengths)))
positions = torch.cat([torch.arange(n) for n in lengths])
print("tokens:", flat.tolist())
print("boundaries:", cu_seqlens.tolist())
print("positions:", positions.tolist())
print("Useful fraction of padded slots:", sum(lengths)/(len(lengths)*max(lengths)))
assert flat[cu_seqlens[1]:cu_seqlens[2]].tolist() == sequences[1]
```

Packing does not authorize cross-request attention. The linear projections can operate on all packed rows together because they are positionwise. Attention must respect sequence boundaries and each request's historical KV state. A giant triangular mask over the concatenated sequence would make B depend on A, which is a serious correctness bug.

```python
width = 16
projection = torch.nn.Linear(width, width, bias=False)
chunks = [torch.randn(n, width) for n in lengths]
with torch.inference_mode():
    packed_result = projection(torch.cat(chunks))
    separate_result = torch.cat([projection(x) for x in chunks])
torch.testing.assert_close(packed_result, separate_result)
```

## Static slots versus new admissions

The following plot is a scheduling illustration. It counts logical decode slots, not measured hardware time. Three requests produce 2, 5, and 8 tokens. Keeping all three slots until the longest finishes allocates 24 slots for 15 useful token outputs. A backend that compacts the batch removes inactive slots, but it still needs an admission policy to fill freed capacity with new requests.

```python
output_lengths = [2, 5, 8]
active = np.array([[step < n for step in range(max(output_lengths))] for n in output_lengths])
plt.imshow(active, cmap="Blues", vmin=0, vmax=1, aspect="auto")
plt.yticks(range(3), ["A", "B", "C"]); plt.xticks(range(8), range(1,9))
plt.xlabel("Decode iteration"); plt.title("Static slots: dark = useful output, light = idle")
plt.show()
print("Useful slots:", active.sum(), "allocated slots:", active.size)
```

A real batch can mix prompt chunks and decode tokens. Its work budget is best expressed in **new token positions to compute**, alongside limits on active sequences and available cache capacity. A batch of eight one-token decodes is not the same work as eight 2,000-token prefills.

## What our runner actually batches

In lesson 07, `forward_paged_batch` concatenates all new hidden states and runs Q/K/V projections, output projections, and feed-forward layers over the packed tensor. It then loops over requests for attention, gathering each request's historical K/V. This is real shared projection work, but the Python attention loop and gather are a performance limitation.

Production backends accept metadata such as sequence lengths, positions, block tables, and slot mappings to execute attention efficiently across requests. Our explicit loop keeps boundaries inspectable and becomes a correctness oracle for a future kernel.

## Build and check

Implement `pack_tokens(sequences, starts)`, returning `(flat_ids, positions, cu_seqlens)` as Python lists. For `[[8,9],[7]]` with starts `[4,20]`, positions must be `[4,5,20]`, not `[0,1,2]` or `[4,5,6]`. The cumulative sequence lengths are `[0,2,3]`.

```python
if RUN_EXERCISES:
    from exercises.checks import check
    check("packing")
else:
    print("Packing exercise skipped.")
```

**Failure lab:** concatenate two prompts and call dense attention with one causal mask. Compare request B's logits with its independent logits. Explain why output shapes look plausible despite the wrong semantics.

**Design question:** why not simply use one Python thread per request? Threads do not establish a common token budget, shared batch metadata, or cache ownership rules. Concurrency in the frontend and batching in the model runner solve different problems.

Upstream bridge: the [pinned GPU model runner](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/worker/gpu_model_runner.py) prepares flattened inputs and attention metadata before executing the model. Read `_prepare_inputs` before attempting to understand the complete `execute_model` method.

Next: [06 — Physical blocks](06_paged_memory.ipynb).
