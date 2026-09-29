# 14 · Read upstream vLLM with a working mental model

**Time:** 2–4 hours plus optional projects. **Prerequisite:** the capstone or equivalent understanding. **Deliverable:** an annotated source walkthrough and an extension plan.

You now have concrete answers to the questions that make a large inference engine difficult to read: what a request owns, what the scheduler returns, which token positions need computation, where model execution writes K/V, when sampling occurs, and what release actually means.

The course source baseline is vLLM **v0.11.0**, commit `b8b302cde434df8c9289a2b465406b47ebab1c2d`. It is pinned for reproducible reading, not selected as a current deployment recommendation. The local `docs/source_map.md` contains permalinks and the exact functions inspected when assembling this course. `scripts/fetch_reference.py` downloads only the selected source files into an ignored `.reference/` directory if you want offline reading.

## Follow one request across boundaries

Read these components in sequence rather than reading each file from top to bottom:

| Question | Pinned upstream target | Toy counterpart |
|---|---|---|
| What state persists for a request? | `vllm/v1/request.py: Request` | `engine.py: Request` |
| Which positions run this step? | `core/sched/scheduler.py: Scheduler.schedule` | `Engine.step` |
| Which prompt blocks already exist? | `core/kv_cache_manager.py: get_computed_blocks` | `PrefixCache.acquire` |
| Can the work obtain physical capacity? | `KVCacheManager.allocate_slots`, `BlockPool.get_new_blocks` | `_admit`, `BlockPool.allocate` |
| How does the core coordinate a step? | `engine/core.py: EngineCore.step` | `Engine.step` and `run` |
| How are inputs and attention metadata prepared? | `worker/gpu_model_runner.py: _prepare_inputs`, `execute_model` | `forward_paged_batch` |
| Where do logits become output IDs? | `sample/sampler.py: Sampler` | `sampling.sample` |
| How is a model layer expressed? | `model_executor/models/llama.py` | `DecoderLayer` |

For each function, write down inputs, mutations, outputs, and ownership transitions. Distinguish a plan for work from confirmation that work has executed. Our synchronous engine increments computed counts after the model returns. More asynchronous systems can maintain scheduled and completed frontiers separately.

## Differences to keep visible

The toy has one process, fixed random weights, full-attention decoder layers, a CPU-oriented sampler, Python gathers, exact prefix tuples, and reservation-based admission. It has no distributed execution, fused custom attention kernel, speculative engine path, checkpoint loader, multimodal support, LoRA adapters, or production compatibility layer.

The standalone COW demonstration is an allocator lesson; the engine does not offer beam search. The CUDA Graph example captures a linear layer; it does not graph-capture the engine. The local HTTP endpoint is custom JSON; it does not implement OpenAI's API schema. These boundaries make the implementation tractable and the claims testable.

## Distributed inference: understand the algebra first

Tensor parallelism divides a layer's work across devices. For a linear map `y=xW`, partitioning output columns lets devices compute different output features independently before any required concatenation. Partitioning input rows gives partial sums that must be reduced. PyTorch stores a linear weight as `[out,in]`, so use storage axes carefully when translating the algebra.

This CPU experiment demonstrates both decompositions without claiming to model communication cost.

```python
x = torch.randn(5,16)
weight = torch.randn(24,16)  # PyTorch [out,in] convention
full = x @ weight.T
# Output feature partition: independent output pieces, then concatenate.
column_parallel = torch.cat([x @ shard.T for shard in weight.chunk(2,dim=0)],dim=-1)
# Input feature partition: partial dot products, then sum (all-reduce in a real system).
row_parallel = sum(a @ w.T for a,w in zip(x.chunk(2,dim=1),weight.chunk(2,dim=1)))
torch.testing.assert_close(column_parallel,full)
torch.testing.assert_close(row_parallel,full)
print("Both tensor-parallel decompositions preserve the linear map")
```

Now ask how attention heads are assigned, whether KV heads are sharded or replicated, and how cache bytes change per rank. Pipeline parallelism instead partitions layers and transfers activations between stages. Data parallel serving uses replicas and routes requests. None of these makes communication free, and small models may lose more to communication than they gain from additional devices.

## Speculative decoding: verify before committing

Ordinary decode computes one next-token decision per target-model invocation. A draft model can propose several IDs; the target evaluates the proposal and accepts only a valid prefix. For greedy decoding, a proposal matches if each draft ID equals the target's greedy choice at that position. On the first mismatch, emit the target choice and discard later draft IDs. If all proposals match, the target can provide one bonus token.

The example below performs one dense target verification pass. It does not use a real draft model, update a paged cache, or constitute a fast implementation. It demonstrates alignment: target logits at index `P−1` score draft token zero, not the next index.

```python
from nanovllm_course.sampling import generate_naive
model = make_model()
prompt = [256,4,8]
expected = generate_naive(model,prompt,4)
draft = expected[:3].copy()
draft[1] = (draft[1]+1) % model.cfg.vocab_size  # Deliberate mismatch.
with torch.inference_mode():
    logits,_ = model(prompt+draft)
choices = logits[len(prompt)-1:].argmax(-1).tolist()
accepted = []
for i,proposal in enumerate(draft):
    if proposal != choices[i]:
        accepted.append(choices[i])
        break
    accepted.append(proposal)
else:
    accepted.append(choices[len(draft)])
assert accepted == expected[:len(accepted)]
print("Verified continuation prefix:",accepted)
```

For stochastic speculative decoding, comparing sampled IDs is not sufficient to preserve the target distribution. The acceptance/rejection rule uses draft and target probabilities, with acceptance probability `min(1,p(token)/q(token))` and a corrected residual distribution on rejection. Read the [speculative decoding paper](https://arxiv.org/abs/2211.17192) before implementing that branch. Cache rollback and committed versus proposed positions become scheduler concerns.

## Quantization and memory planning

Quantizing weights addresses weight bandwidth/capacity. Quantizing K/V addresses a different persistent footprint. Both introduce metadata such as scales and numerical error. A memory plan should separately account for weights, cache, activations, temporary workspaces, graph/static buffers, and runtime overhead. Reducing one term does not guarantee the entire model fits.

```python
cfg = TinyConfig()
parameter_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
cache_bytes = cfg.kv_bytes_per_token() * 64 * 8
print({"model_parameter_bytes":parameter_bytes, "reserved_KV_pool_bytes":cache_bytes})
print("These omit activations, attention temporaries, Python objects, and runtime allocations.")
```

## A source-reading assignment

Choose a request with a cached prefix, a prompt longer than one scheduling chunk, and a short output limit. Trace it through the pinned code. Identify where prefix hits are capped, where blocks become owned, how new positions are scheduled, and when finished resources are released. Compare the sequence with your toy trace. Then identify one additional upstream branch—preemption, speculative tokens, or multimodal work—and explain which invariant it extends.

A useful artifact is a one-page table with three columns: upstream behavior, toy behavior, and the reason for the difference. Avoid copying the upstream implementation wholesale. The purpose of the toy is to let you reconstruct why those branches are needed.

## Where to go next

Pick one performance bottleneck supported by measurements and one correctness invariant it threatens. A kernel project threatens numerical/masking equivalence; a scheduler project threatens fairness and progress; prefix sharing threatens ownership and identity; distributed execution threatens synchronization and rank consistency. Design the failure tests before optimizing.

You have completed the core course when you can build the minimal engine, pass the behavioral checks, interpret its scheduling and memory traces, and explain which additional work would be required for production vLLM performance.
