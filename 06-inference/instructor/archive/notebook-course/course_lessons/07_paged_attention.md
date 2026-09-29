# 07 · Connect paged memory to real model execution

**Time:** 2–3 hours. **Prerequisite:** block tables and packed batches. **Deliverable:** a paged forward pass that matches dense logits. **Build:** a `run_paged` adapter and, optionally, a blockwise attention reference.

An allocator is not yet an inference engine. The model must actually write new K/V into its allocated physical slots, then read history according to the block table. Otherwise a paging demo is only a simulation beside the real model.

`TinyLM.forward_paged_batch(items,pool)` performs this connection. Each item is `(new_token_ids, block_table, computed_length)`. That last value is the absolute position where the new chunk begins. The runner packs new tokens across requests for projections, then executes attention separately for each request.

At each layer it projects all new rows, scatters the request's K/V into physical pages, gathers its valid logical history, attends using an offset mask, and combines the outputs back into the packed stream. The block table is shared across layers, but tensor contents are layer-specific.

## Test an adversarial physical layout

Never validate paging only with table `[0,1,2,...]`. An implementation that ignores the table can pass such a test. Deliberately shuffle physical block IDs and give requests unequal lengths and cache offsets.

```python
from nanovllm_course.cache import BlockPool
model = make_model()
pool = BlockPool(model.cfg, num_blocks=8, block_size=2)
allocated = pool.allocate(8)
table_a = [allocated[i] for i in [5,1,7,3]]
table_b = [allocated[i] for i in [0,6,2,4]]
a, b = [256,10,20,30,40], [256,7,8,9,10,11,12]
first = model.forward_paged_batch([(a[:3], table_a, 0), (b[:4], table_b, 0)], pool)
second = model.forward_paged_batch([(a[3:], table_a, 3), (b[4:], table_b, 4)], pool)
with torch.inference_mode():
    dense_a, _ = model(a)
    dense_b, _ = model(b)
for dense, initial, final in zip([dense_a,dense_b], first, second):
    torch.testing.assert_close(torch.cat([initial,final]), dense, atol=2e-6, rtol=2e-5)
print("Unequal chunks and shuffled pages match dense logits")
pool.release(table_a + table_b); pool.check()
```

This test simultaneously checks absolute positions, sequence isolation, paging indirection, GQA head mapping, layer-specific caches, and chunk causality. If it fails, reduce the case to one layer, one request, and a two-page boundary. Then reintroduce complexity one dimension at a time.

## Gathering is an intentional limitation

Our gather produces contiguous historical K/V for ordinary PyTorch attention. That copies data and creates temporary tensors. Python also loops over requests and queries block ownership. These choices make invariants visible; they do not demonstrate the throughput of a fused PagedAttention kernel.

A fast implementation reads blocks indirectly inside the kernel and avoids materializing the entire attention score matrix. The central mathematical tool is an online softmax reduction. For a running maximum `m`, normalizer `l`, and weighted-value numerator `a`, a new score tile `s` updates:

\[
m' = \max(m,\max(s)),\quad \alpha=\exp(m-m'),
\]
\[
l'=\alpha l + \sum_j \exp(s_j-m'),\quad
 a'=\alpha a + \sum_j\exp(s_j-m')v_j.
\]

The final output is `a/l`. Rescaling the old partial sum is essential whenever the maximum increases. Applying softmax independently per block and averaging block outputs is incorrect because each block would use a different denominator.

## A blockwise single-query reference

The following is a CPU mathematical demonstration for one head. It assumes every supplied key is visible; multi-query causal masking would be applied inside each tile. It avoids a single full-length scores tensor but still uses a Python loop. This is a stepping stone toward a kernel, not a claim of speed.

```python
def online_attention_one_query(q, k, v, tile_size):
    maximum = torch.tensor(-torch.inf)
    denominator = torch.tensor(0.)
    numerator = torch.zeros_like(q)
    for start in range(0, len(k), tile_size):
        scores = k[start:start+tile_size] @ q / math.sqrt(q.numel())
        new_max = torch.maximum(maximum, scores.max())
        rescale = torch.exp(maximum - new_max)
        weights = torch.exp(scores - new_max)
        denominator = denominator * rescale + weights.sum()
        numerator = numerator * rescale + weights @ v[start:start+tile_size]
        maximum = new_max
    return numerator / denominator

q, k, v = torch.randn(16), torch.randn(23,16), torch.randn(23,16)
expected = (k @ q / math.sqrt(16)).softmax(0) @ v
for size in [1,4,8,32]:
    torch.testing.assert_close(online_attention_one_query(q,k,v,size), expected)
print("Online softmax agrees across tile boundaries")
```

## Build a paged generation adapter

Implement `run_paged(model,prompt,count)` using the block pool and `forward_paged_batch`. Reserve `ceil((P+G−1)/B)` blocks, prefill once, then feed back one token at a time. Keep the block table until the request finishes, and release it even if your generation code raises. Start with greedy sampling.

The finished reference engine will call this same paged model path for multiple requests. You should be able to explain why reordering the items in a batch cannot allow one request to read another's keys.

```python
if RUN_EXERCISES:
    from exercises.checks import check
    check("paged")
else:
    print("Paged generation exercise skipped.")
```

**Completion criteria:** match the naive generator for prompt lengths `B−1`, `B`, and `B+1`; handle zero requested tokens without allocation; leave no owned blocks after generation; compare logits with tolerances before insisting on exact sampled IDs.

**Stretch:** replace the gather for one decode query with the online reduction over physical blocks. Test a partially filled last block. The mask must use valid logical length so stale values in unused slots cannot affect the denominator.

Upstream reading: vLLM's [paged attention design](https://docs.vllm.ai/en/v0.10.1/design/paged_attention.html) describes a particular historical kernel layout. Do not assume that layout is used by every current V1 backend. The stable lesson is the relationship between logical sequence order, physical block tables, and attention's reduction.

Next: [08 — Scheduling](08_scheduler.ipynb).
