# vLLM reading map and implementation boundaries

Reading baseline: **v0.11.0**, commit `b8b302cde434df8c9289a2b465406b47ebab1c2d`. Source inspection and course assembly: September 26–27, 2026. This older fixed baseline makes function names and line links reproducible. It is not a deployment-version recommendation. Current documentation was also consulted for architectural context; implementation claims below use the pinned source.

The actual downloaded files, byte counts, and hashes are recorded in [upstream_manifest.json](upstream_manifest.json). Run `uv run python scripts/fetch_reference.py` from the project root for the local reading set. Downloaded upstream files preserve their own copyright/license notices and remain in ignored `.reference/`; the course does not modify them.

| Upstream source | What to inspect | Course connection and reading question |
|---|---|---|
| [Request, line 26](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/request.py#L26) | Token history, computed counts, status, sampling state | Lessons 08/13: which fields describe input history versus completed work? |
| [Scheduler.schedule, line 179](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/sched/scheduler.py#L179) | Token deficits, budgets, running/waiting paths, allocation failure | Lesson 08: how does one abstraction cover prefill and decode? |
| [Scheduler preemption branch, line 243](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/sched/scheduler.py#L243) | Release, reset computed count, prepend waiting request | Capstone extension: what survives recomputation, and why must it not resample historical outputs? |
| [KVCacheManager.get_computed_blocks, line 154](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/kv_cache_manager.py#L154) | Full-block hits and last-token recomputation | Lesson 09: why is KV alone insufficient for a final logit row? |
| [KVCacheManager.allocate_slots, line 193](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/kv_cache_manager.py#L193) | Computed/new/preallocated regions and capacity | Lessons 06/08: how does incremental allocation differ from full reservation? |
| [KVCacheManager.free, line 306](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/kv_cache_manager.py#L306) | Releasing a request's block ownership | Lessons 08/09: why does release not necessarily invalidate a cached prefix? |
| [BlockPool.get_new_blocks, line 257](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/block_pool.py#L257) | Free queue, cached block eviction, active references | Lesson 06: where is capacity actually reclaimed? |
| [BlockPool.free_blocks, line 338](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/block_pool.py#L338) | Reference decrements and queue order | Lesson 09: distinguish active references from cached identity. |
| [EngineCore.step, line 272](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/engine/core.py#L272) | Schedule, execute, update, output | Lessons 08/10: which layer owns each transition? |
| [GPUModelRunner._prepare_inputs, line 923](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/worker/gpu_model_runner.py#L923) | Flattened input and positional metadata | Lessons 05/07: packed offsets and logical positions are different coordinates. |
| [GPUModelRunner.execute_model, line 2231](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/worker/gpu_model_runner.py#L2231) | Backend execution and sampling plumbing | Lesson 12: what Python/device boundaries disappear in a production runner? |
| [Sampler, line 22](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/sample/sampler.py#L22) | Logits processing, greedy/random paths, generators | Lesson 03: which policy changes probabilities without changing model weights? |
| [LlamaAttention, line 103](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/model_executor/models/llama.py#L103) | Projections, rotary positions, attention backend | Lesson 02: compare mathematical layers with parallel-aware abstractions. |
| [LlamaDecoderLayer, line 241](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/model_executor/models/llama.py#L241) | Residual and normalization flow | Lesson 02: identify invariants despite fused implementations. |

## A deliberate fidelity table

| Feature | Course implementation | Production reading boundary |
|---|---|---|
| Transformer | Tiny random byte model; RMSNorm/RoPE/GQA/SwiGLU | Architecture-inspired; no Llama checkpoint compatibility |
| Cached execution | Real per-layer K/V reused by the same model | Mathematical behavior is checked against dense logits |
| Paged storage | Fixed tensors, block tables, scatter/gather | Not an optimized paged GPU kernel |
| Batching | Packed projections/MLPs, per-request attention loop | Production attention backends can batch/fuse more work |
| Scheduling | FCFS admission; round-robin execution; token budget | Does not reproduce every V1 priority/async policy |
| Allocation | Full declared capacity reserved on admission | vLLM can allocate incrementally and preempt/recompute |
| Prefix identity | Exact full-prefix tuples; one immutable engine/model | Upstream uses chained block identity and additional context |
| Prefix ownership | Cache entry holds one ref plus active-request refs | Upstream cached entries can remain evictable with zero active refs |
| Copy on write | Standalone block-pool demonstration | No beam-search/branching engine API |
| Serving | One local shared engine; complete custom JSON responses | No OpenAI compatibility or streaming HTTP in the reference |
| Accelerators | SDPA comparison and small optional device experiments | No claimed GPU speedup or whole-engine CUDA Graph capture |
| Distributed/speculative | Algebra and verification demonstrations | Not integrated distributed/speculative serving |

## Primary references

- [Pinned architecture overview](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/docs/design/arch_overview.md): component boundaries.
- [Pinned automatic prefix caching design](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/docs/design/prefix_caching.md): identity, blocks, and lifecycle.
- [PagedAttention paper](https://arxiv.org/abs/2309.06180): memory-management motivation and original evaluation; the course makes no attempt to reproduce its speedup numbers.
- [Historical paged attention kernel design](https://docs.vllm.ai/en/v0.10.1/design/paged_attention.html): a specific kernel, not a universal current-backend layout.
- [Current vLLM architecture](https://docs.vllm.ai/en/latest/design/arch_overview/) and [optimization guide](https://docs.vllm.ai/en/latest/configuration/optimization/): evolving context, not the pinned code contract.
- [uv/Jupyter integration](https://docs.astral.sh/uv/guides/integration/jupyter/): environment and kernel setup.
- [PyTorch 2.7 SDPA](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html) and [CUDA Graph notes](https://docs.pytorch.org/docs/2.7/notes/cuda.html#cuda-graphs): optional accelerator labs.
- [Speculative decoding paper](https://arxiv.org/abs/2211.17192): probability-preserving verification, beyond the greedy alignment example.

Most course prose and all toy implementations are original explanatory material. Upstream links are reading assignments and evidence for the stated architectural connections; they are not a claim that the toy copies the upstream implementation.
