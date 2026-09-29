# Glossary

| Term | Meaning in this course |
|---|---|
| Autoregressive | Generate a next token conditioned on previous tokens, then append it. |
| Logit | Unnormalized score for a vocabulary item. |
| Prefill | Compute states for input prompt positions; can be split into chunks. |
| Decode | Compute a new continuation input position using historical K/V. |
| KV cache | Persistent per-layer key/value tensors for computed positions. |
| GQA | Several query heads share a key/value head. |
| RoPE | Position-dependent rotations of query/key channels. |
| RMSNorm | Normalize by root mean square, then apply learned channel weights. |
| SwiGLU | Gated feed-forward nonlinearity using SiLU and an elementwise product. |
| Token budget | Maximum new positions scheduled in an engine iteration. |
| Computed frontier | Number of token positions whose K/V state exists. |
| Continuous batching | Admit/retire requests between execution iterations. |
| Packed batch | Concatenate useful token rows and carry sequence-boundary metadata. |
| Block table | Mapping from a request's logical blocks to physical cache blocks. |
| Slot mapping | Per-token physical cache addresses derived from block tables. |
| Internal fragmentation | Unused slots inside allocated blocks, commonly at sequence tails. |
| Reservation slack | Capacity assigned for possible future tokens that have not been computed. |
| Prefix cache | Reuse already computed prompt K/V under matching context and model identity. |
| Copy on write | Clone shared storage before a branch mutates it. |
| Preemption | Suspend running work and reclaim its active resources. |
| Recompute | Rebuild discarded cache states from preserved token history. |
| TTFT | Time from request arrival to the first output token within a stated boundary. |
| ITL | Gap between consecutive output token events for one request. |
| Throughput | Work completed per wall-clock interval; specify the counted work. |
| SDPA | Scaled dot-product attention API; actual backend depends on conditions. |
| Online softmax | Numerically stable accumulation across tiles using a shared denominator. |
| CUDA Graph | Captured GPU operations replayed with stable buffers and capture constraints. |
| Tensor parallelism | Split a layer's computation across devices with necessary collectives. |
| Speculative decoding | Propose multiple tokens and verify them before committing target-model output. |
