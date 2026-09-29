# Build milestones and how to use the checks

Edit `exercises/implementation.py` and run `uv run python scripts/check_exercises.py NAME`. The checker exits nonzero for an unfinished or incorrect exercise. `--reference` tests the supplied solutions; it does not mark your learner code complete. Notebooks skip learner checks by default so reference lessons remain runnable end to end.

| Name | Lesson | Contract | Failure the checker targets |
|---|---|---|---|
| attention | 01 | GQA causal output at a nonzero absolute query offset | Wrong head mapping, wrong mask, future leakage |
| memory | 01 | K+V bytes from layers/tokens/heads/dtype | Missing factor two, query heads mistaken for KV heads |
| rope | 02 | Adjacent-pair rotation at supplied positions | Wrong frequency, position reset, broken rotation |
| sampling | 03 | Normalized nucleus probabilities in original ID order | Removing the crossing token; losing original ordering |
| decode | 04 | Greedy continuation with prompt then one-token inputs | Returning correct outputs while still recomputing history |
| packing | 05 | Flat IDs, absolute positions, cumulative boundaries | Mixing packed offsets with request positions |
| slots | 06 | Physical block IDs and offsets for a logical range | Ignoring table indirection; exceeding capacity |
| paged | 07 | Greedy generation through real paged K/V | Boundary errors at B−1, B, B+1 |
| scheduler | 08 | Counts bounded by deficits, budget, and chunk | Oversubscribing budget or skipping zero deficits incorrectly |
| prefix | 09 | Full-block reusable length leaving logits work | Exact-boundary off-by-one |
| engine | 13 | Engine lifecycle and execution contract | Partial prompt sampling, late-arrival errors, cancellation leaks |

The paged function checker compares outputs; inspect your own pool accounting as well. The full reference tests exercise shuffled physical blocks, atomic allocation, COW, eviction, and randomized arrivals/cancellations. A finite checker is not proof of all possible implementations: add counterexamples when you discover a new failure.

## Hint ladder

1. Write down the shapes or state fields before writing code.
2. Use a hand-computable two-block or two-request example.
3. Inspect the corresponding notebook's explicit implementation/trace.
4. Read the relevant function in `src/nanovllm_course/`.
5. Compare with `reference_solutions.py` and explain every difference.

For the capstone, use `starter_engine.py`. First support a single request without prefix caching. Add multiple requests and bounded scheduling. Then implement all finish paths. Add prefix caching only after ownership and pressure tests pass. You can reuse the supplied model and allocator initially; replacing them with your implementations is a second integration pass.

## Written checks worth keeping

- Explain why a prompt of length P producing G outputs usually computes P+G−1 positions.
- Derive the KV bytes formula and state what it excludes.
- Draw logical/packed/physical coordinates for a token crossing a block boundary.
- Explain why local block tokens alone are not a prefix-cache key.
- Demonstrate why a global RNG couples unrelated requests.
- Prove progress for reservation-based admission, then explain why that proof fails under dynamic allocation.
- Explain why the paged reference may be slower despite doing less historical projection work.

Your final capstone should include a trace, a memory-accounting example, numerical evidence, raw benchmark data, and a clear statement of which production features remain extensions.
