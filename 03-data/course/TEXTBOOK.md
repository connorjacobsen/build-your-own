# Data defines the learning problem

A model optimizes an expectation under its training distribution. Filtering, deduplication, sampling and packing all change that distribution or the conditional prediction problem. Treat them as modeling decisions rather than invisible preprocessing.

## Identity has several meanings

A record ID identifies a source entry. A normalized content hash identifies a chosen equivalence class of text. A file digest identifies exact serialized bytes. Keep these concepts separate: two records can share content but different provenance, and two differently serialized files can contain equivalent records.

Normalization is not neutral. Compatibility forms and casefolding can erase distinctions important in code, identifiers or named entities. The course keeps original text for training and normalized text for matching so you can inspect what each policy does.

## Why transitive grouping matters

Suppose shingle sets are A={ab,bc}, B={bc,cd}, C={cd,de}. Adjacent pairs have Jaccard similarity 1/3 while A and C have zero overlap. At threshold 1/3, all three belong to one connected component. Splitting that component risks leakage through the bridge document B even if A and C do not look similar directly.

Exact quadratic comparisons are reasonable for a tiny learning corpus. At large scale, approximate candidate generation such as locality-sensitive hashing reduces comparisons, but brings missed-pair and threshold behavior that require evaluation of their own.

## Packing changes information flow

Packing two independent documents into one sequence creates two hazards. A target at the second document's beginning can be incorrectly predicted from the first document, and later tokens can attend to unrelated earlier content. A loss mask solves the first problem; a segmented causal attention mask solves the second. Position IDs determine whether positional context restarts. None of those policies is implied by concatenated token IDs alone.

## Sampling and provenance

A source with mixture mass 0.5 supplies half the expected draws even if it has very few unique examples. Record both total draws and unique content counts. A reproducible local RNG fixes one sampling sequence but does not establish that its distribution is appropriate.

Use the experiment journal to inspect rejected content and compare source slices. A filtering rule is only useful in relation to a task, its error costs and its intended population. The output manifest makes experiments repeatable; it does not make a biased or contaminated dataset valid.
