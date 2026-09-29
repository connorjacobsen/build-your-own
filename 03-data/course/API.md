# Public API and input domains

These are scaffold contracts, not implementations. Earlier stages remain dependencies of later stages.

## `normalize`

NFKC normalize, casefold, collapse all whitespace to single spaces, strip ends.
Use for matching; preserve original text separately for training and provenance.

Signature: `normalize(text)`

## `document_id`

SHA256 lowercase hex of normalize(text).encode('utf-8').

Signature: `document_id(text)`

## `quality_filter`

Return bool: normalized whitespace words count >= min_words and the most frequent
word's fraction <= max_repeat. Empty text always fails. Boundary thresholds are inclusive.
This toy heuristic is not a language detector or a universal measure of quality.

Signature: `quality_filter(text, min_words=3, max_repeat=0.5)`

## `deduplicate`

Records are dicts with id,text. Return shallow copies of the first occurrence for
each normalized content ID, preserving input order. Never mutate caller records.

Signature: `deduplicate(records)`

## `shingles`

Set of consecutive n-word tuples from normalized text. n>=1, else ValueError.
Documents with fewer than n words produce an empty set.

Signature: `shingles(text, n=3)`

## `duplicate_components`

Return list of ID lists, components in first-record order, members in input order.
Add an edge when nonempty shingle-set Jaccard >= threshold, or normalized text is equal.
Connected components include transitive matches; distinct short documents do not match.
IDs must be unique; threshold in (0,1]. O(N^2) is intentional for this teaching dataset.

Signature: `duplicate_components(records, threshold=0.5, n=3)`

## `split_groups`

Return ID->'train'/'validation'. Hash salt + ':' + lexicographically smallest group ID;
SHA256 interpreted as big-endian int divided by 2**256; values below fraction go to validation.
Validate fraction in [0,1], nonempty groups and no ID repeated across or within groups.
Group and member input ordering must not affect assignment. This does not enforce exact counts.

Signature: `split_groups(groups, validation_fraction=0.2, salt='course')`

## `decontaminate`

Return copied records with no shared n-word shingle with any holdout text.
Also reject exact normalized matches, including short texts. Preserve order.
This is a strict toy filter; it can discard common phrases and is not a production detector.

Signature: `decontaminate(records, holdout_texts, n=3)`

## `sample_mixture`

sources maps names to nonempty lists of records. weights has identical keys, finite
nonnegative values and positive total. Draw WITH replacement: choose sorted source name
via local random.Random(seed).choices, then a record via that RNG.choice; return copied
dicts with added source field. count>=0. Inputs remain unchanged; no global RNG use.

Signature: `sample_mixture(sources, weights, count, seed=0)`

## `pack_documents`

Greedily pack whole nonempty token-ID lists in input order, never split a document.
Reject block_size<1 or any document longer than it. Return list of dicts: input_ids,
segment_ids (original document index, padding=-1), position_ids (0..len(doc)-1,
padding=0), loss_mask (bool at TARGET position: true except each document first token
and padding). Pad every block to block_size. Empty documents are ignored, but indexes
still refer to the original input. Empty input returns [].

Signature: `pack_documents(documents, block_size, pad_id=0)`

## `attention_mask`

Return list[list[bool]] allowed[q][k]: k<=q, equal segment IDs and segment >=0.
Padding rows are entirely false; callers must avoid softmax over these rows.

Signature: `attention_mask(segment_ids)`

## `write_dataset`

Create directory; write records.jsonl as UTF-8 JSON, sort_keys=True, ensure_ascii=False,
separators=(',',':'), one newline per record. Write manifest.json containing format
'toydata-v1', file 'records.jsonl', sha256 of exact bytes, records count. Return manifest.
Empty datasets have zero bytes. Never overwrite an existing records.jsonl or manifest.json.

Signature: `write_dataset(directory, records)`

## `prepare_dataset`

Integrated preparation: quality_filter defaults, deduplicate, decontaminate defaults,
duplicate_components defaults, split_groups(.2,salt), then write_dataset. Output copied
records gain split and content_sha256 fields. Validate unique input IDs before filtering.
Return manifest. Original text and IDs survive; holdout texts never enter the output.

Signature: `prepare_dataset(records, holdout_texts, directory, salt='course')`
