"""Instructor reference for deterministic dataset preparation."""

import hashlib
import json
from pathlib import Path
import random
import unicodedata


def normalize(text):
    """NFKC normalize, casefold, collapse all whitespace to single spaces, strip ends.
    Use for matching; preserve original text separately for training and provenance.
    """
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def document_id(text):
    """SHA256 lowercase hex of normalize(text).encode('utf-8')."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def quality_filter(text, min_words=3, max_repeat=0.5):
    """Return bool: normalized whitespace words count >= min_words and the most frequent
    word's fraction <= max_repeat. Empty text always fails. Boundary thresholds are inclusive.
    This toy heuristic is not a language detector or a universal measure of quality.
    """
    words = normalize(text).split()
    return (
        bool(words)
        and len(words) >= min_words
        and max(words.count(w) for w in set(words)) / len(words) <= max_repeat
    )


def deduplicate(records):
    """Records are dicts with id,text. Return shallow copies of the first occurrence for
    each normalized content ID, preserving input order. Never mutate caller records.
    """
    seen = set()
    result = []
    for r in records:
        digest = document_id(r["text"])
        if digest not in seen:
            result.append(dict(r))
            seen.add(digest)
    return result


def shingles(text, n=3):
    """Set of consecutive n-word tuples from normalized text. n>=1, else ValueError.
    Documents with fewer than n words produce an empty set.
    """
    if n < 1:
        raise ValueError("Positive shingle size required")
    words = normalize(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def duplicate_components(records, threshold=0.5, n=3):
    """Return list of ID lists, components in first-record order, members in input order.
    Add an edge when nonempty shingle-set Jaccard >= threshold, or normalized text is equal.
    Connected components include transitive matches; distinct short documents do not match.
    IDs must be unique; threshold in (0,1]. O(N^2) is intentional for this teaching dataset.
    """
    if not 0 < threshold <= 1 or len({r["id"] for r in records}) != len(records):
        raise ValueError("Invalid threshold or duplicate IDs")
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    sets = [shingles(r["text"], n) for r in records]
    for i in range(len(records)):
        for j in range(i):
            union = sets[i] | sets[j]
            same = normalize(records[i]["text"]) == normalize(records[j]["text"])
            if same or (union and len(sets[i] & sets[j]) / len(union) >= threshold):
                parent[find(i)] = find(j)
    groups = {}
    for i, r in enumerate(records):
        groups.setdefault(find(i), []).append(r["id"])
    return list(groups.values())


def split_groups(groups, validation_fraction=0.2, salt="course"):
    """Return ID->'train'/'validation'. Hash salt + ':' + lexicographically smallest group ID;
    SHA256 interpreted as big-endian int divided by 2**256; values below fraction go to validation.
    Validate fraction in [0,1], nonempty groups and no ID repeated across or within groups.
    Group and member input ordering must not affect assignment. This does not enforce exact counts.
    """
    flat = [x for g in groups for x in g]
    if (
        not 0 <= validation_fraction <= 1
        or any(not g for g in groups)
        or len(flat) != len(set(flat))
    ):
        raise ValueError("Invalid groups or fraction")
    out = {}
    for group in groups:
        value = int(hashlib.sha256((salt + ":" + min(group)).encode()).hexdigest(), 16) / 2**256
        split = "validation" if value < validation_fraction else "train"
        out.update({x: split for x in group})
    return out


def decontaminate(records, holdout_texts, n=3):
    """Return copied records with no shared n-word shingle with any holdout text.
    Also reject exact normalized matches, including short texts. Preserve order.
    This is a strict toy filter; it can discard common phrases and is not a production detector.
    """
    reserved = set().union(*(shingles(t, n) for t in holdout_texts))
    exact = {normalize(t) for t in holdout_texts}
    return [
        dict(r)
        for r in records
        if normalize(r["text"]) not in exact and not (shingles(r["text"], n) & reserved)
    ]


def sample_mixture(sources, weights, count, seed=0):
    """sources maps names to nonempty lists of records. weights has identical keys, finite
    nonnegative values and positive total. Draw WITH replacement: choose sorted source name
    via local random.Random(seed).choices, then a record via that RNG.choice; return copied
    dicts with added source field. count>=0. Inputs remain unchanged; no global RNG use.
    """
    import math

    if (
        count < 0
        or set(sources) != set(weights)
        or not sources
        or any(not v for v in sources.values())
        or any(not math.isfinite(w) or w < 0 for w in weights.values())
        or sum(weights.values()) <= 0
    ):
        raise ValueError("Invalid mixture")
    rng = random.Random(seed)
    names = sorted(sources)
    out = []
    for _ in range(count):
        name = rng.choices(names, weights=[weights[n] for n in names], k=1)[0]
        out.append(dict(rng.choice(sources[name]), source=name))
    return out


def pack_documents(documents, block_size, pad_id=0):
    """Greedily pack whole nonempty token-ID lists in input order, never split a document.
    Reject block_size<1 or any document longer than it. Return list of dicts: input_ids,
    segment_ids (original document index, padding=-1), position_ids (0..len(doc)-1,
    padding=0), loss_mask (bool at TARGET position: true except each document first token
    and padding). Pad every block to block_size. Empty documents are ignored, but indexes
    still refer to the original input. Empty input returns [].
    """
    if block_size < 1 or any(len(d) > block_size for d in documents):
        raise ValueError("Invalid block size")
    blocks = []
    current = None

    def flush():
        if current is not None:
            size = block_size - len(current["input_ids"])
            for key, value in [
                ("input_ids", pad_id),
                ("segment_ids", -1),
                ("position_ids", 0),
                ("loss_mask", False),
            ]:
                current[key].extend([value] * size)
            blocks.append(current)

    for i, doc in enumerate(documents):
        if not doc:
            continue
        if current is None or len(current["input_ids"]) + len(doc) > block_size:
            flush()
            current = {k: [] for k in ["input_ids", "segment_ids", "position_ids", "loss_mask"]}
        current["input_ids"].extend(doc)
        current["segment_ids"].extend([i] * len(doc))
        current["position_ids"].extend(range(len(doc)))
        current["loss_mask"].extend([False] + [True] * (len(doc) - 1))
    flush()
    return blocks


def attention_mask(segment_ids):
    """Return list[list[bool]] allowed[q][k]: k<=q, equal segment IDs and segment >=0.
    Padding rows are entirely false; callers must avoid softmax over these rows.
    """
    return [
        [k <= q and s >= 0 and s == t for k, t in enumerate(segment_ids)]
        for q, s in enumerate(segment_ids)
    ]


def write_dataset(directory, records):
    """Create directory; write records.jsonl as UTF-8 JSON, sort_keys=True, ensure_ascii=False,
    separators=(',',':'), one newline per record. Write manifest.json containing format
    'toydata-v1', file 'records.jsonl', sha256 of exact bytes, records count. Return manifest.
    Empty datasets have zero bytes. Never overwrite an existing records.jsonl or manifest.json.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "records.jsonl"
    manifest_path = directory / "manifest.json"
    if path.exists() or manifest_path.exists():
        raise FileExistsError("Dataset already exists")
    body = "".join(
        json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for r in records
    ).encode("utf-8")
    path.write_bytes(body)
    manifest = dict(
        format="toydata-v1",
        file="records.jsonl",
        sha256=hashlib.sha256(body).hexdigest(),
        records=len(records),
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest


def prepare_dataset(records, holdout_texts, directory, salt="course"):
    """Integrated preparation: quality_filter defaults, deduplicate, decontaminate defaults,
    duplicate_components defaults, split_groups(.2,salt), then write_dataset. Output copied
    records gain split and content_sha256 fields. Validate unique input IDs before filtering.
    Return manifest. Original text and IDs survive; holdout texts never enter the output.
    """
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate input IDs")
    clean = deduplicate([r for r in records if quality_filter(r["text"])])
    clean = decontaminate(clean, holdout_texts)
    assignments = split_groups(duplicate_components(clean), salt=salt)
    result = [
        dict(r, split=assignments[r["id"]], content_sha256=document_id(r["text"])) for r in clean
    ]
    return write_dataset(directory, result)
