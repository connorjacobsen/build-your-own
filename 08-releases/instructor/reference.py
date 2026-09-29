"""Instructor reference: a local model release registry, not a hosted deployment platform."""

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def canonical_json(value):
    """Return UTF-8 bytes: JSON sorted keys, compact separators, ensure_ascii=False,
    allow_nan=False, followed by one newline. Unsupported/nonfinite values must fail.
    """
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode()


def file_digest(path):
    """SHA256 hex of file bytes, read in chunks no larger than 1 MiB. Empty files allowed."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def safe_artifact_path(root, relative):
    """Return resolved existing regular file inside resolved root. Reject absolute paths,
    empty paths, any '..' component and symlink resolution outside root, with ValueError.
    Internal symlinks to regular files are permitted. Missing/nonfiles raise ValueError.
    """
    root = Path(root).resolve()
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts:
        raise ValueError("Invalid relative path")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Artifact outside bundle or absent")
    return resolved


def build_manifest(root, paths, metadata):
    """Require nonempty unique relative paths. Return format='toyrelease-v1', metadata copied
    via canonical JSON roundtrip, files sorted by path with {path,sha256,size_bytes}.
    Validate each path using safe_artifact_path. Caller selects explicit files, no recursive scan.
    """
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Explicit unique files required")
    files = []
    for name in sorted(paths):
        p = safe_artifact_path(root, name)
        files.append(dict(path=name, sha256=file_digest(p), size_bytes=p.stat().st_size))
    return dict(format="toyrelease-v1", metadata=json.loads(canonical_json(metadata)), files=files)


def verify_manifest(root, manifest):
    """Validate format, nonempty unique file entries, path containment, exact byte sizes and
    hashes. Return True; any mismatch raises ValueError. Unlisted files do not belong to bundle.
    """
    if manifest.get("format") != "toyrelease-v1" or not manifest.get("files"):
        raise ValueError("Invalid manifest")
    names = [r["path"] for r in manifest["files"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate file")
    for row in manifest["files"]:
        path = safe_artifact_path(root, row["path"])
        if path.stat().st_size != row["size_bytes"] or file_digest(path) != row["sha256"]:
            raise ValueError("Artifact integrity mismatch")
    return True


def register_artifact(store, source):
    """Copy source to content-addressed store/<sha256>. Read/write by chunks; atomically
    publish a temporary file with os.replace. Return digest. If object exists, verify it
    matches digest and return without altering it. Corrupt existing objects raise ValueError.
    The source remains unchanged. Clean temporary files on success or failure.
    """
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    digest = file_digest(source)
    target = store / digest
    if target.exists():
        if file_digest(target) != digest:
            raise ValueError("Corrupt object store")
        return digest
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=store, delete=False) as out:
            temporary = Path(out.name)
            with Path(source).open("rb") as inp:
                while chunk := inp.read(1024 * 1024):
                    out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        if file_digest(temporary) != digest:
            raise ValueError("Source changed during registration")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return digest


def validate_evaluation(report, model_sha256, dataset_sha256):
    """Validate toyeval-v1 identities, unique nonempty cases, finite [0,1] scores, metrics.n
    equal case count, metrics.accuracy equal score mean (abs tolerance 1e-12). Both digests
    must be 64 lowercase hex chars and match expected identities. Return True or ValueError.
    This validates internal consistency, not whether an external evaluator was honest.
    """
    try:
        if (
            report["format"] != "toyeval-v1"
            or report["model_sha256"] != model_sha256
            or report["dataset_sha256"] != dataset_sha256
        ):
            raise ValueError("Wrong evaluation identity")
        if any(
            len(d) != 64 or any(c not in "0123456789abcdef" for c in d)
            for d in [model_sha256, dataset_sha256]
        ):
            raise ValueError("Invalid digest")
        rows = report["cases"]
        scores = [r["score"] for r in rows]
        if (
            not rows
            or len({r["id"] for r in rows}) != len(rows)
            or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores)
        ):
            raise ValueError("Invalid cases")
        metrics = report["metrics"]
        if (
            metrics["n"] != len(rows)
            or not math.isfinite(metrics["accuracy"])
            or abs(metrics["accuracy"] - sum(scores) / len(scores)) > 1e-12
        ):
            raise ValueError("Inconsistent aggregate")
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed report") from exc
    return True


def release_gate(baseline, candidate, min_accuracy=0.8, max_drop=0.02, min_cases=20):
    """Validate both reports and require same dataset and case-ID sets. Reject incomparable
    evidence with ValueError. Return {eligible:bool,reasons:list[str]} in fixed order:
    'too_few_cases', 'below_floor', 'regression'. A boundary exactly meeting a threshold passes.
    Require thresholds in [0,1], min_cases>=1. This deterministic gate is policy, not a significance test.
    """
    if not 0 <= min_accuracy <= 1 or not 0 <= max_drop <= 1 or min_cases < 1:
        raise ValueError("Invalid policy")
    validate_evaluation(baseline, baseline["model_sha256"], baseline["dataset_sha256"])
    validate_evaluation(candidate, candidate["model_sha256"], baseline["dataset_sha256"])
    if {r["id"] for r in baseline["cases"]} != {r["id"] for r in candidate["cases"]}:
        raise ValueError("Different evaluation cases")
    reasons = []
    b = baseline["metrics"]
    c = candidate["metrics"]
    if c["n"] < min_cases:
        reasons.append("too_few_cases")
    if c["accuracy"] < min_accuracy:
        reasons.append("below_floor")
    if b["accuracy"] - c["accuracy"] > max_drop + 1e-12:
        reasons.append("regression")
    return dict(eligible=not reasons, reasons=reasons)


def route_request(request_id, candidate_fraction, salt="course"):
    """Deterministic SHA256(salt+':'+request_id) / 2**256 bucket. Return 'candidate' iff
    bucket<fraction else 'baseline'. fraction in [0,1], including exact endpoints.
    This routes one local simulation request; it does not contact a service.
    """
    if not 0 <= candidate_fraction <= 1:
        raise ValueError("Invalid fraction")
    bucket = int(hashlib.sha256((salt + ":" + request_id).encode()).hexdigest(), 16) / 2**256
    return "candidate" if bucket < candidate_fraction else "baseline"


def summarize_requests(records):
    """Nonempty records {status:'ok'|'error',latency_seconds:finite nonnegative}. Return n,
    error_rate, p95_seconds using nearest-rank ceil(.95*n)-1 on ALL requests including errors.
    Reject malformed status or latency. Units remain seconds.
    """
    if not records or any(
        r["status"] not in {"ok", "error"}
        or not math.isfinite(r["latency_seconds"])
        or r["latency_seconds"] < 0
        for r in records
    ):
        raise ValueError("Invalid observations")
    n = len(records)
    times = sorted(r["latency_seconds"] for r in records)
    return dict(
        n=n,
        error_rate=sum(r["status"] == "error" for r in records) / n,
        p95_seconds=times[math.ceil(0.95 * n) - 1],
    )


def rollback_decision(observed, max_error=0.05, max_p95=2.0, min_requests=20):
    """Given summarize_requests output, return 'wait' below min_requests; else 'rollback'
    if error_rate>max_error OR p95_seconds>max_p95; otherwise 'keep'. Threshold equality passes.
    Validate finite metrics and policy, n>=0, rates in [0,1], latencies>=0,min_requests>=1.
    """
    n = observed["n"]
    e = observed["error_rate"]
    p = observed["p95_seconds"]
    if (
        min_requests < 1
        or n < 0
        or not all(math.isfinite(x) for x in [e, p, max_error, max_p95])
        or not 0 <= e <= 1
        or p < 0
        or not 0 <= max_error <= 1
        or max_p95 < 0
    ):
        raise ValueError("Invalid monitoring policy")
    if n < min_requests:
        return "wait"
    return "rollback" if e > max_error or p > max_p95 else "keep"


def promote(pointer, new_digest, expected_current):
    """Local POSIX compare-and-swap pointer. JSON {current,previous}; absent means current=None.
    Acquire exclusive flock on sibling <name>.lock, reread pointer, require current==expected
    else RuntimeError, then atomically replace pointer with current=new_digest,previous=old.
    New digest is 64 lowercase hex chars. Same-current promotion is idempotent and preserves
    previous. Return resulting dict. Create parent directories. No remote deployment occurs.
    """
    if len(new_digest) != 64 or any(c not in "0123456789abcdef" for c in new_digest):
        raise ValueError("Invalid digest")
    path = Path(pointer)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / (path.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = json.loads(path.read_text()) if path.exists() else dict(current=None, previous=None)
        if old["current"] != expected_current:
            raise RuntimeError("Concurrent promotion conflict")
        if old["current"] == new_digest:
            return old
        record = dict(current=new_digest, previous=old["current"])
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(canonical_json(record))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return record


def rollback(pointer, expected_current):
    """Read previous digest and call promote(pointer,previous,expected_current). Reject no
    prior release with ValueError. The compare-and-swap rejects a concurrent pointer change.
    A successful rollback records the rolled-back release as previous, enabling explicit undo.
    """
    record = json.loads(Path(pointer).read_text())
    if record["previous"] is None:
        raise ValueError("No prior release")
    return promote(pointer, record["previous"], expected_current)


def release_model(
    model_path,
    report_path,
    baseline,
    store,
    pointer,
    expected_current,
    min_accuracy=0.8,
    max_drop=0.02,
    min_cases=20,
):
    """Integrated local release: compute model file digest, parse candidate report, validate
    its model/dataset identity against actual bytes and baseline, apply release_gate.
    If ineligible raise ValueError BEFORE storing artifacts or changing pointer. Otherwise
    register model and report, promote with expected_current, return {model_sha256,
    report_sha256,pointer}. Report integrity and model quality remain separate claims.
    """
    model_digest = file_digest(model_path)
    report = json.loads(Path(report_path).read_text())
    validate_evaluation(report, model_digest, baseline["dataset_sha256"])
    decision = release_gate(baseline, report, min_accuracy, max_drop, min_cases)
    if not decision["eligible"]:
        raise ValueError(",".join(decision["reasons"]))
    model_digest = register_artifact(store, model_path)
    report_digest = register_artifact(store, report_path)
    result = promote(pointer, model_digest, expected_current)
    return dict(model_sha256=model_digest, report_sha256=report_digest, pointer=result)
