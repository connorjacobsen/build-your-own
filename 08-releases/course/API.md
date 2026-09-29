# Public API and input domains

These are scaffold contracts, not implementations. Earlier stages remain dependencies of later stages.

## `canonical_json`

Return UTF-8 bytes: JSON sorted keys, compact separators, ensure_ascii=False,
allow_nan=False, followed by one newline. Unsupported/nonfinite values must fail.

Signature: `canonical_json(value)`

## `file_digest`

SHA256 hex of file bytes, read in chunks no larger than 1 MiB. Empty files allowed.

Signature: `file_digest(path)`

## `safe_artifact_path`

Return resolved existing regular file inside resolved root. Reject absolute paths,
empty paths, any '..' component and symlink resolution outside root, with ValueError.
Internal symlinks to regular files are permitted. Missing/nonfiles raise ValueError.

Signature: `safe_artifact_path(root, relative)`

## `build_manifest`

Require nonempty unique relative paths. Return format='toyrelease-v1', metadata copied
via canonical JSON roundtrip, files sorted by path with {path,sha256,size_bytes}.
Validate each path using safe_artifact_path. Caller selects explicit files, no recursive scan.

Signature: `build_manifest(root, paths, metadata)`

## `verify_manifest`

Validate format, nonempty unique file entries, path containment, exact byte sizes and
hashes. Return True; any mismatch raises ValueError. Unlisted files do not belong to bundle.

Signature: `verify_manifest(root, manifest)`

## `register_artifact`

Copy source to content-addressed store/<sha256>. Read/write by chunks; atomically
publish a temporary file with os.replace. Return digest. If object exists, verify it
matches digest and return without altering it. Corrupt existing objects raise ValueError.
The source remains unchanged. Clean temporary files on success or failure.

Signature: `register_artifact(store, source)`

## `validate_evaluation`

Validate toyeval-v1 identities, unique nonempty cases, finite [0,1] scores, metrics.n
equal case count, metrics.accuracy equal score mean (abs tolerance 1e-12). Both digests
must be 64 lowercase hex chars and match expected identities. Return True or ValueError.
This validates internal consistency, not whether an external evaluator was honest.

Signature: `validate_evaluation(report, model_sha256, dataset_sha256)`

## `release_gate`

Validate both reports and require same dataset and case-ID sets. Reject incomparable
evidence with ValueError. Return {eligible:bool,reasons:list[str]} in fixed order:
'too_few_cases', 'below_floor', 'regression'. A boundary exactly meeting a threshold passes.
Require thresholds in [0,1], min_cases>=1. This deterministic gate is policy, not a significance test.

Signature: `release_gate(baseline, candidate, min_accuracy=0.8, max_drop=0.02, min_cases=20)`

## `route_request`

Deterministic SHA256(salt+':'+request_id) / 2**256 bucket. Return 'candidate' iff
bucket<fraction else 'baseline'. fraction in [0,1], including exact endpoints.
This routes one local simulation request; it does not contact a service.

Signature: `route_request(request_id, candidate_fraction, salt='course')`

## `summarize_requests`

Nonempty records {status:'ok'|'error',latency_seconds:finite nonnegative}. Return n,
error_rate, p95_seconds using nearest-rank ceil(.95*n)-1 on ALL requests including errors.
Reject malformed status or latency. Units remain seconds.

Signature: `summarize_requests(records)`

## `rollback_decision`

Given summarize_requests output, return 'wait' below min_requests; else 'rollback'
if error_rate>max_error OR p95_seconds>max_p95; otherwise 'keep'. Threshold equality passes.
Validate finite metrics and policy, n>=0, rates in [0,1], latencies>=0,min_requests>=1.

Signature: `rollback_decision(observed, max_error=0.05, max_p95=2.0, min_requests=20)`

## `promote`

Local POSIX compare-and-swap pointer. JSON {current,previous}; absent means current=None.
Acquire exclusive flock on sibling <name>.lock, reread pointer, require current==expected
else RuntimeError, then atomically replace pointer with current=new_digest,previous=old.
New digest is 64 lowercase hex chars. Same-current promotion is idempotent and preserves
previous. Return resulting dict. Create parent directories. No remote deployment occurs.

Signature: `promote(pointer, new_digest, expected_current)`

## `rollback`

Read previous digest and call promote(pointer,previous,expected_current). Reject no
prior release with ValueError. The compare-and-swap rejects a concurrent pointer change.
A successful rollback records the rolled-back release as previous, enabling explicit undo.

Signature: `rollback(pointer, expected_current)`

## `release_model`

Integrated local release: compute model file digest, parse candidate report, validate
its model/dataset identity against actual bytes and baseline, apply release_gate.
If ineligible raise ValueError BEFORE storing artifacts or changing pointer. Otherwise
register model and report, promote with expected_current, return {model_sha256,
report_sha256,pointer}. Report integrity and model quality remain separate claims.

Signature: `release_model(model_path, report_path, baseline, store, pointer, expected_current, min_accuracy=0.8, max_drop=0.02, min_cases=20)`
