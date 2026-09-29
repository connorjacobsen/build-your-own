# Public API and input domains

These are scaffold contracts, not implementations. Earlier stages remain dependencies of later stages.

## `normalize_answer`

casefold, remove ASCII punctuation (not whitespace), then collapse whitespace.
No article removal, numeric coercion or Unicode punctuation removal in this policy.

Signature: `normalize_answer(text)`

## `exact_match`

Return float 1.0 if normalized prediction matches any acceptable string, else 0.0.
acceptable must be a nonempty list; empty lists raise ValueError.

Signature: `exact_match(prediction, acceptable)`

## `continuation_logp`

logits [T,V] correspond to ids [T]. Sum next-token log probabilities for targets
ids[prompt_length:] using rows prompt_length-1 through T-2. Require 1<=prompt_length<T.
Exclude prompt tokens. Return Python float. Caller supplies concatenated prompt+response.

Signature: `continuation_logp(logits, ids, prompt_length)`

## `rank_choices`

Return winning index, maximum summed logp or logp/count when normalize=True.
Equal scores choose earliest index. Require nonempty equal-length inputs, positive
counts and finite scores. This exposes two different evaluation protocols explicitly.

Signature: `rank_choices(logps, token_counts, normalize=False)`

## `perplexity`

exp(sum total NLL / sum token counts); lists nonempty, equal length; counts>0;
NLLs finite and nonnegative. Return float (possibly inf for overflow).

Signature: `perplexity(total_nlls, token_counts)`

## `confusion_matrix`

Return int64 NumPy [C,C] counts, rows truth and columns prediction. Equal lengths,
classes>=1, IDs in [0,C). Empty inputs produce zeros. Invalid values raise ValueError.

Signature: `confusion_matrix(labels, predictions, classes)`

## `classification_metrics`

Given nonnegative square count matrix, return accuracy, macro_f1, per_class_f1 list.
Zero-denominator F1 is zero; macro includes ALL declared classes. Empty total accuracy=0.
Reject nonsquare/negative matrices.

Signature: `classification_metrics(matrix)`

## `paired_bootstrap`

Equal nonempty finite 1D arrays of per-example scores, larger=better. Resample paired
indices with local np.random.default_rng(seed).integers(0,N,size=(resamples,N)); return
dict delta=mean(candidate-baseline), low/high=quantiles at (1-confidence)/2 and complement.
resamples>=1 and 0<confidence<1. Never bootstrap the two models independently.

Signature: `paired_bootstrap(baseline, candidate, seed=0, resamples=2000, confidence=0.95)`

## `compare_by_id`

Each list contains {'id':str,'score':float}. Require identical nonempty unique ID sets.
Align sorted IDs, then paired_bootstrap with defaults and supplied seed. Return its dict.
Reordering input rows must never alter pairing or RNG interpretation.

Signature: `compare_by_id(baseline, candidate, seed=0)`

## `slice_metrics`

Records {'group':str,'correct':bool}. Return micro_accuracy, macro_accuracy,
worst_accuracy, groups={name:{n,accuracy}}. Require nonempty input; each row belongs to one
group. Macro weights groups equally; micro weights individual examples equally.

Signature: `slice_metrics(records)`

## `calibration_error`

Equal nonempty 1D lists; confidence finite in [0,1], bins>=1. ECE weighted absolute
gap between mean confidence and accuracy within equal-width bins. Internal boundaries
belong to the higher bin; confidence=1 belongs to final bin. Empty bins contribute zero.

Signature: `calibration_error(confidences, correct, bins=10)`

## `latency_metrics`

Monotonic finite timestamps in seconds, nonempty tokens, first>=arrival. Return
ttft, inter_token_seconds list, decode_tokens_per_second=(N-1)/(last-first) for N>1.
For one token decode throughput is None; zero decode duration for N>1 gives inf.
Do not count the prefill-generated first token in decode throughput.

Signature: `latency_metrics(arrival, token_times)`

## `evaluate`

Each case has unique id, prompt:str, acceptable:list[str]. Call predict(prompt) once
per case, preserving order; never expose reference answers to predict. Require nonempty
cases and valid 64-lowercase-hex digests. Prediction must be str, else ValueError.
Return toyeval-v1 dict with model_sha256,dataset_sha256, metrics={accuracy,n}, and
cases=[{id,prediction,score}]. No silent exception swallowing or dropping failed cases.

Signature: `evaluate(predict, cases, model_sha256, dataset_sha256)`

## `write_report`

Serialize report as UTF-8 canonical JSON (sorted keys, compact separators,
ensure_ascii=False, allow_nan=False) followed by newline. Return exact-byte SHA256.
Refuse existing destination. Validate serialization before creating file.

Signature: `write_report(path, report)`
