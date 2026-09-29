"""Instructor reference for a small auditable evaluation harness."""

import hashlib
import json
import math
from pathlib import Path
import string
import numpy as np
import torch


def normalize_answer(text):
    """casefold, remove ASCII punctuation (not whitespace), then collapse whitespace.
    No article removal, numeric coercion or Unicode punctuation removal in this policy.
    """
    return " ".join(text.casefold().translate(str.maketrans("", "", string.punctuation)).split())


def exact_match(prediction, acceptable):
    """Return float 1.0 if normalized prediction matches any acceptable string, else 0.0.
    acceptable must be a nonempty list; empty lists raise ValueError.
    """
    if not acceptable:
        raise ValueError("At least one reference required")
    return float(any(normalize_answer(prediction) == normalize_answer(t) for t in acceptable))


def continuation_logp(logits, ids, prompt_length):
    """logits [T,V] correspond to ids [T]. Sum next-token log probabilities for targets
    ids[prompt_length:] using rows prompt_length-1 through T-2. Require 1<=prompt_length<T.
    Exclude prompt tokens. Return Python float. Caller supplies concatenated prompt+response.
    """
    if not 1 <= prompt_length < len(ids) or logits.shape[0] != len(ids):
        raise ValueError("Invalid prompt boundary")
    logp = logits[prompt_length - 1 : -1].log_softmax(-1)
    targets = torch.tensor(ids[prompt_length:], device=logits.device)
    return float(logp.gather(-1, targets[:, None]).sum())


def rank_choices(logps, token_counts, normalize=False):
    """Return winning index, maximum summed logp or logp/count when normalize=True.
    Equal scores choose earliest index. Require nonempty equal-length inputs, positive
    counts and finite scores. This exposes two different evaluation protocols explicitly.
    """
    if (
        not logps
        or len(logps) != len(token_counts)
        or any(n <= 0 for n in token_counts)
        or not all(math.isfinite(x) for x in logps)
    ):
        raise ValueError("Invalid scores")
    scores = [s / n if normalize else s for s, n in zip(logps, token_counts)]
    return max(range(len(scores)), key=scores.__getitem__)


def perplexity(total_nlls, token_counts):
    """exp(sum total NLL / sum token counts); lists nonempty, equal length; counts>0;
    NLLs finite and nonnegative. Return float (possibly inf for overflow).
    """
    if (
        not total_nlls
        or len(total_nlls) != len(token_counts)
        or any(n <= 0 for n in token_counts)
        or any(not math.isfinite(x) or x < 0 for x in total_nlls)
    ):
        raise ValueError("Invalid totals")
    value = sum(total_nlls) / sum(token_counts)
    return math.exp(value) if value < 709 else float("inf")


def confusion_matrix(labels, predictions, classes):
    """Return int64 NumPy [C,C] counts, rows truth and columns prediction. Equal lengths,
    classes>=1, IDs in [0,C). Empty inputs produce zeros. Invalid values raise ValueError.
    """
    if (
        classes < 1
        or len(labels) != len(predictions)
        or any(not 0 <= x < classes for x in list(labels) + list(predictions))
    ):
        raise ValueError("Invalid labels")
    result = np.zeros((classes, classes), dtype=np.int64)
    for y, p in zip(labels, predictions):
        result[y, p] += 1
    return result


def classification_metrics(matrix):
    """Given nonnegative square count matrix, return accuracy, macro_f1, per_class_f1 list.
    Zero-denominator F1 is zero; macro includes ALL declared classes. Empty total accuracy=0.
    Reject nonsquare/negative matrices.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] == 0 or np.any(m < 0):
        raise ValueError("Invalid confusion matrix")
    tp = np.diag(m)
    den = m.sum(0) + m.sum(1)
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    return dict(
        accuracy=float(tp.sum() / m.sum()) if m.sum() else 0.0,
        macro_f1=float(f1.mean()),
        per_class_f1=f1.tolist(),
    )


def paired_bootstrap(baseline, candidate, seed=0, resamples=2000, confidence=0.95):
    """Equal nonempty finite 1D arrays of per-example scores, larger=better. Resample paired
    indices with local np.random.default_rng(seed).integers(0,N,size=(resamples,N)); return
    dict delta=mean(candidate-baseline), low/high=quantiles at (1-confidence)/2 and complement.
    resamples>=1 and 0<confidence<1. Never bootstrap the two models independently.
    """
    a = np.asarray(baseline, float)
    b = np.asarray(candidate, float)
    if (
        a.ndim != 1
        or a.shape != b.shape
        or not len(a)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or resamples < 1
        or not 0 < confidence < 1
    ):
        raise ValueError("Invalid bootstrap")
    delta = b - a
    rng = np.random.default_rng(seed)
    values = delta[rng.integers(0, len(a), size=(resamples, len(a)))].mean(1)
    lo = (1 - confidence) / 2
    return dict(
        delta=float(delta.mean()),
        low=float(np.quantile(values, lo)),
        high=float(np.quantile(values, 1 - lo)),
    )


def compare_by_id(baseline, candidate, seed=0):
    """Each list contains {'id':str,'score':float}. Require identical nonempty unique ID sets.
    Align sorted IDs, then paired_bootstrap with defaults and supplied seed. Return its dict.
    Reordering input rows must never alter pairing or RNG interpretation.
    """
    a = {r["id"]: r["score"] for r in baseline}
    b = {r["id"]: r["score"] for r in candidate}
    if len(a) != len(baseline) or len(b) != len(candidate) or set(a) != set(b) or not a:
        raise ValueError("Incomparable cases")
    ids = sorted(a)
    return paired_bootstrap([a[i] for i in ids], [b[i] for i in ids], seed=seed)


def slice_metrics(records):
    """Records {'group':str,'correct':bool}. Return micro_accuracy, macro_accuracy,
    worst_accuracy, groups={name:{n,accuracy}}. Require nonempty input; each row belongs to one
    group. Macro weights groups equally; micro weights individual examples equally.
    """
    if not records:
        raise ValueError("No cases")
    groups = {}
    for r in records:
        groups.setdefault(r["group"], []).append(bool(r["correct"]))
    details = {k: dict(n=len(v), accuracy=sum(v) / len(v)) for k, v in sorted(groups.items())}
    values = [d["accuracy"] for d in details.values()]
    return dict(
        micro_accuracy=sum(bool(r["correct"]) for r in records) / len(records),
        macro_accuracy=sum(values) / len(values),
        worst_accuracy=min(values),
        groups=details,
    )


def calibration_error(confidences, correct, bins=10):
    """Equal nonempty 1D lists; confidence finite in [0,1], bins>=1. ECE weighted absolute
    gap between mean confidence and accuracy within equal-width bins. Internal boundaries
    belong to the higher bin; confidence=1 belongs to final bin. Empty bins contribute zero.
    """
    c = np.asarray(confidences, float)
    y = np.asarray(correct, float)
    if (
        bins < 1
        or c.ndim != 1
        or c.shape != y.shape
        or not len(c)
        or not np.isfinite(c).all()
        or np.any(c < 0)
        or np.any(c > 1)
        or np.any((y != 0) & (y != 1))
    ):
        raise ValueError("Invalid calibration data")
    indexes = np.minimum((c * bins).astype(int), bins - 1)
    ece = 0.0
    for i in range(bins):
        mask = indexes == i
        if mask.any():
            ece += mask.mean() * abs(c[mask].mean() - y[mask].mean())
    return float(ece)


def latency_metrics(arrival, token_times):
    """Monotonic finite timestamps in seconds, nonempty tokens, first>=arrival. Return
    ttft, inter_token_seconds list, decode_tokens_per_second=(N-1)/(last-first) for N>1.
    For one token decode throughput is None; zero decode duration for N>1 gives inf.
    Do not count the prefill-generated first token in decode throughput.
    """
    if (
        not token_times
        or not all(math.isfinite(t) for t in [arrival, *token_times])
        or token_times[0] < arrival
        or any(b < a for a, b in zip(token_times, token_times[1:]))
    ):
        raise ValueError("Invalid event times")
    gaps = [b - a for a, b in zip(token_times, token_times[1:])]
    rate = (
        None
        if not gaps
        else (
            len(gaps) / (token_times[-1] - token_times[0])
            if token_times[-1] > token_times[0]
            else float("inf")
        )
    )
    return dict(
        ttft=token_times[0] - arrival, inter_token_seconds=gaps, decode_tokens_per_second=rate
    )


def evaluate(predict, cases, model_sha256, dataset_sha256):
    """Each case has unique id, prompt:str, acceptable:list[str]. Call predict(prompt) once
    per case, preserving order; never expose reference answers to predict. Require nonempty
    cases and valid 64-lowercase-hex digests. Prediction must be str, else ValueError.
    Return toyeval-v1 dict with model_sha256,dataset_sha256, metrics={accuracy,n}, and
    cases=[{id,prediction,score}]. No silent exception swallowing or dropping failed cases.
    """
    digests = [model_sha256, dataset_sha256]
    if (
        not cases
        or len({c["id"] for c in cases}) != len(cases)
        or any(len(d) != 64 or any(x not in "0123456789abcdef" for x in d) for d in digests)
    ):
        raise ValueError("Invalid evaluation identity")
    rows = []
    for case in cases:
        result = predict(case["prompt"])
        if not isinstance(result, str):
            raise ValueError("Non-text prediction")
        rows.append(
            dict(id=case["id"], prediction=result, score=exact_match(result, case["acceptable"]))
        )
    return dict(
        format="toyeval-v1",
        model_sha256=model_sha256,
        dataset_sha256=dataset_sha256,
        metrics=dict(accuracy=sum(r["score"] for r in rows) / len(rows), n=len(rows)),
        cases=rows,
    )


def write_report(path, report):
    """Serialize report as UTF-8 canonical JSON (sorted keys, compact separators,
    ensure_ascii=False, allow_nan=False) followed by newline. Return exact-byte SHA256.
    Refuse existing destination. Validate serialization before creating file.
    """
    body = (
        json.dumps(
            report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode()
    with Path(path).open("xb") as stream:
        stream.write(body)
    return hashlib.sha256(body).hexdigest()
