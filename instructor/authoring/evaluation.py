from build import build
source='''"""Instructor reference for a small auditable evaluation harness."""
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
    return ' '.join(text.casefold().translate(str.maketrans('','',string.punctuation)).split())

def exact_match(prediction, acceptable):
    """Return float 1.0 if normalized prediction matches any acceptable string, else 0.0.
    acceptable must be a nonempty list; empty lists raise ValueError.
    """
    if not acceptable: raise ValueError('At least one reference required')
    return float(any(normalize_answer(prediction)==normalize_answer(t) for t in acceptable))

def continuation_logp(logits, ids, prompt_length):
    """logits [T,V] correspond to ids [T]. Sum next-token log probabilities for targets
    ids[prompt_length:] using rows prompt_length-1 through T-2. Require 1<=prompt_length<T.
    Exclude prompt tokens. Return Python float. Caller supplies concatenated prompt+response.
    """
    if not 1<=prompt_length<len(ids) or logits.shape[0]!=len(ids): raise ValueError('Invalid prompt boundary')
    logp=logits[prompt_length-1:-1].log_softmax(-1)
    targets=torch.tensor(ids[prompt_length:],device=logits.device)
    return float(logp.gather(-1,targets[:,None]).sum())

def rank_choices(logps, token_counts, normalize=False):
    """Return winning index, maximum summed logp or logp/count when normalize=True.
    Equal scores choose earliest index. Require nonempty equal-length inputs, positive
    counts and finite scores. This exposes two different evaluation protocols explicitly.
    """
    if not logps or len(logps)!=len(token_counts) or any(n<=0 for n in token_counts) or not all(math.isfinite(x) for x in logps): raise ValueError('Invalid scores')
    scores=[s/n if normalize else s for s,n in zip(logps,token_counts)]
    return max(range(len(scores)),key=scores.__getitem__)

def perplexity(total_nlls, token_counts):
    """exp(sum total NLL / sum token counts); lists nonempty, equal length; counts>0;
    NLLs finite and nonnegative. Return float (possibly inf for overflow).
    """
    if not total_nlls or len(total_nlls)!=len(token_counts) or any(n<=0 for n in token_counts) or any(not math.isfinite(x) or x<0 for x in total_nlls): raise ValueError('Invalid totals')
    value=sum(total_nlls)/sum(token_counts)
    return math.exp(value) if value<709 else float('inf')

def confusion_matrix(labels, predictions, classes):
    """Return int64 NumPy [C,C] counts, rows truth and columns prediction. Equal lengths,
    classes>=1, IDs in [0,C). Empty inputs produce zeros. Invalid values raise ValueError.
    """
    if classes<1 or len(labels)!=len(predictions) or any(not 0<=x<classes for x in list(labels)+list(predictions)): raise ValueError('Invalid labels')
    result=np.zeros((classes,classes),dtype=np.int64)
    for y,p in zip(labels,predictions): result[y,p]+=1
    return result

def classification_metrics(matrix):
    """Given nonnegative square count matrix, return accuracy, macro_f1, per_class_f1 list.
    Zero-denominator F1 is zero; macro includes ALL declared classes. Empty total accuracy=0.
    Reject nonsquare/negative matrices.
    """
    m=np.asarray(matrix,dtype=float)
    if m.ndim!=2 or m.shape[0]!=m.shape[1] or m.shape[0]==0 or np.any(m<0): raise ValueError('Invalid confusion matrix')
    tp=np.diag(m); den=m.sum(0)+m.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0)
    return dict(accuracy=float(tp.sum()/m.sum()) if m.sum() else 0.,macro_f1=float(f1.mean()),per_class_f1=f1.tolist())

def paired_bootstrap(baseline, candidate, seed=0, resamples=2000, confidence=.95):
    """Equal nonempty finite 1D arrays of per-example scores, larger=better. Resample paired
    indices with local np.random.default_rng(seed).integers(0,N,size=(resamples,N)); return
    dict delta=mean(candidate-baseline), low/high=quantiles at (1-confidence)/2 and complement.
    resamples>=1 and 0<confidence<1. Never bootstrap the two models independently.
    """
    a=np.asarray(baseline,float); b=np.asarray(candidate,float)
    if a.ndim!=1 or a.shape!=b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all() or resamples<1 or not 0<confidence<1: raise ValueError('Invalid bootstrap')
    delta=b-a; rng=np.random.default_rng(seed); values=delta[rng.integers(0,len(a),size=(resamples,len(a)))].mean(1)
    lo=(1-confidence)/2
    return dict(delta=float(delta.mean()),low=float(np.quantile(values,lo)),high=float(np.quantile(values,1-lo)))

def compare_by_id(baseline, candidate, seed=0):
    """Each list contains {'id':str,'score':float}. Require identical nonempty unique ID sets.
    Align sorted IDs, then paired_bootstrap with defaults and supplied seed. Return its dict.
    Reordering input rows must never alter pairing or RNG interpretation.
    """
    a={r['id']:r['score'] for r in baseline}; b={r['id']:r['score'] for r in candidate}
    if len(a)!=len(baseline) or len(b)!=len(candidate) or set(a)!=set(b) or not a: raise ValueError('Incomparable cases')
    ids=sorted(a)
    return paired_bootstrap([a[i] for i in ids],[b[i] for i in ids],seed=seed)

def slice_metrics(records):
    """Records {'group':str,'correct':bool}. Return micro_accuracy, macro_accuracy,
    worst_accuracy, groups={name:{n,accuracy}}. Require nonempty input; each row belongs to one
    group. Macro weights groups equally; micro weights individual examples equally.
    """
    if not records: raise ValueError('No cases')
    groups={}
    for r in records: groups.setdefault(r['group'],[]).append(bool(r['correct']))
    details={k:dict(n=len(v),accuracy=sum(v)/len(v)) for k,v in sorted(groups.items())}
    values=[d['accuracy'] for d in details.values()]
    return dict(micro_accuracy=sum(bool(r['correct']) for r in records)/len(records),macro_accuracy=sum(values)/len(values),worst_accuracy=min(values),groups=details)

def calibration_error(confidences, correct, bins=10):
    """Equal nonempty 1D lists; confidence finite in [0,1], bins>=1. ECE weighted absolute
    gap between mean confidence and accuracy within equal-width bins. Internal boundaries
    belong to the higher bin; confidence=1 belongs to final bin. Empty bins contribute zero.
    """
    c=np.asarray(confidences,float); y=np.asarray(correct,float)
    if bins<1 or c.ndim!=1 or c.shape!=y.shape or not len(c) or not np.isfinite(c).all() or np.any(c<0) or np.any(c>1) or np.any((y!=0)&(y!=1)): raise ValueError('Invalid calibration data')
    indexes=np.minimum((c*bins).astype(int),bins-1); ece=0.
    for i in range(bins):
        mask=indexes==i
        if mask.any(): ece+=mask.mean()*abs(c[mask].mean()-y[mask].mean())
    return float(ece)

def latency_metrics(arrival, token_times):
    """Monotonic finite timestamps in seconds, nonempty tokens, first>=arrival. Return
    ttft, inter_token_seconds list, decode_tokens_per_second=(N-1)/(last-first) for N>1.
    For one token decode throughput is None; zero decode duration for N>1 gives inf.
    Do not count the prefill-generated first token in decode throughput.
    """
    if not token_times or not all(math.isfinite(t) for t in [arrival,*token_times]) or token_times[0]<arrival or any(b<a for a,b in zip(token_times,token_times[1:])): raise ValueError('Invalid event times')
    gaps=[b-a for a,b in zip(token_times,token_times[1:])]
    rate=None if not gaps else (len(gaps)/(token_times[-1]-token_times[0]) if token_times[-1]>token_times[0] else float('inf'))
    return dict(ttft=token_times[0]-arrival,inter_token_seconds=gaps,decode_tokens_per_second=rate)

def evaluate(predict, cases, model_sha256, dataset_sha256):
    """Each case has unique id, prompt:str, acceptable:list[str]. Call predict(prompt) once
    per case, preserving order; never expose reference answers to predict. Require nonempty
    cases and valid 64-lowercase-hex digests. Prediction must be str, else ValueError.
    Return toyeval-v1 dict with model_sha256,dataset_sha256, metrics={accuracy,n}, and
    cases=[{id,prediction,score}]. No silent exception swallowing or dropping failed cases.
    """
    digests=[model_sha256,dataset_sha256]
    if not cases or len({c['id'] for c in cases})!=len(cases) or any(len(d)!=64 or any(x not in '0123456789abcdef' for x in d) for d in digests): raise ValueError('Invalid evaluation identity')
    rows=[]
    for case in cases:
        result=predict(case['prompt'])
        if not isinstance(result,str): raise ValueError('Non-text prediction')
        rows.append(dict(id=case['id'],prediction=result,score=exact_match(result,case['acceptable'])))
    return dict(format='toyeval-v1',model_sha256=model_sha256,dataset_sha256=dataset_sha256,metrics=dict(accuracy=sum(r['score'] for r in rows)/len(rows),n=len(rows)),cases=rows)

def write_report(path, report):
    """Serialize report as UTF-8 canonical JSON (sorted keys, compact separators,
    ensure_ascii=False, allow_nan=False) followed by newline. Return exact-byte SHA256.
    Refuse existing destination. Validate serialization before creating file.
    """
    body=(json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\\n').encode()
    with Path(path).open('xb') as stream: stream.write(body)
    return hashlib.sha256(body).hexdigest()
'''
tests='''import hashlib
import json
import math
import numpy as np
import pytest
import torch


def test_01(api):
    assert api.normalize_answer('  The CAT!  ')=='the cat'
    assert api.normalize_answer('a-b')=='ab' and api.normalize_answer('a b')=='a b'
    assert api.normalize_answer('Straße')=='strasse'


def test_02(api):
    assert api.exact_match('Paris!',['paris','city of paris'])==1.
    assert api.exact_match('the paris',['paris'])==0.
    with pytest.raises(ValueError): api.exact_match('',[])


def test_03(api,seed):
    g=torch.Generator().manual_seed(seed); x=torch.randn(5,7,generator=g); ids=[1,2,3,4,5]
    expected=sum(float(x[i-1].log_softmax(0)[ids[i]]) for i in range(2,5))
    assert api.continuation_logp(x,ids,2)==pytest.approx(expected)
    altered=x.clone(); altered[0]+=torch.arange(7)*100; altered[-1]*=100
    assert api.continuation_logp(altered,ids,2)==pytest.approx(expected)
    with pytest.raises(ValueError): api.continuation_logp(x,ids,0)


def test_04(api):
    assert api.rank_choices([-2.,-3.],[1,3])==0
    assert api.rank_choices([-2.,-3.],[1,3],True)==1
    assert api.rank_choices([-1.,-1.],[1,1])==0
    with pytest.raises(ValueError): api.rank_choices([-1],[0])


def test_05(api):
    assert api.perplexity([2.,12.],[1,3])==pytest.approx(math.exp(3.5))
    assert api.perplexity([0.],[7])==1
    with pytest.raises(ValueError): api.perplexity([1.],[0])


def test_06(api):
    m=api.confusion_matrix([0,0,1,2],[0,1,1,0],3)
    np.testing.assert_array_equal(m,[[1,1,0],[0,1,0],[1,0,0]])
    assert m.dtype==np.int64
    with pytest.raises(ValueError): api.confusion_matrix([3],[0],3)


def test_07(api):
    r=api.classification_metrics([[1,1,0],[0,1,0],[1,0,0]])
    assert r['accuracy']==.5 and r['macro_f1']==pytest.approx((.5+2/3)/3)
    assert r['per_class_f1']==pytest.approx([.5,2/3,0])
    assert api.classification_metrics(np.zeros((2,2)))['macro_f1']==0


def test_08(api,seed):
    a=[0,1,0,1]; b=[1,1,1,1]; r=api.paired_bootstrap(a,b,seed,resamples=100)
    delta=np.array(b)-a; rng=np.random.default_rng(seed); samples=delta[rng.integers(0,4,size=(100,4))].mean(1)
    assert r==dict(delta=.5,low=float(np.quantile(samples,.025)),high=float(np.quantile(samples,.975)))
    zero=api.paired_bootstrap(a,a,seed); assert zero==dict(delta=0,low=0,high=0)
    constant=api.paired_bootstrap(a,np.array(a)+.2,seed); assert constant['low']==pytest.approx(.2) and constant['high']==pytest.approx(.2)


def test_09(api,seed):
    a=[dict(id='a',score=0),dict(id='b',score=1)]; b=[dict(id='b',score=0),dict(id='a',score=1)]
    assert api.compare_by_id(a,b,seed)==api.compare_by_id(a[::-1],b[::-1],seed)
    assert api.compare_by_id(a,b,seed)['delta']==0
    with pytest.raises(ValueError): api.compare_by_id(a,b[:1])
    with pytest.raises(ValueError): api.compare_by_id(a,[b[0],b[0]])


def test_10(api):
    rows=[dict(group='large',correct=True)]*9+[dict(group='small',correct=False)]
    r=api.slice_metrics(rows); assert r['micro_accuracy']==.9 and r['macro_accuracy']==.5 and r['worst_accuracy']==0
    assert r['groups']['small']==dict(n=1,accuracy=0)


def test_11(api):
    assert api.calibration_error([.1,.5,1.],[False,True,True],2)==pytest.approx((.1+2*.25)/3)
    assert api.calibration_error([0,1],[False,True])==0
    with pytest.raises(ValueError): api.calibration_error([1.1],[True])


def test_12(api):
    r=api.latency_metrics(1,[1.5,1.7,2.]); assert r['ttft']==.5 and r['inter_token_seconds']==pytest.approx([.2,.3]) and r['decode_tokens_per_second']==4
    assert api.latency_metrics(0,[1])['decode_tokens_per_second'] is None
    with pytest.raises(ValueError): api.latency_metrics(0,[2,1])


def test_13(api):
    seen=[]
    def predict(prompt): seen.append(prompt); return 'yes'
    cases=[dict(id='a',prompt='question 1',acceptable=['yes']),dict(id='b',prompt='question 2',acceptable=['no'])]
    r=api.evaluate(predict,cases,'a'*64,'b'*64)
    assert seen==['question 1','question 2'] and r['metrics']==dict(accuracy=.5,n=2) and r['format']=='toyeval-v1'
    assert r['model_sha256']=='a'*64 and [x['id'] for x in r['cases']]==['a','b']
    with pytest.raises(ValueError): api.evaluate(predict,[cases[0],cases[0]],'a'*64,'b'*64)
    def bad(_): raise RuntimeError('model failed')
    with pytest.raises(RuntimeError): api.evaluate(bad,cases,'a'*64,'b'*64)


def test_14(api,tmp_path):
    path=tmp_path/'eval.json'; report={'z':2,'a':'é'}; digest=api.write_report(path,report)
    assert path.read_bytes()=='{"a":"é","z":2}\\n'.encode() and digest==hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError): api.write_report(path,{})
    with pytest.raises(ValueError): api.write_report(tmp_path/'bad.json',{'n':float('nan')})
    assert not (tmp_path/'bad.json').exists()
'''
rows=[
('Specify answer normalization','normalize_answer','A metric embeds decisions about what counts as the same answer. Removing punctuation can help harmless formatting differences but can also merge distinct strings. A clear policy is more useful than a vague claim of semantic correctness. This first stage deliberately uses a narrow deterministic normalizer whose limitations are easy to inspect.','Do not remove articles or Unicode punctuation. Changing normalization creates a different evaluation protocol.','Find a punctuation-sensitive answer pair that this metric would confuse.'),
('Score acceptable references','exact_match','Some questions have several valid surface forms. Evaluate against the set of acceptable references rather than choosing one arbitrarily. The metric remains exact after normalization; it does not infer synonyms. An empty reference list is malformed evaluation data and must fail rather than make every model incorrect.','Return a numeric score suitable for aggregation, not a generated explanation.','When would exact match be suitable for model behavior, and when would it be misleading?'),
('Score a continuation at the right offset','continuation_logp','Likelihood evaluation conditions on the prompt and scores only the continuation. A token’s probability comes from the logit immediately before it. Including prompt tokens rewards the model for reproducing the question rather than answering it. The final logit predicts an unobserved next token and is outside the score.','The full prompt must be provided to the model even though its targets are excluded from the metric.','Draw token positions and corresponding prediction rows for a two-token prompt.'),
('Make length normalization explicit','rank_choices','Summed log probability tends to favor shorter strings, while average log probability defines a different comparison. Neither choice should be hidden inside a helper. Declaring the protocol allows results to be reproduced and compared. Stable tie handling prevents candidate order from producing undocumented random variation.','Lengths count scored continuation tokens, not prompt tokens.','Construct two choices whose ranking flips under normalization.'),
('Aggregate perplexity by token count','perplexity','Perplexity exponentiates average negative log likelihood. Averaging per-document perplexities is not equivalent because exponentiation is nonlinear and documents contain different numbers of tokens. Carrying loss sums and counts through aggregation avoids both mistakes. Comparisons require the same tokenizer and evaluation corpus.','Inputs are NLL sums, not already averaged losses.','Why is perplexity across different token vocabularies difficult to compare?'),
('Count classification outcomes','confusion_matrix','A single accuracy number hides which classes are confused. The confusion matrix records truth on rows and predictions on columns, a convention that must remain consistent downstream. Empty declared classes still belong in the matrix. Fixed class dimensions make comparisons across runs meaningful even when a small sample omits a class.','Reject out-of-range IDs instead of resizing the matrix silently.','Which axis would you sum to compute predicted class frequency?'),
('Distinguish macro and micro behavior','classification_metrics','Class imbalance makes aggregate accuracy easy to misread. Macro F1 gives every declared class equal influence, including absent classes under the explicit zero convention here. Zero denominators need a declared policy. Computing metrics from raw counts makes the weighting decisions auditable.','F1 for class c is 2TP divided by predicted_count+true_count.','Build a classifier that gets high accuracy by ignoring a rare class.'),
('Estimate uncertainty with paired resampling','paired_bootstrap','Two models evaluated on the same examples produce paired observations. Resampling example indices jointly preserves easy and hard cases across the comparison. Independently resampling each model throws away this relationship and changes uncertainty. A bootstrap interval estimates sampling variation under assumptions about these cases; it does not account for all dataset or judge biases.','Use local seeded NumPy randomness and percentile intervals. Larger scores mean better performance.','Why does a constant per-example improvement have a zero-width paired interval?'),
('Align comparisons by identity','compare_by_id','Evaluation outputs often finish in a different order because of batching, retries or distributed workers. Zipping files can compare unrelated examples while producing plausible numbers. Stable IDs form the join key. Missing or duplicate cases should stop the comparison instead of silently reducing it to a convenient subset.','Sort IDs before resampling so reordering files leaves the report unchanged.','Explain how silently dropping failed examples can inflate a benchmark score.'),
('Inspect performance slices','slice_metrics','A model can improve overall while failing a small important group. Micro averaging follows the dataset’s group proportions; macro averaging weights groups equally. Reporting the worst slice helps expose concentrated failures, but small slice counts also mean higher uncertainty. Always keep counts beside slice metrics.','Each record has exactly one group in this teaching interface.','How would overlapping slices change the interpretation of a macro score?'),
('Measure calibration','calibration_error','Confidence and correctness are different quantities. Calibration asks whether outcomes occur at the rate the confidence predicts. Binned ECE is an approximation whose value depends on binning choices. Internal boundaries, confidence one and empty bins need explicit conventions. A low ECE does not by itself imply a model is accurate or useful.','Use confidence in the predicted answer, not arbitrary raw logits.','Can a constant-confidence model be calibrated but uninformative?'),
('Measure user-visible generation latency','latency_metrics','Time to first token includes the delay before any answer appears. Inter-token latency describes the continuation experience. Counting the first token in decode throughput mixes prefill with decode work. Timestamp traces preserve enough detail to compute different summaries later and to detect non-monotonic measurement errors.','The timestamps are supplied measurements; this function must not fabricate timing data.','Compare two systems with equal total latency but different first-token latency.'),
('Run an auditable evaluation','evaluate','The harness should control references and expose only prompts to the predictor. Carrying model and dataset identities into the result makes claims traceable. Every case must produce an output or an explicit failure: dropping failures changes the denominator and often rewards unreliable systems. This harness keeps the execution path intentionally small and inspectable.','Exceptions propagate. Case outputs preserve input order, even though later comparisons align by ID.','What metadata would you add for stochastic generation or a model-based judge?'),
('Publish a finite immutable report','write_report','A machine-readable result is useful only if downstream consumers interpret it consistently. JSON NaN and infinity are nonstandard and can hide failed calculations. Canonical serialization and a file digest identify the exact evidence used by a release gate. Refusing overwrite preserves earlier reports for comparison.','This writes a local artifact only; it does not publish a model or call an external service.','Trace how a report digest connects an evaluation to a model release.')]
stages=[(*r,['Identify exactly what is being counted or conditioned on.','Check a tiny case where averaging or positional alignment changes the result.','Keep identity, normalization and uncertainty policies explicit.']) for r in rows]
build('05-evaluation','Build a model evaluation harness','Learn to make and audit model-quality claims through fourteen small stages. Implement likelihood scoring, classification metrics, paired uncertainty, slices, calibration, latency summaries and artifact-backed reports. Statistical evidence remains distinct from passing software tests.',stages,source,tests,[('CS336 evaluation curriculum','https://cs336.stanford.edu/'),('ARENA evaluations','https://www.arena.education/curriculum')])
