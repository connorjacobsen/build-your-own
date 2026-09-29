from build import build
source='''"""Instructor reference: a local model release registry, not a hosted deployment platform."""
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
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\\n').encode()

def file_digest(path):
    """SHA256 hex of file bytes, read in chunks no larger than 1 MiB. Empty files allowed."""
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024): h.update(chunk)
    return h.hexdigest()

def safe_artifact_path(root, relative):
    """Return resolved existing regular file inside resolved root. Reject absolute paths,
    empty paths, any '..' component and symlink resolution outside root, with ValueError.
    Internal symlinks to regular files are permitted. Missing/nonfiles raise ValueError.
    """
    root=Path(root).resolve(); path=Path(relative)
    if not relative or path.is_absolute() or '..' in path.parts: raise ValueError('Invalid relative path')
    resolved=(root/path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file(): raise ValueError('Artifact outside bundle or absent')
    return resolved

def build_manifest(root, paths, metadata):
    """Require nonempty unique relative paths. Return format='toyrelease-v1', metadata copied
    via canonical JSON roundtrip, files sorted by path with {path,sha256,size_bytes}.
    Validate each path using safe_artifact_path. Caller selects explicit files, no recursive scan.
    """
    if not paths or len(set(paths))!=len(paths): raise ValueError('Explicit unique files required')
    files=[]
    for name in sorted(paths):
        p=safe_artifact_path(root,name); files.append(dict(path=name,sha256=file_digest(p),size_bytes=p.stat().st_size))
    return dict(format='toyrelease-v1',metadata=json.loads(canonical_json(metadata)),files=files)

def verify_manifest(root, manifest):
    """Validate format, nonempty unique file entries, path containment, exact byte sizes and
    hashes. Return True; any mismatch raises ValueError. Unlisted files do not belong to bundle.
    """
    if manifest.get('format')!='toyrelease-v1' or not manifest.get('files'): raise ValueError('Invalid manifest')
    names=[r['path'] for r in manifest['files']]
    if len(names)!=len(set(names)): raise ValueError('Duplicate file')
    for row in manifest['files']:
        path=safe_artifact_path(root,row['path'])
        if path.stat().st_size!=row['size_bytes'] or file_digest(path)!=row['sha256']: raise ValueError('Artifact integrity mismatch')
    return True

def register_artifact(store, source):
    """Copy source to content-addressed store/<sha256>. Read/write by chunks; atomically
    publish a temporary file with os.replace. Return digest. If object exists, verify it
    matches digest and return without altering it. Corrupt existing objects raise ValueError.
    The source remains unchanged. Clean temporary files on success or failure.
    """
    store=Path(store); store.mkdir(parents=True,exist_ok=True); digest=file_digest(source); target=store/digest
    if target.exists():
        if file_digest(target)!=digest: raise ValueError('Corrupt object store')
        return digest
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=store,delete=False) as out:
            temporary=Path(out.name)
            with Path(source).open('rb') as inp:
                while chunk:=inp.read(1024*1024): out.write(chunk)
            out.flush(); os.fsync(out.fileno())
        if file_digest(temporary)!=digest: raise ValueError('Source changed during registration')
        os.replace(temporary,target)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)
    return digest

def validate_evaluation(report, model_sha256, dataset_sha256):
    """Validate toyeval-v1 identities, unique nonempty cases, finite [0,1] scores, metrics.n
    equal case count, metrics.accuracy equal score mean (abs tolerance 1e-12). Both digests
    must be 64 lowercase hex chars and match expected identities. Return True or ValueError.
    This validates internal consistency, not whether an external evaluator was honest.
    """
    try:
        if report['format']!='toyeval-v1' or report['model_sha256']!=model_sha256 or report['dataset_sha256']!=dataset_sha256: raise ValueError('Wrong evaluation identity')
        if any(len(d)!=64 or any(c not in '0123456789abcdef' for c in d) for d in [model_sha256,dataset_sha256]): raise ValueError('Invalid digest')
        rows=report['cases']; scores=[r['score'] for r in rows]
        if not rows or len({r['id'] for r in rows})!=len(rows) or any(not math.isfinite(s) or not 0<=s<=1 for s in scores): raise ValueError('Invalid cases')
        metrics=report['metrics']
        if metrics['n']!=len(rows) or not math.isfinite(metrics['accuracy']) or abs(metrics['accuracy']-sum(scores)/len(scores))>1e-12: raise ValueError('Inconsistent aggregate')
    except (KeyError,TypeError) as exc: raise ValueError('Malformed report') from exc
    return True

def release_gate(baseline, candidate, min_accuracy=.8, max_drop=.02, min_cases=20):
    """Validate both reports and require same dataset and case-ID sets. Reject incomparable
    evidence with ValueError. Return {eligible:bool,reasons:list[str]} in fixed order:
    'too_few_cases', 'below_floor', 'regression'. A boundary exactly meeting a threshold passes.
    Require thresholds in [0,1], min_cases>=1. This deterministic gate is policy, not a significance test.
    """
    if not 0<=min_accuracy<=1 or not 0<=max_drop<=1 or min_cases<1: raise ValueError('Invalid policy')
    validate_evaluation(baseline,baseline['model_sha256'],baseline['dataset_sha256'])
    validate_evaluation(candidate,candidate['model_sha256'],baseline['dataset_sha256'])
    if {r['id'] for r in baseline['cases']}!={r['id'] for r in candidate['cases']}: raise ValueError('Different evaluation cases')
    reasons=[]; b=baseline['metrics']; c=candidate['metrics']
    if c['n']<min_cases: reasons.append('too_few_cases')
    if c['accuracy']<min_accuracy: reasons.append('below_floor')
    if b['accuracy']-c['accuracy']>max_drop+1e-12: reasons.append('regression')
    return dict(eligible=not reasons,reasons=reasons)

def route_request(request_id, candidate_fraction, salt='course'):
    """Deterministic SHA256(salt+':'+request_id) / 2**256 bucket. Return 'candidate' iff
    bucket<fraction else 'baseline'. fraction in [0,1], including exact endpoints.
    This routes one local simulation request; it does not contact a service.
    """
    if not 0<=candidate_fraction<=1: raise ValueError('Invalid fraction')
    bucket=int(hashlib.sha256((salt+':'+request_id).encode()).hexdigest(),16)/2**256
    return 'candidate' if bucket<candidate_fraction else 'baseline'

def summarize_requests(records):
    """Nonempty records {status:'ok'|'error',latency_seconds:finite nonnegative}. Return n,
    error_rate, p95_seconds using nearest-rank ceil(.95*n)-1 on ALL requests including errors.
    Reject malformed status or latency. Units remain seconds.
    """
    if not records or any(r['status'] not in {'ok','error'} or not math.isfinite(r['latency_seconds']) or r['latency_seconds']<0 for r in records): raise ValueError('Invalid observations')
    n=len(records); times=sorted(r['latency_seconds'] for r in records)
    return dict(n=n,error_rate=sum(r['status']=='error' for r in records)/n,p95_seconds=times[math.ceil(.95*n)-1])

def rollback_decision(observed, max_error=.05, max_p95=2., min_requests=20):
    """Given summarize_requests output, return 'wait' below min_requests; else 'rollback'
    if error_rate>max_error OR p95_seconds>max_p95; otherwise 'keep'. Threshold equality passes.
    Validate finite metrics and policy, n>=0, rates in [0,1], latencies>=0,min_requests>=1.
    """
    n=observed['n']; e=observed['error_rate']; p=observed['p95_seconds']
    if min_requests<1 or n<0 or not all(math.isfinite(x) for x in [e,p,max_error,max_p95]) or not 0<=e<=1 or p<0 or not 0<=max_error<=1 or max_p95<0: raise ValueError('Invalid monitoring policy')
    if n<min_requests: return 'wait'
    return 'rollback' if e>max_error or p>max_p95 else 'keep'

def promote(pointer, new_digest, expected_current):
    """Local POSIX compare-and-swap pointer. JSON {current,previous}; absent means current=None.
    Acquire exclusive flock on sibling <name>.lock, reread pointer, require current==expected
    else RuntimeError, then atomically replace pointer with current=new_digest,previous=old.
    New digest is 64 lowercase hex chars. Same-current promotion is idempotent and preserves
    previous. Return resulting dict. Create parent directories. No remote deployment occurs.
    """
    if len(new_digest)!=64 or any(c not in '0123456789abcdef' for c in new_digest): raise ValueError('Invalid digest')
    path=Path(pointer); path.parent.mkdir(parents=True,exist_ok=True)
    with (path.parent/(path.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        old=json.loads(path.read_text()) if path.exists() else dict(current=None,previous=None)
        if old['current']!=expected_current: raise RuntimeError('Concurrent promotion conflict')
        if old['current']==new_digest: return old
        record=dict(current=new_digest,previous=old['current']); temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
                temporary=Path(stream.name); stream.write(canonical_json(record)); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary,path)
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)
        return record

def rollback(pointer, expected_current):
    """Read previous digest and call promote(pointer,previous,expected_current). Reject no
    prior release with ValueError. The compare-and-swap rejects a concurrent pointer change.
    A successful rollback records the rolled-back release as previous, enabling explicit undo.
    """
    record=json.loads(Path(pointer).read_text())
    if record['previous'] is None: raise ValueError('No prior release')
    return promote(pointer,record['previous'],expected_current)

def release_model(model_path, report_path, baseline, store, pointer, expected_current, min_accuracy=.8, max_drop=.02, min_cases=20):
    """Integrated local release: compute model file digest, parse candidate report, validate
    its model/dataset identity against actual bytes and baseline, apply release_gate.
    If ineligible raise ValueError BEFORE storing artifacts or changing pointer. Otherwise
    register model and report, promote with expected_current, return {model_sha256,
    report_sha256,pointer}. Report integrity and model quality remain separate claims.
    """
    model_digest=file_digest(model_path); report=json.loads(Path(report_path).read_text())
    validate_evaluation(report,model_digest,baseline['dataset_sha256'])
    decision=release_gate(baseline,report,min_accuracy,max_drop,min_cases)
    if not decision['eligible']: raise ValueError(','.join(decision['reasons']))
    model_digest=register_artifact(store,model_path); report_digest=register_artifact(store,report_path)
    result=promote(pointer,model_digest,expected_current)
    return dict(model_sha256=model_digest,report_sha256=report_digest,pointer=result)
'''
tests='''from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import pytest


def report(scores,model='a'*64,data='d'*64):
    return dict(format='toyeval-v1',model_sha256=model,dataset_sha256=data,metrics=dict(n=len(scores),accuracy=sum(scores)/len(scores)),cases=[dict(id=str(i),prediction='x',score=s) for i,s in enumerate(scores)])


def test_01(api):
    assert api.canonical_json({'z':2,'a':'é'})=='{"a":"é","z":2}\\n'.encode()
    with pytest.raises(ValueError): api.canonical_json({'bad':float('nan')})


def test_02(api,tmp_path):
    p=tmp_path/'blob'; body=b'abc'*800000; p.write_bytes(body)
    assert api.file_digest(p)==hashlib.sha256(body).hexdigest()
    p.write_bytes(b''); assert api.file_digest(p)==hashlib.sha256(b'').hexdigest()


def test_03(api,tmp_path):
    root=tmp_path/'bundle'; root.mkdir(); (root/'a').write_text('ok'); outside=tmp_path/'secret'; outside.write_text('x'); (root/'link').symlink_to(outside)
    assert api.safe_artifact_path(root,'a')==(root/'a').resolve()
    for bad in ['../secret',str(outside),'link','missing','']:
        with pytest.raises(ValueError): api.safe_artifact_path(root,bad)


def test_04(api,tmp_path):
    (tmp_path/'b').write_bytes(b'b'); (tmp_path/'a').write_bytes(b'a'); metadata={'seed':1}
    m=api.build_manifest(tmp_path,['b','a'],metadata)
    assert m['format']=='toyrelease-v1' and [r['path'] for r in m['files']]==['a','b']
    assert m['files'][0]==dict(path='a',sha256=hashlib.sha256(b'a').hexdigest(),size_bytes=1)
    m['metadata']['seed']=9; assert metadata['seed']==1
    with pytest.raises(ValueError): api.build_manifest(tmp_path,['a','a'],{})


def test_05(api,tmp_path):
    p=tmp_path/'a'; p.write_text('first'); m=api.build_manifest(tmp_path,['a'],{})
    assert api.verify_manifest(tmp_path,m) is True
    p.write_text('other')
    with pytest.raises(ValueError): api.verify_manifest(tmp_path,m)


def test_06(api,tmp_path):
    src=tmp_path/'model'; src.write_bytes(b'weights'); store=tmp_path/'store'; digest=api.register_artifact(store,src)
    assert (store/digest).read_bytes()==b'weights' and src.read_bytes()==b'weights'
    assert api.register_artifact(store,src)==digest and len(list(store.iterdir()))==1
    (store/digest).write_bytes(b'corrupt')
    with pytest.raises(ValueError): api.register_artifact(store,src)


def test_07(api):
    r=report([1,0]); assert api.validate_evaluation(r,'a'*64,'d'*64) is True
    r['metrics']['accuracy']=1
    with pytest.raises(ValueError): api.validate_evaluation(r,'a'*64,'d'*64)
    with pytest.raises(ValueError): api.validate_evaluation(report([1]),'b'*64,'d'*64)
    r=report([1,0]); r['cases'][1]['id']='0'
    with pytest.raises(ValueError): api.validate_evaluation(r,'a'*64,'d'*64)


def test_08(api):
    a=report([1,1,1,1,0]); b=report([1,1,1,0,0],model='b'*64)
    assert api.release_gate(a,b,min_accuracy=.7,max_drop=.1,min_cases=10)==dict(eligible=False,reasons=['too_few_cases','below_floor','regression'])
    assert api.release_gate(a,a,min_accuracy=.8,max_drop=0,min_cases=5)['eligible']
    with pytest.raises(ValueError): api.release_gate(a,report([1],data='c'*64))


def test_09(api):
    assert api.route_request('id',0)=='baseline' and api.route_request('id',1)=='candidate'
    for i in range(30):
        key=str(i); ratio=int(hashlib.sha256(('course:'+key).encode()).hexdigest(),16)/2**256
        assert api.route_request(key,.3)==('candidate' if ratio<.3 else 'baseline')
    with pytest.raises(ValueError): api.route_request('id',2)


def test_10(api):
    records=[dict(status='error' if i==19 else 'ok',latency_seconds=i/10) for i in range(20)]
    assert api.summarize_requests(records)==dict(n=20,error_rate=.05,p95_seconds=1.8)
    with pytest.raises(ValueError): api.summarize_requests([dict(status='ok',latency_seconds=-1)])


def test_11(api):
    assert api.rollback_decision(dict(n=19,error_rate=1,p95_seconds=9))=='wait'
    assert api.rollback_decision(dict(n=20,error_rate=.05,p95_seconds=2))=='keep'
    assert api.rollback_decision(dict(n=20,error_rate=.1,p95_seconds=1))=='rollback'
    assert api.rollback_decision(dict(n=20,error_rate=0,p95_seconds=3))=='rollback'


def test_12(api,tmp_path):
    p=tmp_path/'active.json'; assert api.promote(p,'a'*64,None)==dict(current='a'*64,previous=None)
    assert api.promote(p,'a'*64,'a'*64)==dict(current='a'*64,previous=None)
    def attempt(d):
        try: api.promote(p,d,'a'*64); return 'ok'
        except RuntimeError: return 'conflict'
    with ThreadPoolExecutor(2) as pool: results=list(pool.map(attempt,['b'*64,'c'*64]))
    assert sorted(results)==['conflict','ok']
    r=json.loads(p.read_text()); assert r['previous']=='a'*64 and r['current'] in ['b'*64,'c'*64]


def test_13(api,tmp_path):
    p=tmp_path/'active.json'; api.promote(p,'a'*64,None)
    with pytest.raises(ValueError): api.rollback(p,'a'*64)
    api.promote(p,'b'*64,'a'*64); assert api.rollback(p,'b'*64)==dict(current='a'*64,previous='b'*64)
    with pytest.raises(RuntimeError): api.rollback(p,'b'*64)


def test_14(api,tmp_path):
    model=tmp_path/'model'; model.write_bytes(b'weights'); digest=hashlib.sha256(b'weights').hexdigest(); ev=tmp_path/'eval.json'; baseline=report([1,1,0,0]); pointer=tmp_path/'active.json'; store=tmp_path/'objects'
    ev.write_text(json.dumps(report([0,0,0,0],model=digest)))
    with pytest.raises(ValueError): api.release_model(model,ev,baseline,store,pointer,None,min_accuracy=.5,min_cases=4)
    assert not pointer.exists() and not store.exists()
    ev.write_text(json.dumps(report([1,1,1,1],model=digest)))
    r=api.release_model(model,ev,baseline,store,pointer,None,min_accuracy=.5,min_cases=4)
    assert r['model_sha256']==digest and json.loads(pointer.read_text())['current']==digest
    assert (store/r['report_sha256']).read_bytes()==ev.read_bytes()
    ev.write_text(json.dumps(report([1,1,1,1],model='f'*64)))
    with pytest.raises(ValueError): api.release_model(model,ev,baseline,store,pointer,digest,min_accuracy=.5,min_cases=4)
'''
rows=[
('Make metadata byte-stable','canonical_json','Content identity requires an agreed serialization. Dictionary insertion order and whitespace should not change the digest of logically identical metadata. UTF-8 preserves human-readable text, while rejecting NaN and infinity prevents nonstandard numeric values from entering release evidence. The newline convention is part of the format.','This is a local JSON convention, not a universal canonical-JSON standard for signatures.','Why is signing arbitrary pretty-printed JSON fragile?'),
('Hash large artifacts incrementally','file_digest','Weights can be much larger than available working memory. A streaming hash reads bounded chunks while producing the same digest as a one-shot hash. The digest identifies bytes, not semantics: two equivalent parameter serializations can have different hashes. That distinction is useful when tracing exactly which artifact was evaluated.','Do not read the whole file in one call.','Compare byte identity, parameter equality and prediction equivalence.'),
('Bound artifact paths','safe_artifact_path','A bundle should refer only to its own files. Checking string prefixes is insufficient because paths can contain parent traversal or symlinks. Resolve the path and verify containment in the resolved bundle root. This also prevents a manifest from accidentally including a local file outside the intended model package.','Internal symlinks are allowed, escaping symlinks are not. Existing regular files only.','Why does a path starting with the text of a directory name not establish containment?'),
('Describe an explicit bundle','build_manifest','A model release can include weights, tokenizer metadata, configuration and evaluation evidence. Explicit file selection makes that boundary reviewable. A manifest records size and digest for each file and preserves experiment metadata independently of mutable caller dictionaries. Sorted entries make the result deterministic.','The manifest describes files but does not certify their quality.','Which files are necessary to reproduce inference, and which are useful only for training?'),
('Verify integrity before use','verify_manifest','A successful training run does not guarantee that later files are unchanged. Integrity verification checks the bundle against recorded byte identities before loading or promotion. Checking size alone misses equal-size corruption. Checking only one file misses a swapped tokenizer or configuration. Every listed artifact contributes to the release’s identity.','Unexpected unlisted files are ignored; only listed files form this bundle.','Demonstrate corruption that preserves byte length but changes predictions.'),
('Register immutable content','register_artifact','A content-addressed store names objects by their bytes. Repeated registration becomes idempotent, while corruption at an existing digest is a detectable inconsistency. Temporary writes and atomic replacement prevent readers from seeing a partially copied object. Checking the copied digest also catches source changes during registration.','This is a local educational object store, not a distributed storage service.','What extra guarantees would be needed for remote storage and multiple machines?'),
('Validate evaluation evidence','validate_evaluation','A release decision must refer to the actual model and dataset being considered. Reports also need internal consistency: duplicate cases or an inflated aggregate can make an apparently valid number meaningless. Recomputing the aggregate from per-case scores catches bookkeeping defects. It cannot establish the honesty or relevance of the evaluator itself.','Scores are bounded [0,1] and n equals the number of unique cases.','Why does matching a model hash still not prove benchmark validity?'),
('Encode a release policy','release_gate','Release policy combines absolute quality requirements, regression limits and evidence size. It is different from estimating statistical uncertainty. Returning ordered reasons makes failures actionable and auditable. Candidate and baseline must refer to comparable cases; otherwise a delta could reflect an easier dataset rather than an improved model.','The capstone adds paired uncertainty from course 05; this gate intentionally uses explicit deterministic thresholds.','How would you avoid tuning a candidate repeatedly against the final test set?'),
('Route a stable canary cohort','route_request','A canary exposes a limited cohort to a candidate. Stable hashing keeps the same request identity in the same cohort across retries, making behavior reproducible. The salt defines the experiment assignment. Changing it reshuffles the cohort and should be recorded. Increasing the fraction under the same salt grows the candidate cohort predictably.','This returns a routing label only; no live traffic is sent.','Should routing identity be per request or per user for a stateful application?'),
('Summarize serving observations','summarize_requests','A model can pass offline evaluations and still fail operationally. Error rate and tail latency expose different issues. Including failed requests in latency accounting prevents a fast-failing system from looking artificially healthy. The percentile definition matters on small samples, so this course uses an explicit nearest-rank rule.','Use seconds consistently; do not infer missing observations.','How can averaging latency hide a severe tail problem?'),
('Decide when to roll back','rollback_decision','An automated policy needs both failure thresholds and a minimum evidence rule. Very small samples can be noisy, but waiting can prolong impact. The policy here makes that tradeoff visible and testable. A decision is separated from its execution so a learner can inspect the evidence before changing active state.','Return exactly wait, keep or rollback; threshold equality passes.','What risks arise from always waiting for a minimum request count?'),
('Promote with compare-and-swap','promote','Two evaluators can both decide to promote a candidate based on the same active model. A compare-and-swap detects stale decisions rather than allowing the last writer to win silently. A process lock protects the read-check-write transaction, while atomic replacement prevents partial JSON reads. The previous pointer supports recovery.','Uses local POSIX flock, available on macOS and Linux. Atomic visibility is tested; full power-loss durability is outside this toy contract.','Why is atomic file replacement alone insufficient to prevent lost updates?'),
('Restore the previous release','rollback','Rollback should use the same concurrency controls as promotion. Reading a prior version and then switching to it without checking the current version can undo someone else’s newer decision. Compare-and-swap preserves the expectation about what is being rolled back. Keeping the displaced release as previous makes an explicit undo possible.','A first release has no prior target and must fail clearly.','What happens if a promotion occurs between reading the previous pointer and applying rollback?'),
('Release from verified evidence','release_model','The capstone-sized integration combines artifact identity, evidence validation, quality policy, immutable storage and promotion. A failed quality gate must not activate a model or register it as a successful release. Hashing the actual model file binds evaluation metadata to the bytes being promoted. This local workflow provides a concrete foundation for later deployment integrations.','This updates only a local registry pointer. It never deploys to a public endpoint. Failed concurrency checks may leave harmless immutable objects in the store.','Trace every artifact and decision from training output to a rollback.')]
stages=[(*r,['Separate evidence validation from state mutation.','Identify which failures must leave existing state unchanged.','Use explicit identities and validate the full operation before publishing it.']) for r in rows]
build('08-releases','Build a model release and monitoring system','Fourteen stages turn model artifacts and evaluation reports into a local reproducible release workflow. Build integrity checks, release gates, canary routing, monitoring decisions and concurrency-safe promotion/rollback. All operations stay local; no production service is changed.',stages,source,tests,[('Full Stack Deep Learning','https://fullstackdeeplearning.com/course/2022/'),('Python os.replace','https://docs.python.org/3/library/os.html#os.replace'),('Python file locks','https://docs.python.org/3/library/fcntl.html')])
