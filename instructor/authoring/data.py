from build import build
source='''"""Instructor reference for deterministic dataset preparation."""
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata


def normalize(text):
    """NFKC normalize, casefold, collapse all whitespace to single spaces, strip ends.
    Use for matching; preserve original text separately for training and provenance.
    """
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())

def document_id(text):
    """SHA256 lowercase hex of normalize(text).encode('utf-8')."""
    return hashlib.sha256(normalize(text).encode('utf-8')).hexdigest()

def quality_filter(text, min_words=3, max_repeat=.5):
    """Return bool: normalized whitespace words count >= min_words and the most frequent
    word's fraction <= max_repeat. Empty text always fails. Boundary thresholds are inclusive.
    This toy heuristic is not a language detector or a universal measure of quality.
    """
    words=normalize(text).split()
    return bool(words) and len(words)>=min_words and max(words.count(w) for w in set(words))/len(words)<=max_repeat

def deduplicate(records):
    """Records are dicts with id,text. Return shallow copies of the first occurrence for
    each normalized content ID, preserving input order. Never mutate caller records.
    """
    seen=set(); result=[]
    for r in records:
        digest=document_id(r['text'])
        if digest not in seen:
            result.append(dict(r)); seen.add(digest)
    return result

def shingles(text, n=3):
    """Set of consecutive n-word tuples from normalized text. n>=1, else ValueError.
    Documents with fewer than n words produce an empty set.
    """
    if n<1: raise ValueError('Positive shingle size required')
    words=normalize(text).split()
    return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}

def duplicate_components(records, threshold=.5, n=3):
    """Return list of ID lists, components in first-record order, members in input order.
    Add an edge when nonempty shingle-set Jaccard >= threshold, or normalized text is equal.
    Connected components include transitive matches; distinct short documents do not match.
    IDs must be unique; threshold in (0,1]. O(N^2) is intentional for this teaching dataset.
    """
    if not 0<threshold<=1 or len({r['id'] for r in records})!=len(records):
        raise ValueError('Invalid threshold or duplicate IDs')
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:
            i=parent[i]
        return i
    sets=[shingles(r['text'],n) for r in records]
    for i in range(len(records)):
        for j in range(i):
            union=sets[i]|sets[j]
            same=normalize(records[i]['text'])==normalize(records[j]['text'])
            if same or (union and len(sets[i]&sets[j])/len(union)>=threshold):
                parent[find(i)]=find(j)
    groups={}
    for i,r in enumerate(records): groups.setdefault(find(i),[]).append(r['id'])
    return list(groups.values())

def split_groups(groups, validation_fraction=.2, salt='course'):
    """Return ID->'train'/'validation'. Hash salt + ':' + lexicographically smallest group ID;
    SHA256 interpreted as big-endian int divided by 2**256; values below fraction go to validation.
    Validate fraction in [0,1], nonempty groups and no ID repeated across or within groups.
    Group and member input ordering must not affect assignment. This does not enforce exact counts.
    """
    flat=[x for g in groups for x in g]
    if not 0<=validation_fraction<=1 or any(not g for g in groups) or len(flat)!=len(set(flat)):
        raise ValueError('Invalid groups or fraction')
    out={}
    for group in groups:
        value=int(hashlib.sha256((salt+':'+min(group)).encode()).hexdigest(),16)/2**256
        split='validation' if value<validation_fraction else 'train'
        out.update({x:split for x in group})
    return out

def decontaminate(records, holdout_texts, n=3):
    """Return copied records with no shared n-word shingle with any holdout text.
    Also reject exact normalized matches, including short texts. Preserve order.
    This is a strict toy filter; it can discard common phrases and is not a production detector.
    """
    reserved=set().union(*(shingles(t,n) for t in holdout_texts))
    exact={normalize(t) for t in holdout_texts}
    return [dict(r) for r in records if normalize(r['text']) not in exact and not (shingles(r['text'],n)&reserved)]

def sample_mixture(sources, weights, count, seed=0):
    """sources maps names to nonempty lists of records. weights has identical keys, finite
    nonnegative values and positive total. Draw WITH replacement: choose sorted source name
    via local random.Random(seed).choices, then a record via that RNG.choice; return copied
    dicts with added source field. count>=0. Inputs remain unchanged; no global RNG use.
    """
    import math
    if count<0 or set(sources)!=set(weights) or not sources or any(not v for v in sources.values()) or any(not math.isfinite(w) or w<0 for w in weights.values()) or sum(weights.values())<=0:
        raise ValueError('Invalid mixture')
    rng=random.Random(seed); names=sorted(sources); out=[]
    for _ in range(count):
        name=rng.choices(names,weights=[weights[n] for n in names],k=1)[0]
        out.append(dict(rng.choice(sources[name]),source=name))
    return out

def pack_documents(documents, block_size, pad_id=0):
    """Greedily pack whole nonempty token-ID lists in input order, never split a document.
    Reject block_size<1 or any document longer than it. Return list of dicts: input_ids,
    segment_ids (original document index, padding=-1), position_ids (0..len(doc)-1,
    padding=0), loss_mask (bool at TARGET position: true except each document first token
    and padding). Pad every block to block_size. Empty documents are ignored, but indexes
    still refer to the original input. Empty input returns [].
    """
    if block_size<1 or any(len(d)>block_size for d in documents): raise ValueError('Invalid block size')
    blocks=[]; current=None
    def flush():
        if current is not None:
            size=block_size-len(current['input_ids'])
            for key,value in [('input_ids',pad_id),('segment_ids',-1),('position_ids',0),('loss_mask',False)]:
                current[key].extend([value]*size)
            blocks.append(current)
    for i,doc in enumerate(documents):
        if not doc: continue
        if current is None or len(current['input_ids'])+len(doc)>block_size:
            flush(); current={k:[] for k in ['input_ids','segment_ids','position_ids','loss_mask']}
        current['input_ids'].extend(doc); current['segment_ids'].extend([i]*len(doc))
        current['position_ids'].extend(range(len(doc))); current['loss_mask'].extend([False]+[True]*(len(doc)-1))
    flush()
    return blocks

def attention_mask(segment_ids):
    """Return list[list[bool]] allowed[q][k]: k<=q, equal segment IDs and segment >=0.
    Padding rows are entirely false; callers must avoid softmax over these rows.
    """
    return [[k<=q and s>=0 and s==t for k,t in enumerate(segment_ids)] for q,s in enumerate(segment_ids)]

def write_dataset(directory, records):
    """Create directory; write records.jsonl as UTF-8 JSON, sort_keys=True, ensure_ascii=False,
    separators=(',',':'), one newline per record. Write manifest.json containing format
    'toydata-v1', file 'records.jsonl', sha256 of exact bytes, records count. Return manifest.
    Empty datasets have zero bytes. Never overwrite an existing records.jsonl or manifest.json.
    """
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    path=directory/'records.jsonl'; manifest_path=directory/'manifest.json'
    if path.exists() or manifest_path.exists(): raise FileExistsError('Dataset already exists')
    body=''.join(json.dumps(r,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\\n' for r in records).encode('utf-8')
    path.write_bytes(body)
    manifest=dict(format='toydata-v1',file='records.jsonl',sha256=hashlib.sha256(body).hexdigest(),records=len(records))
    manifest_path.write_text(json.dumps(manifest,sort_keys=True)+'\\n')
    return manifest

def prepare_dataset(records, holdout_texts, directory, salt='course'):
    """Integrated preparation: quality_filter defaults, deduplicate, decontaminate defaults,
    duplicate_components defaults, split_groups(.2,salt), then write_dataset. Output copied
    records gain split and content_sha256 fields. Validate unique input IDs before filtering.
    Return manifest. Original text and IDs survive; holdout texts never enter the output.
    """
    if len({r['id'] for r in records})!=len(records): raise ValueError('Duplicate input IDs')
    clean=deduplicate([r for r in records if quality_filter(r['text'])])
    clean=decontaminate(clean,holdout_texts)
    assignments=split_groups(duplicate_components(clean),salt=salt)
    result=[dict(r,split=assignments[r['id']],content_sha256=document_id(r['text'])) for r in clean]
    return write_dataset(directory,result)
'''
tests='''import hashlib
import json
import random
import pytest


def test_01(api):
    assert api.normalize('  ＨＥＬＬＯ\\tStraße\\n')=='hello strasse'
    assert api.normalize('e\\u0301')==api.normalize('é')
    assert api.normalize(' \\n ')==''


def test_02(api):
    assert api.document_id(' Hello ') == hashlib.sha256(b'hello').hexdigest()
    assert api.document_id('Ａ')==api.document_id('a')
    assert api.document_id('ab')!=api.document_id('a b')


def test_03(api):
    assert api.quality_filter('one two three')
    assert not api.quality_filter('a a a b') and not api.quality_filter('')
    assert api.quality_filter('a a b c') and not api.quality_filter('one two')


def test_04(api):
    records=[{'id':'a','text':'Hello world'},{'id':'b','text':' HELLO  world '},{'id':'c','text':'Different'}]
    result=api.deduplicate(records); assert [r['id'] for r in result]==['a','c']
    result[0]['text']='changed'; assert records[0]['text']=='Hello world'


def test_05(api):
    assert api.shingles('A b c d',2)=={('a','b'),('b','c'),('c','d')}
    assert api.shingles('a',2)==set()
    with pytest.raises(ValueError): api.shingles('a',0)


def test_06(api):
    records=[dict(id='a',text='a b c'),dict(id='b',text='b c d'),dict(id='c',text='c d e'),dict(id='d',text='short'),dict(id='e',text='other')]
    assert api.duplicate_components(records,threshold=1/3,n=2)==[['a','b','c'],['d'],['e']]
    with pytest.raises(ValueError): api.duplicate_components([records[0],records[0]])
    assert api.duplicate_components([dict(id='x',text='short'),dict(id='y',text='SHORT')])==[['x','y']]


def test_07(api):
    groups=[['a','b'],['c'],['d','e']]; a=api.split_groups(groups)
    assert a==api.split_groups([['e','d'],['c'],['b','a']]) and a['a']==a['b']
    assert set(api.split_groups(groups,1).values())=={'validation'}
    assert set(api.split_groups(groups,0).values())=={'train'}
    for g in groups:
        value=int(hashlib.sha256(('course:'+min(g)).encode()).hexdigest(),16)/2**256
        assert a[g[0]]==('validation' if value<.2 else 'train')
    with pytest.raises(ValueError): api.split_groups([['a'],['a']])


def test_08(api):
    r=[dict(id='a',text='the hidden test answer'),dict(id='b',text='fresh original training sample'),dict(id='c',text='short')]
    assert [x['id'] for x in api.decontaminate(r,['hidden test answer','SHORT'])]==['b']
    assert api.decontaminate(r,[])==r


def test_09(api,seed):
    src={'a':[dict(id=1)],'b':[dict(id=2),dict(id=3)]}; weights={'a':.2,'b':.8}
    before=random.getstate(); a=api.sample_mixture(src,weights,100,seed)
    assert random.getstate()==before and a==api.sample_mixture(src,weights,100,seed)
    assert a==api.sample_mixture(dict(reversed(list(src.items()))),weights,100,seed)
    assert len(a)==100 and {r['source'] for r in a}=={'a','b'} and 'source' not in src['a'][0]
    assert all(x['source']=='b' for x in api.sample_mixture(src,{'a':0,'b':1},10,seed))
    with pytest.raises(ValueError): api.sample_mixture(src,{'a':0,'b':0},2)


def test_10(api):
    docs=[[256,1,257],[],[256,2,3,257],[256,4,257]]; saved=[d[:] for d in docs]
    blocks=api.pack_documents(docs,8)
    assert docs==saved and len(blocks)==2
    assert blocks[0]==dict(input_ids=[256,1,257,256,2,3,257,0],segment_ids=[0,0,0,2,2,2,2,-1],position_ids=[0,1,2,0,1,2,3,0],loss_mask=[False,True,True,False,True,True,True,False])
    assert api.pack_documents([],4)==[]
    with pytest.raises(ValueError): api.pack_documents([[1,2,3]],2)


def test_11(api):
    assert api.attention_mask([0,0,1,-1])==[[True,False,False,False],[True,True,False,False],[False,False,True,False],[False,False,False,False]]
    assert api.attention_mask([])==[]


def test_12(api,tmp_path):
    records=[{'id':'x','text':'é'}]; m=api.write_dataset(tmp_path,records)
    body=(tmp_path/'records.jsonl').read_bytes(); assert body=='{"id":"x","text":"é"}\\n'.encode()
    assert m==dict(format='toydata-v1',file='records.jsonl',sha256=hashlib.sha256(body).hexdigest(),records=1)
    assert json.loads((tmp_path/'manifest.json').read_text())==m
    with pytest.raises(FileExistsError): api.write_dataset(tmp_path,[])
    assert (tmp_path/'records.jsonl').read_bytes()==body


def test_13(api,tmp_path):
    records=[dict(id='a',text='A fresh original document'),dict(id='b',text='a fresh ORIGINAL document'),dict(id='c',text='secret held out answer'),dict(id='d',text='x x x'),dict(id='e',text='a fresh original document extended')]
    m=api.prepare_dataset(records,['secret held out answer'],tmp_path)
    out=[json.loads(x) for x in (tmp_path/'records.jsonl').read_text().splitlines()]
    assert m['records']==2 and [r['id'] for r in out]==['a','e']
    assert out[0]['split']==out[1]['split'] and all(len(r['content_sha256'])==64 for r in out)
    assert 'split' not in records[0]
    with pytest.raises(ValueError): api.prepare_dataset([records[0],records[0]],[],tmp_path/'bad')
'''
rows=[
('Separate matching text from training text','normalize','Text equality is a policy decision. Compatibility normalization merges some visually or semantically related spellings, while casefolding handles more than ASCII lowercase. Whitespace normalization makes layout differences irrelevant for matching. These operations can remove useful distinctions, so preserve original content and apply this representation only to matching.','Normalization must be deterministic and idempotent. Never silently replace original training text.','Find two strings that this normalization merges but that might matter in a code corpus.'),
('Give content a stable identity','document_id','A dataset row ID describes a record; a content hash describes a chosen representation of its content. Stable hashes let independent runs agree about duplicates. Python’s built-in hash is process-dependent and inappropriate here. Hashing normalized text makes the matching policy an implicit part of content identity, so record that policy in a real dataset manifest.','Use cryptographic SHA256 rather than Python hash.','Should changing the normalization policy produce a new dataset version?'),
('Make filtering explainable','quality_filter','A cheap heuristic can eliminate obvious repetition, but filtering changes the data distribution. A threshold that helps one language or document type can harm another. This toy filter exposes two measurable features: word count and maximum repetition fraction. Keeping the rule small makes false positives easy to examine before adopting more complex scoring models.','Use normalized whitespace-separated words; punctuation is not separately stripped.','Construct a legitimate document this filter rejects and describe the resulting bias.'),
('Remove exact duplicates stably','deduplicate','Duplicate examples give repeated content extra weight. Removing duplicates before splitting reduces direct leakage and makes dataset size more meaningful. Stable first-occurrence retention provides a clear provenance rule and avoids nondeterminism from set iteration. Returning independent record dictionaries prevents downstream annotations from modifying the raw input collection.','Copy dictionaries shallowly; records contain scalar fields in the acceptance cases.','When would you prefer retaining the newest rather than the first copy?'),
('Represent local overlap','shingles','Word shingles capture consecutive local context. A bag of words can call reordered text identical; shingles preserve a limited amount of order. Larger windows are more specific but miss shorter overlaps. Sets ignore repeated occurrence counts, which is intentional for Jaccard similarity but means repetition is handled elsewhere.','Short documents yield empty sets rather than a special match token.','Compare shingle sizes one, two and three on a lightly edited sentence.'),
('Group transitive near duplicates','duplicate_components','Pairwise similarity is not transitive, but leakage groups need a transitive closure. If A resembles B and B resembles C, assigning A and C independently can leak related content through B. Connected components solve this grouping problem. This exact quadratic implementation prioritizes correctness; approximate candidate generation would be needed at web scale.','Use Jaccard intersection/union for nonempty unions. Equality of short normalized texts still creates an edge.','Construct a chain where A and C do not directly meet the threshold.'),
('Split groups, not rows','split_groups','A random row split can scatter related documents across training and validation. Group-level assignment keeps an entire connected component together. Hash-based assignment is reproducible across input ordering and allows streaming lookup once group identity is known. It gives an expected fraction rather than an exact count, particularly with a small number of uneven groups.','Group identity is its lexicographically smallest member ID. A changing group can change its assignment.','What happens to the validation fraction when one group contains half the corpus?'),
('Decontaminate against held-out content','decontaminate','A model can appear capable because evaluation examples or close variants appeared in training. Decontamination checks training candidates against reserved content before fitting. This stage uses a strict overlap policy to make the mechanics explicit. It will over-remove common phrases, so an actual data pipeline needs an examined tradeoff and a versioned evaluation set.','Holdout content is a filter input only and must never be appended to training.','Why is removing contaminated examples after model training too late?'),
('Sample a reproducible mixture','sample_mixture','A training mixture changes how frequently each source contributes, regardless of raw dataset size. Sampling source then example makes that policy explicit. Replacement allows small sources to recur, which can overfit them. Stable source ordering and a local RNG ensure a harmless dictionary reordering does not change the sequence of training examples.','Weights are relative masses, not required to sum to one. Zero-weight sources never appear.','Estimate how often a one-record source repeats in a 1000-sample run.'),
('Pack without erasing boundaries','pack_documents','Packing saves padding but introduces a semantic risk: independent documents can attend to one another or create cross-document targets. Token IDs alone cannot express which tokens belong together. Segment IDs, position IDs and a target-position loss mask preserve those boundaries. This stage keeps documents whole; truncation or chunking must be an explicit upstream policy.','First tokens have loss_mask=false because no earlier token in that document predicts them. EOS is supervised.','Explain why masking the BOS target does not remove the need for an attention mask.'),
('Build isolated causal visibility','attention_mask','An attention mask is a statement about allowed information flow. Causality permits keys no later than the query; isolation additionally requires equal valid segment IDs. Padding queries have no valid keys, making naive all-negative-infinity softmax undefined. A model consuming this representation should avoid evaluating padding rows or explicitly handle them.','Return Python boolean rows, not a floating additive mask. True means allowed.','Draw a block diagonal causal mask for two packed documents.'),
('Write an immutable dataset artifact','write_dataset','Reproducibility begins with identifying the exact bytes used for training. Canonical JSON serialization makes a small dataset inspectable and repeatable. The manifest records a content digest and count; it is metadata, not proof that the content is good. Refusing to overwrite avoids silently invalidating an earlier experiment’s dataset reference.','This educational writer is single-process and local. Concurrent writers and crash-atomic directory publication are later release concerns.','Why must the trailing newline convention be part of byte identity?'),
('Integrate the preparation pipeline','prepare_dataset','Pipeline ordering matters. Exact deduplication precedes near-duplicate grouping, and decontamination removes reserved content before split assignment. Retaining original text and adding content IDs preserves both usability and traceability. The final artifact is a real input to the training course, so this integration gate checks the combined behavior rather than isolated helper outputs.','Validate raw record IDs before filtering so a discarded record cannot hide a duplicate identifier.','Change one preprocessing rule and describe which experiment metadata must change.')]
stages=[(*r,['Identify the invariant that must survive reordering or copying.','Use a two-document counterexample to an obvious implementation.','Separate content identity, record identity and output order.']) for r in rows]
build('03-data','Build a training-data pipeline','Construct reproducible, leakage-aware datasets in thirteen stages. Work on tiny local records so you can inspect every filtering and splitting decision. No web crawl or external dataset download is required.',stages,source,tests,[('CS336 data curriculum','https://cs336.stanford.edu/'),('Python Unicode normalization','https://docs.python.org/3/library/unicodedata.html')])
