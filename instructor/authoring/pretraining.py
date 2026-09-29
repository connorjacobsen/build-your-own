from pathlib import Path
from build import build
# Preserve the exact teaching-model parameter layout across training and serving.
base=Path('/Users/connor/Code/ai/nanovllm/grader/oracle/model.py').read_text().split('    @torch.inference_mode()\n    def forward_batch')[0]
source=base+'''

def encode_bytes(text):
    """UTF-8 bytes with BOS=256 prepended and EOS=257 appended; return list[int]."""
    return [256]+list(text.encode('utf-8'))+[257]

def next_token_batch(tokens, starts, context):
    """From one 1D token stream, return long X,Y tensors [len(starts),context].
    X starts at each specified offset; Y is shifted one position. Reject empty starts,
    context<1, negative starts and windows extending past the available next token.
    """
    if context<1 or not starts or any(s<0 or s+context>=len(tokens) for s in starts):
        raise ValueError('Invalid window')
    x=torch.tensor([tokens[s:s+context] for s in starts],dtype=torch.long)
    y=torch.tensor([tokens[s+1:s+context+1] for s in starts],dtype=torch.long)
    return x,y

def token_loss(logits, targets):
    """Mean next-token cross entropy for logits [...,V] and same-prefix-shape targets.
    Compute stable logsumexp minus selected logits; do not call F.cross_entropy.
    """
    selected=logits.gather(-1,targets.unsqueeze(-1)).squeeze(-1)
    return (torch.logsumexp(logits,dim=-1)-selected).mean()

def adamw_step(parameters, grads, state, lr, betas=(.9,.999), eps=1e-8, weight_decay=.01):
    """In-place AdamW over matching lists of tensors. state is initially {}, then contains
    step (integer), m and v (tensor lists). Bias-correct both moments. Apply decoupled
    decay p *= (1-lr*weight_decay). Do not mutate gradient tensors. No torch optimizer.
    All parameters receive a dense gradient in this educational version.
    """
    if not state:
        state.update(step=0,m=[torch.zeros_like(p) for p in parameters],v=[torch.zeros_like(p) for p in parameters])
    state['step']+=1
    b1,b2=betas
    with torch.no_grad():
        for p,g,m,v in zip(parameters,grads,state['m'],state['v']):
            m.mul_(b1).add_(g,alpha=1-b1)
            v.mul_(b2).addcmul_(g,g,value=1-b2)
            p.mul_(1-lr*weight_decay)
            p.addcdiv_(m/(1-b1**state['step']), (v/(1-b2**state['step'])).sqrt()+eps,value=-lr)

def cosine_lr(step, warmup, total, peak, floor=0.):
    """0<=warmup<total; step>=0. With warmup>0, linear step/warmup * peak before warmup.
    Cosine decay from peak at warmup to floor at total; clamp later steps to floor.
    Reject invalid steps, bounds, negative floor or peak<floor.
    """
    if not 0<=warmup<total or step<0 or floor<0 or peak<floor:
        raise ValueError('Invalid schedule')
    if step<warmup:
        return peak*step/warmup
    ratio=min(1.,(step-warmup)/(total-warmup))
    return floor+.5*(peak-floor)*(1+math.cos(math.pi*ratio))

def clip_grad(parameters, max_norm):
    """Clip the global L2 norm of existing gradients in place; return the preclip norm as float.
    Ignore parameters whose grad is None. Reject max_norm<=0 and nonfinite gradient norm.
    """
    if max_norm<=0:
        raise ValueError('Positive threshold required')
    ps=[p for p in parameters if p.grad is not None]
    norm=math.sqrt(sum(float(p.grad.double().square().sum()) for p in ps))
    if not math.isfinite(norm):
        raise ValueError('Nonfinite gradient')
    if norm>max_norm:
        for p in ps:
            p.grad.mul_(max_norm/norm)
    return norm

def train_step(model, optimizer, sequences):
    """One update on nonempty sequences, each at least two token IDs long.
    Weight all target tokens equally, not all sequences equally. Clear old gradients,
    set training mode, backpropagate the aggregate loss, optimizer.step(), return float loss.
    """
    if not sequences or any(len(s)<2 for s in sequences):
        raise ValueError('Each sequence needs an input and target')
    model.train(); optimizer.zero_grad(set_to_none=True)
    count=sum(len(s)-1 for s in sequences)
    loss=0
    for seq in sequences:
        logits,_=model(seq[:-1])
        targets=torch.tensor(seq[1:],device=model.device)
        loss=loss+token_loss(logits,targets)*(len(seq)-1)/count
    loss.backward(); optimizer.step()
    return float(loss.detach())

def save_checkpoint(path, model, optimizer, step):
    """Save dict model, optimizer, step, rng (CPU torch RNG state) using torch.save.
    This resume format is distinct from the portable inference export. Parent directory exists.
    """
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,rng=torch.get_rng_state()),path)

def load_checkpoint(path, model, optimizer):
    """Restore model, optimizer, CPU RNG from our trusted local checkpoint; return saved step.
    Use torch.load(weights_only=True,map_location='cpu') and strict parameter matching.
    """
    record=torch.load(path,weights_only=True,map_location='cpu')
    model.load_state_dict(record['model'],strict=True)
    optimizer.load_state_dict(record['optimizer'])
    torch.set_rng_state(record['rng'])
    return record['step']

def train_run(sequences, steps=30, seed=0, cfg=None, lr=.01):
    """Initialize TinyLM under fork_rng, AdamW(lr=lr,weight_decay=.01), then train_step
    on all supplied sequences for each update. Return model, list of pre-update losses.
    Preserve the caller's CPU torch RNG state. Default config has dim=16, n_layers=1,
    n_heads=2,n_kv_heads=1,hidden_dim=32; vocab and max length keep TinyConfig defaults.
    """
    cfg=cfg or TinyConfig(dim=16,n_layers=1,n_heads=2,n_kv_heads=1,hidden_dim=32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=TinyLM(cfg)
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
        losses=[train_step(model,optimizer,sequences) for _ in range(steps)]
    return model,losses

def export_model(path, model, provenance):
    """Write portable torch payload: format='toylm-v1', config=vars(model.cfg),
    tokenizer={'kind':'utf8-byte','bos_id':256,'eos_id':257,'vocab_size':258},
    state_dict=detached CPU cloned tensors, provenance=caller dict of JSON-safe metadata.
    Require cfg.vocab_size=258. Return SHA256 of the exact file bytes. No optimizer in export.
    """
    import hashlib
    if model.cfg.vocab_size!=258:
        raise ValueError('Byte vocabulary required')
    torch.save(dict(format='toylm-v1',config=vars(model.cfg),tokenizer=dict(kind='utf8-byte',bos_id=256,eos_id=257,vocab_size=258),state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},provenance=provenance),path)
    return hashlib.sha256(__import__('pathlib').Path(path).read_bytes()).hexdigest()
'''
tests='''import copy
import hashlib
import math
import pytest
import torch


def make(api,seed=7):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return api.TinyLM(api.TinyConfig(dim=16,n_layers=1,n_heads=2,n_kv_heads=1,hidden_dim=32))


def test_01(api):
    assert api.encode_bytes('é')==[256,195,169,257]
    assert api.encode_bytes('')==[256,257]


def test_02(api):
    x,y=api.next_token_batch([1,2,3,4,5],[0,2],2)
    assert x.tolist()==[[1,2],[3,4]] and y.tolist()==[[2,3],[4,5]]
    assert x.dtype==y.dtype==torch.long
    for starts,ctx in [([],2),([-1],2),([3],2),([0],0)]:
        with pytest.raises(ValueError): api.next_token_batch([1,2,3,4,5],starts,ctx)


def test_03(api,seed):
    g=torch.Generator().manual_seed(seed); x=(torch.randn(2,3,7,generator=g)*100).requires_grad_(); y=torch.randint(7,(2,3),generator=g)
    got=api.token_loss(x,y); ref=torch.nn.functional.cross_entropy(x.flatten(0,1),y.flatten())
    torch.testing.assert_close(got,ref)
    torch.testing.assert_close(torch.autograd.grad(got,x)[0],torch.autograd.grad(ref,x)[0])


def test_04(api):
    n=api.RMSNorm(4); n.weight.data.copy_(torch.tensor([1,2,3,4.])); x=torch.tensor([[0.,0,0,0],[1.,2,3,4]])
    torch.testing.assert_close(n(x),x/torch.sqrt(x.square().mean(-1,keepdim=True)+1e-6)*n.weight)


def test_05(api):
    x=torch.tensor([[[1.,2.,3.,4.]],[[1.,2.,3.,4.]]]); p=torch.tensor([0,3]); y=api.rope(x,p)
    angles=p[:,None,None]*torch.tensor([1.,.01])[None,None,:]
    expected=torch.stack([x[...,::2]*angles.cos()-x[...,1::2]*angles.sin(),x[...,1::2]*angles.cos()+x[...,::2]*angles.sin()],-1).flatten(-2)
    torch.testing.assert_close(y,expected)


def test_06(api,seed):
    g=torch.Generator().manual_seed(seed); q=torch.randn(3,4,4,generator=g); k=torch.randn(5,2,4,generator=g); v=torch.randn(5,2,4,generator=g)
    out=api.attention(q,k,v,2)
    for i in range(3):
        for h in range(4):
            expected=(k[:i+3,h//2]@q[i,h]/2).softmax(0)@v[:i+3,h//2]
            torch.testing.assert_close(out[i,h],expected)


def test_07(api,seed):
    torch.manual_seed(seed); cfg=api.TinyConfig(dim=16,n_heads=2,n_kv_heads=1,hidden_dim=32)
    b=api.DecoderLayer(cfg); x=torch.randn(5,16); q,k,v=b.project(x,torch.arange(5))
    assert q.shape==(5,2,8) and k.shape==v.shape==(5,1,8)
    z=b.attn_norm(x); torch.testing.assert_close(v,b.v(z).view(5,1,8))
    attended=torch.randn(5,2,8); residual=x+b.o(attended.flatten(1)); z=b.ffn_norm(residual)
    torch.testing.assert_close(b.finish(x,attended),residual+b.down(torch.nn.functional.silu(b.gate(z))*b.up(z)))


def test_08(api,seed):
    m=make(api,seed); a,_=m([256,1,2,3]); b,_=m([256,1,9,8]); torch.testing.assert_close(a[:2],b[:2])
    assert a.shape==(4,258)
    x=m.embedding(torch.tensor([256,1,2,3])); pos=torch.arange(4)
    for layer in m.layers:
        q,k,v=layer.project(x,pos); x=layer.finish(x,api.attention(q,k,v))
    torch.testing.assert_close(a,m.lm_head(m.norm(x)))
    a.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_09(api,seed):
    g=torch.Generator().manual_seed(seed); p=torch.randn(3,4,generator=g); ref=torch.nn.Parameter(p.clone()); opt=torch.optim.AdamW([ref],lr=.02,weight_decay=.1); state={}
    for _ in range(4):
        grad=torch.randn(3,4,generator=g); saved=grad.clone(); ref.grad=grad.clone(); opt.step()
        api.adamw_step([p],[grad],state,.02,weight_decay=.1)
        torch.testing.assert_close(p,ref); torch.testing.assert_close(grad,saved)
    assert state['step']==4


def test_10(api):
    assert api.cosine_lr(0,2,10,1)==0 and api.cosine_lr(2,2,10,1)==1
    assert api.cosine_lr(6,2,10,1,.1)==pytest.approx(.55)
    assert api.cosine_lr(99,2,10,1,.1)==pytest.approx(.1)
    assert api.cosine_lr(0,0,10,1)==1
    with pytest.raises(ValueError): api.cosine_lr(1,3,3,1)


def test_11(api):
    a=torch.nn.Parameter(torch.ones(2)); a.grad=torch.tensor([3.,4.]); b=torch.nn.Parameter(torch.ones(1))
    assert api.clip_grad([a,b],2)==pytest.approx(5); torch.testing.assert_close(a.grad,torch.tensor([1.2,1.6]))
    a.grad[0]=float('nan')
    with pytest.raises(ValueError): api.clip_grad([a],1)


def test_12(api):
    a=make(api); b=copy.deepcopy(a); seqs=[[1,2,3],[4,5,6,7,8]]
    opt=torch.optim.SGD(a.parameters(),lr=.03); refopt=torch.optim.SGD(b.parameters(),lr=.03)
    terms=[]
    for s in seqs:
        logits,_=b(s[:-1]); terms.append(torch.nn.functional.cross_entropy(logits,torch.tensor(s[1:]),reduction='sum'))
    expected=sum(terms)/6; expected.backward(); refopt.step()
    got=api.train_step(a,opt,seqs); assert got==pytest.approx(expected.item())
    for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p,q)
    with pytest.raises(ValueError): api.train_step(a,opt,[[1]])


def test_13(api,tmp_path):
    m=make(api); opt=torch.optim.AdamW(m.parameters()); api.train_step(m,opt,[[1,2,3]])
    path=tmp_path/'resume.pt'; api.save_checkpoint(path,m,opt,7); r=torch.load(path,weights_only=True)
    assert r['step']==7 and r['optimizer']['state']
    torch.testing.assert_close(r['rng'],torch.get_rng_state())
    assert set(r['model'])==set(m.state_dict())


def test_14(api,tmp_path):
    m=make(api); opt=torch.optim.AdamW(m.parameters(),lr=.01); api.train_step(m,opt,[[1,2,3]])
    path=tmp_path/'resume.pt'; api.save_checkpoint(path,m,opt,1); expected_rng=torch.rand(3)
    expected=api.train_step(m,opt,[[2,3,4]])
    other=make(api,9); otheropt=torch.optim.AdamW(other.parameters(),lr=.9)
    assert api.load_checkpoint(path,other,otheropt)==1; torch.testing.assert_close(torch.rand(3),expected_rng)
    assert api.train_step(other,otheropt,[[2,3,4]])==pytest.approx(expected)
    for a,b in zip(m.parameters(),other.parameters()): torch.testing.assert_close(a,b)


def test_15(api,seed):
    state=torch.get_rng_state().clone(); m,loss=api.train_run([[256,1,2,3,257]]*2,steps=25,seed=seed)
    assert len(loss)==25 and all(math.isfinite(v) for v in loss) and loss[-1]<loss[0]*.3
    torch.testing.assert_close(torch.get_rng_state(),state)
    _,again=api.train_run([[256,1,2,3,257]]*2,steps=25,seed=seed); assert loss==again


def test_16(api,tmp_path):
    m=make(api); path=tmp_path/'model.pt'; digest=api.export_model(path,m,{'dataset_sha256':'abc','seed':7})
    assert digest==hashlib.sha256(path.read_bytes()).hexdigest()
    record=torch.load(path,weights_only=True); assert record['format']=='toylm-v1'
    assert record['tokenizer']==dict(kind='utf8-byte',bos_id=256,eos_id=257,vocab_size=258)
    other=api.TinyLM(api.TinyConfig(**record['config'])); other.load_state_dict(record['state_dict'],strict=True)
    torch.testing.assert_close(m([1,2,3])[0],other([1,2,3])[0]); assert record['provenance']['seed']==7
'''
rows=[
('Define the token boundary','encode_bytes','The training objective depends on the tokenizer. A byte vocabulary is small, deterministic and handles any UTF-8 text without vocabulary fitting. BOS and EOS are distinct from byte values: empty text still has a beginning and an end. Tokenization choices must accompany exported weights because swapping token identities changes the model even when tensor shapes remain valid.','Use exactly 258 token IDs. Inference prompts omit EOS, while training documents include it.','What would happen if EOS were a normal byte value?'),
('Shift inputs and targets','next_token_batch','A causal language model predicts the next token at each input position. The target window therefore extends one position beyond the input window. Off-by-one mistakes can produce a model trained to copy its current input or predict across unintended boundaries. Explicit starting offsets make data selection testable separately from the model.','This stage samples within one stream. The data course adds document-isolated packing.','Trace context=3 over five tokens. Which starts are valid?'),
('Compute stable token loss','token_loss','Cross entropy compares unnormalized scores with observed classes. The selected target logit is subtracted from a stable log partition function. Flattening batch and sequence dimensions is valid only when every target is supervised equally; later courses introduce masks and counts. Autograd should flow through the numerical expression, so returning a Python number here would break training.','Return a scalar differentiable tensor. Only implement the expression; do not use the framework cross-entropy shortcut.','Why does adding a constant to every logit leave the loss unchanged?'),
('Normalize residual activations','RMSNorm','Residual streams accumulate information across layers. RMS normalization controls scale without subtracting a mean. A learned per-feature weight restores expressiveness. Computing squared magnitudes in float32 helps low-precision stability. The epsilon belongs inside the square root; moving it outside changes both forward values and gradients.','Allocate weight=ones(dim). Use eps=1e-6 by default. Preserve the input dtype after normalization.','Compare all-zero input and a constant nonzero vector.'),
('Encode relative position by rotation','rope','Rotary position embeddings rotate adjacent coordinate pairs by an angle determined by position and frequency. Rotation preserves each pair’s squared length while changing dot products according to position differences. The exact pairing convention is part of the checkpoint contract; alternating pairs and split-half conventions are not interchangeable.','x is [T,H,d], positions is [T], frequencies are base**(-arange(0,d,2)/d). Use adjacent pairs and base 10000.','Verify length preservation for positions 0, 1 and 100.'),
('Implement causal grouped-query attention','attention','Each query compares with permitted keys, normalizes those scores and averages values. Query heads may outnumber KV heads: consecutive query groups share the same KV head. The causal mask uses absolute positions, including a query_start offset. Using a plain triangular mask for a later chunk incorrectly hides part of its history.','Scores divide by sqrt(head_dim). Compute softmax in float32. Queries have shape [T,Hq,d], keys/values [S,Hkv,d].','Draw the mask for three queries beginning at position two.'),
('Build one decoder layer','DecoderLayer','A pre-normalized decoder alternates attention and a gated feed-forward transformation, with a residual connection around each. SwiGLU multiplies a SiLU gate by a separate up projection before returning to model width. Keeping projection and residual completion as separate methods makes the same weights usable later in packed inference.','Use bias-free q,k,v,o,gate,up,down linear layers; attn_norm and ffn_norm. project returns rotated q/k and unrotated v. finish applies attention residual then SwiGLU residual.','Explain why every projection need not have the same output width.'),
('Assemble a trainable decoder','TinyLM','A model is a parameterized composition, not just a function that emits the right shape. The output head projects the normalized residual stream to vocabulary scores. Matching module names and orientations gives the serving course an exact checkpoint interface. Causality can be tested by changing future tokens and checking that earlier logits remain identical.','State keys: embedding.weight; layers.N.{attn_norm.weight,q.weight,k.weight,v.weight,o.weight,ffn_norm.weight,gate.weight,up.weight,down.weight}; norm.weight; lm_head.weight. Implement dense forward, device and _ids. Cached forward is optional here.','Count parameters and estimate bytes before allocating a larger config.'),
('Implement AdamW','adamw_step','Adam tracks a moving mean and uncentered second moment of gradients. Both estimates initially underestimate their steady-state magnitude and need step-dependent bias correction. AdamW applies weight decay directly to parameters, separately from the adaptive gradient update. Confusing it with adding an L2 term to the gradient produces different dynamics.','Use an initially empty mutable state dictionary. Dense gradients only; sparse and missing gradients are outside this stage.','Compare a zero gradient update with nonzero weight decay.'),
('Schedule the learning rate','cosine_lr','Warmup and decay encode a choice about how aggressively to change a model over time. Endpoints are part of the API: this course defines the first warmup rate as zero and reaches peak exactly at warmup. Explicit clamping prevents a cosine schedule from rising again after the planned training horizon.','Validate bounds rather than dividing by zero for an invalid schedule.','Plot the schedule with and without warmup and label its endpoints.'),
('Clip a global gradient norm','clip_grad','Clipping all gradients together preserves their direction while limiting the magnitude of the proposed update. Clipping each tensor independently changes that direction. Reporting the preclip norm is useful for diagnosing unstable training. Nonfinite values must stop the step rather than turning parameters into NaNs.','The threshold applies over every parameter gradient combined. Ignore missing gradients.','Show a two-parameter example where per-tensor clipping differs from global clipping.'),
('Perform one token-weighted update','train_step','Unequal sequence lengths expose a common reduction bug. Averaging sequence losses equally gives a token in a short sequence more weight than one in a long sequence. Summing token losses and dividing by the total supervised count matches the intended objective. Clear previous gradients before computing the new step.','Use a supplied torch optimizer; the earlier manual AdamW exercise established its mechanics. Do not detach the loss until after backpropagation.','Compare a batch of lengths 2 and 20 under sequence and token averaging.'),
('Save resumable state','save_checkpoint','Weights alone do not describe the state of training. Adam moments, the update counter and random-number state affect the next update. A resumable checkpoint and an inference export serve different purposes. This stage stores trusted local training state; later the release course records content identity and governs which artifacts become active.','Save the CPU RNG as a tensor. The caller owns directory creation.','List additional state needed for dropout, a shuffled data iterator and a learning-rate scheduler.'),
('Resume without changing the trajectory','load_checkpoint','A strong resume test compares the next update with uninterrupted training. Merely checking that a file loads misses omitted optimizer moments and an incorrect learning rate. Restoring RNG also ensures later stochastic choices follow the saved trajectory. Exact reproducibility here is a CPU teaching guarantee, not a promise across arbitrary GPU libraries and hardware.','Restore into a newly constructed matching model and optimizer, including saved optimizer hyperparameters.','Deliberately omit optimizer restoration and compare the next update.'),
('Train and diagnose a tiny model','train_run','Overfitting a tiny controlled corpus is a useful systems diagnostic. It shows the loss, gradients and optimizer are connected. It does not show generalization. Repeated local seeds make regressions visible and separate initialization effects from coding changes. A fresh initialization belongs inside a random-state scope so running the exercise does not alter unrelated experiments.','Use train_step rather than a second independently coded training loop. The acceptance task tests mechanics; use the capstone for held-out quality.','Compare memorization loss with loss on a held-out pattern.'),
('Export a portable model artifact','export_model','Serving needs weights, architecture configuration and tokenizer identity together. Optimizer moments are unnecessary. A digest identifies exact file bytes; it does not by itself prove the quality or provenance of the training data. The shared toylm-v1 format makes it possible to test a training-to-inference handoff with strict parameter loading and logit equivalence.','The shared artifact contract is documented at ../ARTIFACTS.md from the course root. Do not relabel an incompatible vocabulary.','Verify exported logits before attempting any serving optimization.')]
stages=[(*r,['Start from the shapes and stated boundary behavior.','Use a tiny hand-computable example before a random case.','Check normalization, ordering and state ownership independently.']) for r in rows]
root=build('02-pretraining','Train a tiny language model','Build a byte-level causal transformer, understand its optimizer, resume training exactly, and export the same architecture used by the inference course. Sixteen stages separate mathematical components from training workflow. Use PyTorch autograd here; course 01 explains what it does.',stages,source,tests,[('CS336: Language Modeling from Scratch','https://cs336.stanford.edu/'),('Attention Is All You Need','https://arxiv.org/abs/1706.03762'),('AdamW','https://arxiv.org/abs/1711.05101')],provided=('TinyConfig',))
