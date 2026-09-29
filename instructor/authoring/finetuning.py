from build import build
source='''"""Instructor reference: instruction tuning, adapters and preference optimization."""
import hashlib
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


def chat_tokens(messages):
    """Return (ids, assistant_mask) lists. Start with BOS=256 (mask false).
    For each {'role': system|user|assistant, 'content':str}, append UTF-8 bytes of
    '<|ROLE|>\\n' (mask false), UTF-8 content (mask true iff assistant), then EOS=257
    (mask true iff assistant). Reject empty message lists, unknown roles or non-string content.
    Literal role headers are ordinary bytes, not additional vocabulary IDs.
    """
    if not messages: raise ValueError('Messages required')
    ids=[256]; mask=[False]
    for m in messages:
        role=m['role']; content=m['content']
        if role not in {'system','user','assistant'} or not isinstance(content,str): raise ValueError('Invalid message')
        header=list(f'<|{role}|>\\n'.encode()); body=list(content.encode())+[257]
        ids.extend(header+body); mask.extend([False]*len(header)+[role=='assistant']*len(body))
    return ids,mask

def collate(examples, pad_id=0):
    """examples are (ids,assistant_mask) pairs with equal nonzero lengths. Return dict
    input_ids long [B,T], labels long [B,T] (original ID if mask true, else -100),
    attention_mask bool [B,T]. Right-pad; padding labels=-100, attention=false.
    Reject empty batches, empty examples and mismatched lengths. Do not shift labels here.
    """
    if not examples or any(not ids or len(ids)!=len(mask) for ids,mask in examples): raise ValueError('Invalid examples')
    size=max(len(ids) for ids,_ in examples)
    ids=torch.full((len(examples),size),pad_id,dtype=torch.long)
    labels=torch.full_like(ids,-100); attention=torch.zeros_like(ids,dtype=torch.bool)
    for i,(tokens,mask) in enumerate(examples):
        n=len(tokens); ids[i,:n]=torch.tensor(tokens); attention[i,:n]=True
        labels[i,:n]=torch.where(torch.tensor(mask,dtype=torch.bool),ids[i,:n],-100)
    return dict(input_ids=ids,labels=labels,attention_mask=attention)

def masked_loss(logits, labels):
    """Next-token mean CE: logits [B,T,V] at :-1 predict labels at 1:.
    Ignore labels=-100, divide by total supervised tokens across batch. Raise ValueError
    when no shifted target is supervised. Return differentiable scalar tensor.
    """
    targets=labels[:,1:]; mask=targets!=-100
    if not mask.any(): raise ValueError('No supervised target tokens')
    logp=logits[:,:-1].log_softmax(-1)
    selected=logp.gather(-1,targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return -selected[mask].mean()

class LoRALinear(nn.Module):
    """Wrap an nn.Linear as base. Require rank>=1 and alpha>0. Freeze base parameters.
    A is nn.Parameter [rank,in], Kaiming-uniform initialized; B is zeros [out,rank].
    Both match base dtype/device. Forward: base(x) + (x @ A.T @ B.T)*(alpha/rank).
    Arbitrary leading input dimensions are supported. Do not modify base weights in forward.
    """
    def __init__(self, base, rank=2, alpha=2.):
        super().__init__()
        if rank<1 or alpha<=0: raise ValueError('Invalid adapter')
        self.base=base; self.rank=rank; self.alpha=alpha
        for p in self.base.parameters(): p.requires_grad_(False)
        self.A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=base.weight.dtype))
        self.B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.A,a=5**.5)
    def forward(self,x):
        return self.base(x)+(x@self.A.T@self.B.T)*(self.alpha/self.rank)

def inject_lora(model, targets, rank=2, alpha=2.):
    """In-place replace exact qualified names of nn.Linear modules with LoRALinear.
    Validate every name and type before changing any module. Reject duplicates, missing names,
    empty targets and non-linear targets. Return the same model. Other modules stay untouched.
    """
    modules=dict(model.named_modules())
    if not targets or len(set(targets))!=len(targets) or any(not n or not isinstance(modules.get(n),nn.Linear) for n in targets): raise ValueError('Invalid targets')
    for name in targets:
        parent,_,leaf=name.rpartition('.')
        setattr(modules[parent],leaf,LoRALinear(modules[name],rank,alpha))
    return model

def freeze_except_adapters(model):
    """Freeze all parameters except actual LoRALinear A/B objects. Return unique trainable
    Parameters in model.parameters() order. Reject models without adapters. Names alone
    must not cause unrelated parameters named A or B to become trainable.
    """
    allowed={id(p) for m in model.modules() if isinstance(m,LoRALinear) for p in [m.A,m.B]}
    if not allowed: raise ValueError('No adapters found')
    selected=[]
    for p in model.parameters():
        p.requires_grad_(id(p) in allowed)
        if p.requires_grad: selected.append(p)
    return selected

def sft_step(model, optimizer, batch):
    """One optimizer update using model(input_ids)->[B,T,V] logits and masked_loss.
    Set train mode, clear old grads, backpropagate, step; return pre-update float loss.
    The teaching model interface needs no attention mask because test fixtures are causal
    positionwise networks; capstone wrappers must handle padding and document isolation.
    """
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss=masked_loss(model(batch['input_ids']),batch['labels'])
    loss.backward(); optimizer.step()
    return float(loss.detach())

def merge_lora(layer):
    """Return a new ordinary nn.Linear on the same device/dtype with W + alpha/rank * B@A
    and the same bias. Preserve the input adapter and all its parameters. Merging twice
    from the same original gives the same result; do not add the delta into base in place.
    """
    base=layer.base
    out=nn.Linear(base.in_features,base.out_features,bias=base.bias is not None,device=base.weight.device,dtype=base.weight.dtype)
    with torch.no_grad():
        out.weight.copy_(base.weight+(layer.B@layer.A)*(layer.alpha/layer.rank))
        if base.bias is not None: out.bias.copy_(base.bias)
    return out

def sequence_logps(logits, labels):
    """Return [B] SUM of next-token log probabilities for labels[:,1:] excluding -100.
    An unsupervised row contributes zero. Keep gradients; do not length-normalize for DPO.
    """
    target=labels[:,1:]; valid=target!=-100
    chosen=logits[:,:-1].log_softmax(-1).gather(-1,target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (chosen*valid).sum(-1)

def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=.1):
    """Mean -logsigmoid(beta*((pc-pr)-(rc-rr))). All arguments [B]; beta>0.
    Detach reference terms so the frozen baseline receives no gradient. Stable for large margins.
    """
    if beta<=0: raise ValueError('beta must be positive')
    margin=(policy_chosen-policy_rejected)-(reference_chosen-reference_rejected).detach()
    return -F.logsigmoid(beta*margin).mean()

def dpo_step(policy, reference, optimizer, chosen, rejected, beta=.1):
    """chosen/rejected are collate-style batches of equal batch size. One policy update
    from summed response log probabilities. Run reference in eval mode under no_grad;
    clear policy gradients, leave reference parameters unchanged, return float DPO loss.
    """
    policy.train(); reference.eval(); optimizer.zero_grad(set_to_none=True)
    pc=sequence_logps(policy(chosen['input_ids']),chosen['labels'])
    pr=sequence_logps(policy(rejected['input_ids']),rejected['labels'])
    with torch.no_grad():
        rc=sequence_logps(reference(chosen['input_ids']),chosen['labels'])
        rr=sequence_logps(reference(rejected['input_ids']),rejected['labels'])
    loss=dpo_loss(pc,pr,rc,rr,beta); loss.backward(); optimizer.step()
    return float(loss.detach())

def adapter_state(model):
    """Return dict qualified_name+'.A'/'.B' -> detached CPU cloned tensors for actual
    LoRALinear modules only. Reject models with no adapters. No base weights in this state.
    """
    state={}
    for name,m in model.named_modules():
        if isinstance(m,LoRALinear):
            state[name+'.A']=m.A.detach().cpu().clone(); state[name+'.B']=m.B.detach().cpu().clone()
    if not state: raise ValueError('No adapters')
    return state

def save_adapter(path, model, base_sha256):
    """Save trusted local torch payload format='toyadapter-v1', base_sha256 (64 hex chars),
    state=adapter_state(model), config={module_name:{rank,alpha}}. Return SHA256 file digest.
    Reject malformed base digest. The base digest identifies a toylm-v1 artifact externally.
    """
    if len(base_sha256)!=64 or any(c not in '0123456789abcdef' for c in base_sha256): raise ValueError('Invalid digest')
    config={n:dict(rank=m.rank,alpha=m.alpha) for n,m in model.named_modules() if isinstance(m,LoRALinear)}
    torch.save(dict(format='toyadapter-v1',base_sha256=base_sha256,state=adapter_state(model),config=config),path)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_adapter(path, model, expected_base_sha256):
    """Load onto an already injected matching model. Validate format, base digest, config,
    all adapter keys and shapes BEFORE modifying anything; mismatch raises ValueError.
    Copy tensors in place under no_grad, converting to destination dtype/device. Base stays fixed.
    """
    record=torch.load(path,weights_only=True,map_location='cpu')
    current=adapter_state(model)
    config={n:dict(rank=m.rank,alpha=m.alpha) for n,m in model.named_modules() if isinstance(m,LoRALinear)}
    if record.get('format')!='toyadapter-v1' or record.get('base_sha256')!=expected_base_sha256 or record.get('config')!=config or set(record.get('state',{}))!=set(current) or any(record['state'][k].shape!=v.shape for k,v in current.items()): raise ValueError('Incompatible adapter')
    params=dict(model.named_parameters())
    with torch.no_grad():
        for k,v in record['state'].items(): params[k].copy_(v)
'''
tests='''import copy
import hashlib
import math
import pytest
import torch
from torch import nn

class TokenModel(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding=nn.Embedding(258,8); self.head=nn.Linear(8,258)
    def forward(self,x): return self.head(self.embedding(x))

def test_01(api):
    ids,mask=api.chat_tokens([dict(role='user',content='é'),dict(role='assistant',content='ok')])
    assert ids==[256]+list('<|user|>\\né'.encode())+[257]+list(b'<|assistant|>\\nok')+[257]
    assert [x for x,m in zip(ids,mask) if m]==list(b'ok')+[257]
    with pytest.raises(ValueError): api.chat_tokens([dict(role='tool',content='x')])


def test_02(api):
    b=api.collate([([256,1,2],[False,False,True]),([256,3],[False,True])])
    assert b['labels'].tolist()==[[-100,-100,2],[-100,3,-100]]
    assert b['input_ids'].tolist()==[[256,1,2],[256,3,0]] and b['attention_mask'].tolist()==[[True]*3,[True,True,False]]
    with pytest.raises(ValueError): api.collate([([1],[True,False])])


def test_03(api,seed):
    x=torch.randn(2,4,7,generator=torch.Generator().manual_seed(seed),requires_grad=True); y=torch.tensor([[-100,-100,2,3],[-100,4,-100,-100]])
    ref=torch.nn.functional.cross_entropy(x[:,:-1].reshape(-1,7),y[:,1:].reshape(-1),ignore_index=-100)
    loss=api.masked_loss(x,y); torch.testing.assert_close(loss,ref)
    grad=torch.autograd.grad(loss,x)[0]; assert torch.count_nonzero(grad[:,-1])==0 and torch.count_nonzero(grad[0,0])==0
    with pytest.raises(ValueError): api.masked_loss(x,torch.full_like(y,-100))


def test_04(api,seed):
    torch.manual_seed(seed); base=nn.Linear(4,3,dtype=torch.float64); layer=api.LoRALinear(base,2,6); x=torch.randn(2,5,4,dtype=torch.float64)
    torch.testing.assert_close(layer(x),base(x)); assert layer.A.shape==(2,4) and layer.B.shape==(3,2)
    assert not base.weight.requires_grad and not base.bias.requires_grad
    layer.B.data.normal_(); before=base.weight.clone()
    torch.testing.assert_close(layer(x),base(x)+(x@layer.A.T@layer.B.T)*3)
    layer(x).sum().backward(); assert layer.A.grad is not None and layer.B.grad is not None and base.weight.grad is None
    torch.testing.assert_close(before,base.weight)


def test_05(api):
    model=nn.Sequential(nn.Linear(4,4),nn.Sequential(nn.Linear(4,2)))
    old=model[0]; assert api.inject_lora(model,['1.0']) is model
    assert model[0] is old and isinstance(model[1][0],api.LoRALinear)
    fresh=nn.Sequential(nn.Linear(4,4),nn.ReLU()); before=fresh[0]
    with pytest.raises(ValueError): api.inject_lora(fresh,['0','1'])
    assert fresh[0] is before


def test_06(api):
    model=TokenModel(); api.inject_lora(model,['head']); selected=api.freeze_except_adapters(model)
    assert {id(p) for p in selected}=={id(model.head.A),id(model.head.B)}
    assert not model.embedding.weight.requires_grad
    with pytest.raises(ValueError): api.freeze_except_adapters(TokenModel())


def test_07(api,seed):
    torch.manual_seed(seed); model=TokenModel(); api.inject_lora(model,['head']); ps=api.freeze_except_adapters(model)
    batch=api.collate([([256,1,2,257],[False,False,True,True])]); before={n:p.clone() for n,p in model.named_parameters()}
    opt=torch.optim.SGD(ps,lr=.1); loss=api.sft_step(model,opt,batch); assert math.isfinite(loss)
    assert not torch.equal(model.head.B,before['head.B'])
    for n,p in model.named_parameters():
        if n not in ['head.A','head.B']: torch.testing.assert_close(p,before[n])


def test_08(api,seed):
    torch.manual_seed(seed); layer=api.LoRALinear(nn.Linear(4,3),2,3); layer.B.data.normal_(); x=torch.randn(7,4); before=layer.base.weight.clone()
    merged=api.merge_lora(layer); assert isinstance(merged,nn.Linear)
    torch.testing.assert_close(merged(x),layer(x)); torch.testing.assert_close(api.merge_lora(layer)(x),merged(x)); torch.testing.assert_close(layer.base.weight,before)


def test_09(api):
    x=torch.zeros(2,4,5,requires_grad=True); labels=torch.tensor([[-100,1,2,3],[-100,-100,2,-100]])
    out=api.sequence_logps(x,labels); torch.testing.assert_close(out,torch.tensor([-3*math.log(5),-math.log(5)]))
    out.sum().backward(); assert torch.count_nonzero(x.grad[1,0])==0
    assert api.sequence_logps(x,torch.full_like(labels,-100)).tolist()==[0,0]


def test_10(api):
    pc=torch.tensor([2.,-1000.],requires_grad=True); pr=torch.tensor([0.,1000.],requires_grad=True); rc=torch.tensor([1.,0.],requires_grad=True); rr=torch.zeros(2,requires_grad=True)
    loss=api.dpo_loss(pc,pr,rc,rr,.5)
    expected=(torch.nn.functional.softplus(torch.tensor(-.5))+torch.tensor(1000.))/2
    torch.testing.assert_close(loss,expected); loss.backward()
    assert (pc.grad<0).all() and (pr.grad>0).all() and rc.grad is None and rr.grad is None
    with pytest.raises(ValueError): api.dpo_loss(pc,pr,rc,rr,0)


def test_11(api,seed):
    torch.manual_seed(seed); model=TokenModel(); reference=copy.deepcopy(model); opt=torch.optim.SGD(model.parameters(),lr=.1)
    c=api.collate([([256,1,2],[False,False,True])]); r=api.collate([([256,1,3],[False,False,True])]); before={k:v.clone() for k,v in reference.state_dict().items()}
    loss=api.dpo_step(model,reference,opt,c,r,beta=1); assert loss==pytest.approx(math.log(2))
    for k,v in reference.state_dict().items(): torch.testing.assert_close(v,before[k])
    assert all(p.grad is None for p in reference.parameters())
    margin=(api.sequence_logps(model(c['input_ids']),c['labels'])-api.sequence_logps(model(r['input_ids']),r['labels'])).item()
    baseline=(api.sequence_logps(reference(c['input_ids']),c['labels'])-api.sequence_logps(reference(r['input_ids']),r['labels'])).item()
    assert margin>baseline


def test_12(api):
    model=TokenModel(); api.inject_lora(model,['head']); state=api.adapter_state(model)
    assert set(state)=={'head.A','head.B'} and all(not x.requires_grad and x.device.type=='cpu' for x in state.values())
    saved=model.head.A.clone(); state['head.A'].add_(3); torch.testing.assert_close(model.head.A,saved)


def test_13(api,tmp_path):
    model=TokenModel(); api.inject_lora(model,['head']); path=tmp_path/'adapter.pt'
    digest=api.save_adapter(path,model,'a'*64); assert digest==hashlib.sha256(path.read_bytes()).hexdigest()
    r=torch.load(path,weights_only=True); assert r['format']=='toyadapter-v1' and r['base_sha256']=='a'*64 and r['config']=={'head':{'rank':2,'alpha':2.}}
    with pytest.raises(ValueError): api.save_adapter(path,model,'bad')


def test_14(api,tmp_path):
    model=TokenModel(); api.inject_lora(model,['head']); model.head.B.data.fill_(2); path=tmp_path/'adapter.pt'; api.save_adapter(path,model,'a'*64)
    other=TokenModel(); api.inject_lora(other,['head']); base=other.head.base.weight.clone()
    api.load_adapter(path,other,'a'*64); torch.testing.assert_close(other.head.B,model.head.B); torch.testing.assert_close(base,other.head.base.weight)
    before={k:v.clone() for k,v in other.state_dict().items()}
    with pytest.raises(ValueError): api.load_adapter(path,other,'b'*64)
    for k,v in other.state_dict().items(): torch.testing.assert_close(v,before[k])
    r=torch.load(path,weights_only=True); r['state']['head.B']=torch.ones(999,2); torch.save(r,path)
    with pytest.raises(ValueError): api.load_adapter(path,other,'a'*64)
    for k,v in other.state_dict().items(): torch.testing.assert_close(v,before[k])
'''
rows=[
('Serialize supervised conversations','chat_tokens','Instruction tuning begins with an exact token sequence. Role headers, separators and end markers influence what the model learns to emit. Supervision typically belongs to assistant responses rather than user prompts. A response mask must be created alongside serialization; reconstructing it later by searching token strings can confuse literal user content with control syntax.','Use the byte vocabulary shared with pretraining. This toy template is explicit and is not interchangeable with a downloaded model’s chat template.','What breaks if a user types a role header inside their message?'),
('Collate variable-length examples','collate','Padding makes examples rectangular, but padded positions must not become training targets. Attention visibility and loss supervision answer different questions: whether a token is readable and whether predicting it contributes to the objective. Keeping both masks makes that distinction visible. Labels remain aligned to original input positions at this stage.','Right-pad only. Loss labels use -100; attention_mask is boolean.','Explain why a real prompt token can have attention=true but label=-100.'),
('Shift and mask supervised loss','masked_loss','A logit at position t predicts token t+1. Shifting in both the collator and loss would move targets two steps, while never shifting trains copying. The denominator is the number of supervised targets, not padded positions or sequences. An empty supervised batch should fail visibly rather than return NaN or a misleading zero.','Perform exactly one shift inside this function. Never supervise padding.','Manually mark the logit positions responsible for a two-token assistant answer.'),
('Implement low-rank adaptation','LoRALinear','LoRA represents a weight update as the product of two smaller matrices. Freezing the original weight reduces optimizer state and lets the same base support several adapters. A nonzero A and zero B start from exactly the base function while allowing B to receive a gradient. Initializing both factors to zero would prevent either from learning.','Support leading batch and sequence dimensions. Preserve base bias and dtype/device.','Which factor receives a gradient on the first update, and why?'),
('Inject adapters deliberately','inject_lora','Target selection is an architectural decision. Exact qualified module names avoid accidentally adapting every layer whose name contains a substring. Validate the complete plan before replacement so a misspelled later target cannot leave the model half-modified. This stage replaces existing module objects and preserves all unaffected modules.','Targets refer to named_modules paths, including numeric Sequential children.','Compare adapting the output head with adapting attention projections.'),
('Control the trainable boundary','freeze_except_adapters','An optimizer only sees the parameters it is given, but autograd may still build gradients for other trainable parameters. Explicit freezing clarifies the intended training boundary and saves work. Parameter identity is stronger than a name suffix: an unrelated module could legitimately have a parameter called A.','Return the actual trainable Parameter objects, without duplicates.','Measure trainable parameter count before and after injection.'),
('Run one supervised update','sft_step','A fine-tuning step reuses the familiar training loop with a changed supervision policy and trainable parameter set. Testing frozen weights before and after the update catches accidental base training. The first LoRA update can leave A unchanged because B starts at zero; tests should check expected learning mechanics rather than assume every parameter moves immediately.','The optimizer should be constructed from freeze_except_adapters. The small test model is positionwise; real transformer masking remains the caller’s responsibility.','Compare a fully supervised loss with assistant-only loss on the same batch.'),
('Merge an adapter for inference','merge_lora','The low-rank update is linear and can be folded into the original weight for inference. Doing so removes the extra adapter matrix multiplications. Numerical equivalence should be checked before any benchmark. Returning a fresh layer avoids a double-merge bug in which repeated calls silently apply the same update again.','This stage merges one layer; the capstone traverses and replaces selected wrappers.','How would you serve two adapters if both were merged into the same base object?'),
('Score whole responses','sequence_logps','Preference optimization compares response probabilities conditioned on prompts. Summing token log probabilities produces the log probability of the sequence under the autoregressive factorization. Averaging instead changes the objective and its relationship to response length. Prompt tokens are excluded from the sum but still supply context to the model.','Unlike masked_loss, return one sum per example and permit all-masked rows.','Compare two equally probable-per-token responses of different lengths.'),
('Derive the DPO objective','dpo_loss','DPO increases the policy’s chosen-versus-rejected log-probability margin relative to a fixed reference. Subtracting reference margins accounts for the base model’s existing preferences. A stable log-sigmoid avoids overflow for strongly incorrect preferences. The reference is a constant in differentiation, even if a caller supplies tensors that require gradients.','beta scales the reference-relative margin; require it to be positive.','Derive the sign of each policy gradient when the chosen response is disfavored.'),
('Optimize a preference pair','dpo_step','The reference model defines a fixed comparison point. Accidentally updating it changes the objective during the run. Disabling its gradients is separate from setting evaluation mode: the latter controls behaviors such as dropout, while the former controls the differentiation graph. A one-step test can verify the preferred response’s relative margin moves in the correct direction.','Equal policy and reference initially imply loss log(2), regardless of the original margin.','Why can a lower DPO loss coexist with worse performance on an unrelated task?'),
('Extract portable adapter weights','adapter_state','An adapter checkpoint should contain adaptation state rather than a second copy of the base model. Cloning detached CPU tensors prevents future updates from modifying a supposedly captured state. The qualified names describe where each matrix belongs and must agree with the receiving architecture.','Only actual adapter objects count; ignore similarly named unrelated parameters.','Estimate storage for a rank-r adapter versus a full weight matrix.'),
('Bind an adapter to its base','save_adapter','Adapter matrices have meaning relative to the base weights used during training. Two models with identical tensor shapes can require different adapters. A base artifact digest gives this relationship an explicit identity. Rank and scaling belong in the artifact too, because the same matrices produce different updates under a different alpha.','The caller supplies the previously verified base artifact digest. A valid hex string alone is not evidence that the base file exists.','What should an experiment registry record besides the adapter file?'),
('Restore an adapter atomically','load_adapter','A failed restore should leave a model usable in its previous state. Validate all keys, shapes, scaling and base identity before copying the first tensor. This matters when one early tensor is compatible but a later one is not. A strict boundary turns silent partial loading into an actionable error.','This is a trusted local torch artifact using weights_only=True. Reject incompatibility with ValueError.','Corrupt the second tensor and verify that the first tensor was not changed.')]
stages=[(*r,['Locate the supervised positions or trainable state first.','Check the smallest case where a mask, shift or ownership rule matters.','Compare values and gradients separately; a matching output can hide a wrong update.']) for r in rows]
build('04-finetuning','Build a fine-tuning and preference system','Adapt models through fourteen stages covering conversation serialization, masked supervision, LoRA, supervised updates, DPO and portable adapters. The grader uses tiny controlled models; the capstone applies these components to the checkpoint you trained.',stages,source,tests,[('LoRA paper','https://arxiv.org/abs/2106.09685'),('DPO paper','https://arxiv.org/abs/2305.18290'),('Hugging Face Smol Course','https://huggingface.co/learn/smol-course/')])
