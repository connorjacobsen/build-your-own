from build import build
source='''"""Instructor reference for real-process distributed training exercises."""
import math
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint


def partition_indices(size, rank, world_size):
    """Strided indices rank,rank+world_size,... below size. No duplication or padding.
    Require size>=0, world_size>=1, 0<=rank<world_size. Return list[int].
    """
    if size<0 or world_size<1 or not 0<=rank<world_size: raise ValueError('Invalid partition')
    return list(range(rank,size,world_size))

def shard_bounds(total, rank, world_size):
    """Balanced contiguous [start,end) ownership. First total%world_size ranks own one extra.
    Require total>=0,world_size>=1, valid rank. Empty shards allowed.
    """
    if total<0 or world_size<1 or not 0<=rank<world_size: raise ValueError('Invalid shard')
    base,extra=divmod(total,world_size); start=rank*base+min(rank,extra)
    return start,start+base+(rank<extra)

def flatten_tensors(tensors):
    """Concatenate flattened nonempty list of same-dtype, same-device tensors into a new 1D
    tensor. Preserve autograd connectivity, tensor order and values. Zero-size tensors allowed.
    Reject empty list, mixed dtypes or mixed devices. Do not modify inputs.
    """
    if not tensors or any(t.dtype!=tensors[0].dtype or t.device!=tensors[0].device for t in tensors): raise ValueError('Incompatible tensors')
    return torch.cat([t.reshape(-1) for t in tensors])

def unflatten(flat, shapes):
    """Return views into rank-one flat with specified shapes, in order. Require exact element
    count and all shape dimensions nonnegative. Scalar shape () consumes one element.
    Mutating a returned view must mutate the corresponding flat segment.
    """
    sizes=[math.prod(s) for s in shapes]
    if flat.ndim!=1 or any(d<0 for s in shapes for d in s) or sum(sizes)!=flat.numel(): raise ValueError('Invalid layout')
    result=[]; offset=0
    for shape,size in zip(shapes,sizes):
        result.append(flat[offset:offset+size].view(shape)); offset+=size
    return result

def accumulate_gradients(model, microbatches, loss_sum_fn):
    """Each microbatch is (x,y), nonempty and equal leading size. loss_sum_fn(model(x),y)
    returns SUM over examples. Clear grads once, backprop each loss/total_examples, return
    total detached loss/total_examples. Do not optimizer.step. Unequal microbatch sizes matter.
    """
    if not microbatches or any(len(x)==0 or len(x)!=len(y) for x,y in microbatches): raise ValueError('Invalid microbatches')
    total=sum(len(y) for _,y in microbatches); model.zero_grad(set_to_none=True); loss=0.
    for x,y in microbatches:
        value=loss_sum_fn(model(x),y); (value/total).backward(); loss+=float(value.detach())
    return loss/total

def weighted_allreduce(local_mean, local_count):
    """With initialized process group, return count-weighted global mean, leaving input intact.
    local_count>=0; zero count contributes zero even if its local mean is arbitrary.
    All ranks must call with equal tensor shape/dtype/device. Global count zero raises ValueError
    on all ranks. Counts travel as float64 tensors on the input device. Use SUM collectives.
    """
    if not dist.is_initialized() or local_count<0: raise ValueError('Invalid group or count')
    value=local_mean.clone()*local_count if local_count else torch.zeros_like(local_mean)
    count=torch.tensor(float(local_count),dtype=torch.float64,device=value.device)
    dist.all_reduce(value); dist.all_reduce(count)
    if count.item()==0: raise ValueError('No global examples')
    return value/count.to(value.dtype)

def distributed_step(model, optimizer, x, y):
    """One global mean-squared-error update with initialized group; model(x) and y both [N,1].
    Unequal local N, including zero, are allowed; total N must be positive. Every rank executes
    collectives in identical order. Sum local squared errors and gradients globally, divide by
    GLOBAL example count, then step. Return identical global mean loss on every rank.
    """
    model.train(); optimizer.zero_grad(set_to_none=True)
    output=model(x); local=(output-y).square().sum(); local.backward()
    count=torch.tensor(float(len(y)),device=x.device,dtype=torch.float64)
    dist.all_reduce(count)
    if count.item()==0: raise ValueError('No examples')
    for p in model.parameters():
        dist.all_reduce(p.grad)
        p.grad.div_(count.to(p.grad.dtype))
    loss=local.detach().clone(); dist.all_reduce(loss); loss.div_(count.to(loss.dtype))
    optimizer.step()
    return float(loss)

def shard_state(momentum, rank, world_size):
    """For a flat global momentum tensor return dict start,end,momentum containing only
    an owned clone of this rank's balanced contiguous shard. Do not retain global storage.
    """
    start,end=shard_bounds(momentum.numel(),rank,world_size)
    return dict(start=start,end=end,momentum=momentum[start:end].clone())

def sharded_momentum_step(parameters, state, lr, momentum=.9):
    """Initialized group, identical parameters and already globally averaged grads on each rank.
    Keep ONLY a local momentum shard in state (initially {}). Update local shard using
    velocity=momentum*velocity+grad, param-=lr*velocity; all_gather padded equal-size shards
    and restore full replicated parameters. state keys start,end,momentum. No full optimizer
    state replication. This is an educational ZeRO-1-style step, not full parameter sharding.
    """
    flat=flatten_tensors([p.detach() for p in parameters]); grads=flatten_tensors([p.grad for p in parameters])
    world=dist.get_world_size(); rank=dist.get_rank(); start,end=shard_bounds(flat.numel(),rank,world)
    if not state: state.update(start=start,end=end,momentum=torch.zeros_like(flat[start:end]))
    state['momentum'].mul_(momentum).add_(grads[start:end])
    piece=flat[start:end]-lr*state['momentum']
    width=(flat.numel()+world-1)//world
    padded=torch.zeros(width,device=flat.device,dtype=flat.dtype); padded[:end-start]=piece
    gathered=[torch.empty_like(padded) for _ in range(world)]; dist.all_gather(gathered,padded)
    full=torch.cat([part[:shard_bounds(flat.numel(),r,world)[1]-shard_bounds(flat.numel(),r,world)[0]] for r,part in enumerate(gathered)])
    with torch.no_grad():
        for p,v in zip(parameters,unflatten(full,[tuple(p.shape) for p in parameters])): p.copy_(v)

def unscale_gradients(parameters, scale):
    """Positive finite scale. If any present gradient is nonfinite, return False and change
    nothing. Otherwise divide all present grads by scale and return True. Ignore None grads.
    This stage is local; collective_finite later coordinates the skip decision across ranks.
    """
    if not math.isfinite(scale) or scale<=0: raise ValueError('Invalid scale')
    ps=[p for p in parameters if p.grad is not None]
    if any(not torch.isfinite(p.grad).all() for p in ps): return False
    for p in ps: p.grad.div_(scale)
    return True

def checkpointed(function, *args):
    """Call PyTorch non-reentrant activation checkpointing, preserving RNG state.
    Forward outputs and backward gradients must match ordinary execution, while the function
    executes again during backward when intermediates are required. This is activation
    recomputation, not serialization of model weights. Use the framework checkpoint primitive.
    """
    return checkpoint(function,*args,use_reentrant=False,preserve_rng_state=True)

def consolidate_shards(shards, total):
    """Reconstruct 1D momentum from dicts start,end,momentum. Input order arbitrary.
    Validate gap-free, nonoverlapping coverage [0,total), matching slice lengths and rank-one
    tensors BEFORE concatenation. Empty slices allowed. Require at least one shard.
    Return new owned tensor; reject malformed layouts with ValueError.
    """
    if not shards or total<0: raise ValueError('Invalid shards')
    ordered=sorted(shards,key=lambda s:(s['start'],s['end'])); cursor=0
    for shard in ordered:
        start,end=shard['start'],shard['end']; data=shard['momentum']
        if start!=cursor or end<start or end>total or data.ndim!=1 or data.numel()!=end-start: raise ValueError('Invalid shard coverage')
        cursor=end
    if cursor!=total: raise ValueError('Missing shard')
    return torch.cat([s['momentum'] for s in ordered])

def collective_finite(parameters):
    """Initialized group and nonempty parameter list all on one device. Return bool true
    on ALL ranks iff every present gradient on EVERY rank is finite. None is allowed.
    Do not change gradients. Reduce a device-local int flag using MIN, so one bad rank
    makes the same skip decision everywhere. Call before any optimizer update.
    """
    good=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
    flag=torch.tensor(int(good),dtype=torch.int32,device=parameters[0].device)
    dist.all_reduce(flag,op=dist.ReduceOp.MIN)
    return bool(flag.item())
'''
tests='''import copy
import importlib
from datetime import timedelta
from pathlib import Path
import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp


def test_01(api):
    parts=[api.partition_indices(7,r,3) for r in range(3)]
    assert parts==[[0,3,6],[1,4],[2,5]] and sorted(sum(parts,[]))==list(range(7))
    assert api.partition_indices(1,2,3)==[]
    with pytest.raises(ValueError): api.partition_indices(5,3,3)


def test_02(api):
    assert [api.shard_bounds(8,r,3) for r in range(3)]==[(0,3),(3,6),(6,8)]
    assert [api.shard_bounds(1,r,3) for r in range(3)]==[(0,1),(1,1),(1,1)]
    with pytest.raises(ValueError): api.shard_bounds(-1,0,2)


def test_03(api):
    x=torch.tensor([[1.,2.]],requires_grad=True); y=torch.tensor([3.],requires_grad=True)
    flat=api.flatten_tensors([x,y]); torch.testing.assert_close(flat,torch.tensor([1.,2.,3.])); flat.sum().backward()
    torch.testing.assert_close(x.grad,torch.ones_like(x)); flat.detach()[0]=99; assert x[0,0]==1
    with pytest.raises(ValueError): api.flatten_tensors([x,torch.ones(2,dtype=torch.float64)])


def test_04(api):
    flat=torch.arange(7.); pieces=api.unflatten(flat,[(2,2),(),(2,)])
    assert [p.shape for p in pieces]==[(2,2),(),(2,)]; pieces[1].fill_(99); assert flat[4]==99
    with pytest.raises(ValueError): api.unflatten(flat,[(3,3)])


def test_05(api,seed):
    torch.manual_seed(seed); a=nn.Linear(3,1); b=copy.deepcopy(a); x=torch.randn(5,3); y=torch.randn(5,1)
    expected=(b(x)-y).square().mean(); expected.backward()
    loss=api.accumulate_gradients(a,[(x[:1],y[:1]),(x[1:],y[1:])],lambda p,t:(p-t).square().sum())
    assert loss==pytest.approx(expected.item())
    for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p.grad,q.grad)


def _worker(rank, namespace, directory, mode, backend):
    api=importlib.import_module(namespace); directory=Path(directory)
    if backend=='nccl': torch.cuda.set_device(rank)
    device=torch.device('cuda',rank) if backend=='nccl' else torch.device('cpu')
    dist.init_process_group(backend,init_method=(directory/'rendezvous').as_uri(),rank=rank,world_size=2,timeout=timedelta(seconds=40))
    try:
        if mode=='reduce':
            original=torch.tensor([2.,4.],device=device) if rank==0 else torch.tensor([8.,10.],device=device)
            result=api.weighted_allreduce(original,1 if rank==0 else 3)
            empty=api.weighted_allreduce(original,0 if rank==0 else 2)
            payload=dict(value=result.cpu(),original=original.cpu(),empty=empty.cpu())
        elif mode=='step':
            torch.manual_seed(31); model=nn.Linear(3,1).to(device); opt=torch.optim.SGD(model.parameters(),lr=.03)
            x=torch.arange(15.,device=device).reshape(5,3)/10; y=x.sum(-1,keepdim=True)
            sl=slice(0,1) if rank==0 else slice(1,5)
            loss=api.distributed_step(model,opt,x[sl],y[sl])
            # A second update includes an entirely empty rank.
            sl2=slice(0,0) if rank==0 else slice(None)
            loss2=api.distributed_step(model,opt,x[sl2],y[sl2])
            payload=dict(loss=loss,loss2=loss2,weights={k:v.cpu() for k,v in model.state_dict().items()})
        elif mode=='sharded':
            p=nn.Parameter(torch.arange(5.,device=device)); state={}
            for i in [1,2]:
                p.grad=torch.full_like(p,float(i)); api.sharded_momentum_step([p],state,.1,.9)
            payload=dict(value=p.detach().cpu(),start=state['start'],end=state['end'],momentum=state['momentum'].cpu())
        elif mode=='finite':
            p=nn.Parameter(torch.ones(2,device=device)); p.grad=torch.ones_like(p)
            first=api.collective_finite([p]); p.grad[0]=float('inf') if rank==1 else 1
            second=api.collective_finite([p]); payload=dict(first=first,second=second)
        torch.save(payload,directory/f'{rank}.pt')
    finally: dist.destroy_process_group()


def run_workers(api,tmp_path,mode,backend='gloo'):
    mp.spawn(_worker,args=(api.__name__,str(tmp_path),mode,backend),nprocs=2,join=True)
    return [torch.load(tmp_path/f'{rank}.pt',weights_only=True) for rank in range(2)]

@pytest.mark.distributed
@pytest.mark.timeout(90)
def test_06(api,tmp_path):
    rows=run_workers(api,tmp_path,'reduce')
    for r in rows:
        torch.testing.assert_close(r['value'],torch.tensor([6.5,8.5])); torch.testing.assert_close(r['empty'],torch.tensor([8.,10.]))
    torch.testing.assert_close(rows[0]['original'],torch.tensor([2.,4.]))

@pytest.mark.distributed
@pytest.mark.timeout(90)
def test_07(api,tmp_path):
    rows=run_workers(api,tmp_path,'step'); torch.manual_seed(31); model=nn.Linear(3,1); opt=torch.optim.SGD(model.parameters(),lr=.03)
    x=torch.arange(15.).reshape(5,3)/10; y=x.sum(-1,keepdim=True); losses=[]
    for _ in range(2):
        opt.zero_grad(); loss=(model(x)-y).square().mean(); losses.append(loss.item()); loss.backward(); opt.step()
    for row in rows:
        assert row['loss']==pytest.approx(losses[0]) and row['loss2']==pytest.approx(losses[1])
        for k,v in model.state_dict().items(): torch.testing.assert_close(row['weights'][k],v)


def test_08(api):
    x=torch.arange(7.); r=api.shard_state(x,1,3)
    assert r['start']==3 and r['end']==5; torch.testing.assert_close(r['momentum'],torch.tensor([3.,4.]))
    r['momentum'].zero_(); assert x[3]==3

@pytest.mark.distributed
@pytest.mark.timeout(90)
def test_09(api,tmp_path):
    rows=run_workers(api,tmp_path,'sharded'); p=nn.Parameter(torch.arange(5.)); opt=torch.optim.SGD([p],lr=.1,momentum=.9)
    for i in [1,2]: p.grad=torch.full_like(p,float(i)); opt.step()
    for rank,r in enumerate(rows):
        torch.testing.assert_close(r['value'],p); assert r['momentum'].numel()==(3 if rank==0 else 2)
        torch.testing.assert_close(r['momentum'],torch.full_like(r['momentum'],2.9))


def test_10(api):
    a=nn.Parameter(torch.ones(2)); b=nn.Parameter(torch.ones(1)); a.grad=torch.tensor([4.,8.]); b.grad=torch.tensor([float('inf')])
    assert api.unscale_gradients([a,b],4) is False; torch.testing.assert_close(a.grad,torch.tensor([4.,8.]))
    b.grad=None; assert api.unscale_gradients([a,b],4) is True; torch.testing.assert_close(a.grad,torch.tensor([1.,2.]))
    with pytest.raises(ValueError): api.unscale_gradients([a],0)


def test_11(api,seed):
    torch.manual_seed(seed); x=torch.randn(5,requires_grad=True); y=x.detach().clone().requires_grad_(); calls=[]
    def fn(v): calls.append(1); return v.sin().square()
    out=api.checkpointed(fn,x); out.sum().backward(); y.sin().square().sum().backward()
    torch.testing.assert_close(x.grad,y.grad); assert len(calls)>=2, 'Must recompute intermediates during backward'


def test_12(api):
    shards=[dict(start=3,end=5,momentum=torch.tensor([3.,4.])),dict(start=0,end=3,momentum=torch.tensor([0.,1.,2.]))]
    full=api.consolidate_shards(shards,5); torch.testing.assert_close(full,torch.arange(5.)); full[0]=99; assert shards[1]['momentum'][0]==0
    with pytest.raises(ValueError): api.consolidate_shards(shards[:1],5)
    with pytest.raises(ValueError): api.consolidate_shards([shards[0],shards[1],shards[1]],5)

@pytest.mark.distributed
@pytest.mark.timeout(90)
def test_13(api,tmp_path):
    rows=run_workers(api,tmp_path,'finite'); assert rows==[dict(first=True,second=False)]*2

@pytest.mark.gpu
@pytest.mark.distributed
@pytest.mark.timeout(120)
def test_14(api,tmp_path):
    assert torch.cuda.device_count()>=2, 'Final hardware gate requires TWO NVIDIA GPUs; use scripts/modal_grade.py explicitly. CPU gates 1-13 are separate.'
    rows=run_workers(api,tmp_path,'step',backend='nccl')
    for key in rows[0]['weights']: torch.testing.assert_close(rows[0]['weights'][key],rows[1]['weights'][key])
    torch.manual_seed(31); m=nn.Linear(3,1); opt=torch.optim.SGD(m.parameters(),lr=.03); x=torch.arange(15.).reshape(5,3)/10; y=x.sum(-1,keepdim=True)
    for _ in range(2): opt.zero_grad(); loss=(m(x)-y).square().mean(); loss.backward(); opt.step()
    for k,v in m.state_dict().items(): torch.testing.assert_close(rows[0]['weights'][k],v,atol=2e-5,rtol=2e-4)
'''
rows=[
('Partition examples without duplication','partition_indices','Data parallelism starts with an ownership rule for examples. Padding a partition to equal rank lengths can duplicate examples and subtly change the objective. This course begins with unpadded strided assignment so unequal local counts remain visible. Some ranks can have no work when the world is larger than the batch, which later collective code must handle deliberately.','This partitions one finite epoch. Shuffling is a separate deterministic permutation before partitioning.','What fraction of examples would be duplicated by padding seven items to three equal ranks?'),
('Assign contiguous tensor shards','shard_bounds','Optimizer state is easier to partition as contiguous flat ranges. Quotient and remainder determine balanced ownership, with at most one element difference between ranks. Empty shards are legitimate when tensors are small. An explicit half-open convention prevents overlapping endpoint ownership and makes gather reconstruction straightforward.','The first remainder ranks own the extra elements.','Enumerate ranges for total=2 and world_size=4.'),
('Flatten a communication bucket','flatten_tensors','Collectives have latency overhead as well as bandwidth cost. Combining small tensors into one communication bucket amortizes that overhead. Tensor order becomes part of the bucket layout. A copied flat buffer can still retain autograd connectivity through concatenation; storage ownership and differentiability are different properties.','Require identical dtype and device rather than silently promoting or copying devices.','Why might one enormous bucket delay communication overlap?'),
('Restore shapes through views','unflatten','A flat communication buffer loses tensor boundaries unless the layout is retained. Restoring views rather than copies lets one buffer back several shaped tensors. That makes mutation semantics important: changes through a view affect the shared bucket. Validating total element count prevents a subtly shifted parameter mapping.','Shapes may include scalars and zero-sized dimensions.','Describe a bug caused by restoring parameters in a different order from flattening.'),
('Accumulate uneven microbatches','accumulate_gradients','Gradient accumulation can reproduce a larger batch without holding every activation simultaneously. The accumulated objective must use the same denominator as the large batch. Dividing each microbatch mean by the number of microbatches is wrong when their sizes differ. Backpropagating sums divided by the total example count gives the desired weighting.','Clear gradients once before the microbatch loop. Do not step the optimizer inside it.','Compare microbatch sizes one and four with averaging two mean losses.'),
('Reduce weighted means across processes','weighted_allreduce','An average of rank means is only correct when every rank has the same count. Reduce weighted sums and counts to handle unequal work. Zero-count ranks must still participate in every collective so other ranks can finish. The numerical payload and count must be on a device supported by the selected backend.','The grader launches two real CPU processes with Gloo. These are distributed tests, not a list-based simulation.','Why can one rank skipping an all-reduce hang another rank?'),
('Train with real synchronized gradients','distributed_step','Every rank starts with the same weights, computes local contributions and participates in reductions before applying an identical update. Comparing the result with a single-process global batch is a strong correctness check. It detects normalization errors and inconsistent optimizer ordering. Collectives must occur in the same sequence even when one rank has an empty local batch.','The teaching loss is scalar-output squared error. Production token training needs the same logic with supervised-token counts.','Predict the failure if gradients are divided by world size instead of global count.'),
('Keep only local optimizer state','shard_state','Replicated optimizer moments can consume more memory than model weights. Sharding that state assigns each rank responsibility for only part of it. A tensor slice that still references the full backing storage does not deliver the intended memory saving; clone the owned range. This stage isolates that memory ownership contract before adding communication.','Return only the owned momentum data and its range metadata.','Estimate momentum memory per rank as world size increases.'),
('Update sharded momentum and gather weights','sharded_momentum_step','State ownership and parameter ownership need not be the same. Here every rank has the full parameters and gradients, but retains momentum only for its slice. Each rank updates its portion, then gathers updated weights so replicas agree. This resembles the basic idea of optimizer-state sharding; it does not yet implement full parameter or gradient sharding.','Use equal-size padded all_gather payloads; trim each rank according to its true bounds. Momentum starts at zero.','What memory still scales with total model size on every rank?'),
('Unscale only finite gradients','unscale_gradients','Loss scaling increases gradient magnitudes during low-precision backpropagation, then reverses that scaling before the optimizer sees them. A nonfinite gradient means the update should be skipped, not partially applied. Scan first, mutate second. The next coordination stage ensures all ranks agree about that decision.','This function is a local operation and does not execute collectives.','Why would unscaling twice silently shrink every update?'),
('Trade activation memory for recomputation','checkpointed','Activation checkpointing stores fewer intermediates and recomputes them during backward. It must preserve both gradients and stochastic behavior. This is unrelated to saving training state to disk despite the shared name. A mechanism test counts forward invocations so a plain function call cannot pass merely because its values match.','Use non-reentrant PyTorch checkpointing with RNG preservation.','Measure the time-memory tradeoff for a longer stack of blocks.'),
('Consolidate checkpoint shards','consolidate_shards','Sharded state needs explicit ranges for recovery and changes in world size. Reconstructing a global tensor from validated shards allows repartitioning under a new topology. Missing, overlapping or malformed ranges are errors, not zeros to be guessed. Sorting by range supports arbitrary arrival order while preserving deterministic assembly.','This consolidates momentum tensors in memory. The capstone writes shard files and a manifest around this operation.','How would you validate that shard files belong to the same training step?'),
('Coordinate skipped updates','collective_finite','If one rank skips an optimizer update while another applies it, replicated weights diverge. Nonfinite detection therefore becomes a collective decision. A minimum over boolean-like integer flags makes one failure visible everywhere. All ranks must reach this check before any state update, including momentum and update counters.','None gradients are allowed. The grader injects infinity on only one rank.','What other state must remain unchanged when an update is skipped?'),
('Run the same contract on two GPUs','distributed_step, collective_finite','Correct CPU collectives establish semantics but cannot establish GPU execution. The final gate launches one process per NVIDIA device using NCCL and compares the result to a CPU global-batch reference. Device assignment, collective placement and numerical tolerances become part of the systems contract. Passing a simulated or skipped hardware check would make a false claim.','Requires two real NVIDIA GPUs. Run stages 1–13 locally; invoke the supplied Modal runner explicitly for stage 14. No paid job is launched by importing the runner.','Record GPU model, dtype, world size, throughput and communication time before drawing scaling conclusions.')]
stages=[(*r,['Write down local ownership and the global invariant.','Check unequal counts or a rank with no owned elements.','Ensure all ranks execute compatible collectives in the same order.']) for r in rows]
build('07-distributed','Build a distributed training system','Fourteen stages take you from partitioning to real multi-process gradient synchronization, sharded optimizer state, activation recomputation and a two-GPU NCCL gate. Stages 1–13 run on CPU; the final hardware gate is explicit and cannot pass by skipping.',stages,source,tests,[('PyTorch distributed documentation','https://docs.pytorch.org/docs/stable/distributed.html'),('ZeRO paper','https://arxiv.org/abs/1910.02054'),('PyTorch activation checkpointing','https://docs.pytorch.org/docs/stable/checkpoint.html')])
