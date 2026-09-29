from build import build
source = '''"""Instructor reference: a deliberately small NumPy reverse-mode engine."""
import numpy as np

class Tensor:
    """Float64 tensor with owned data, zero grad, tuple parents and a no-argument _backward callback.
    Constructor accepts data, parents=(), backward=None. Operations attach callbacks which
    add vector-Jacobian products into parent.grad. No torch/autograd delegation is allowed.
    """
    def __init__(self, data, parents=(), backward=None):
        self.data = np.array(data, dtype=np.float64, copy=True)
        self.grad = np.zeros_like(self.data)
        self.parents = tuple(parents)
        self._backward = backward if backward is not None else lambda: None

    def backward(self, grad=None):
        """Reset reachable gradients, seed this output, propagate in reverse topological order.
        Without grad the output must have exactly one element. Explicit grad must match shape.
        Repeated backward calls replace the previous gradients, rather than accumulate them.
        """
        if grad is None:
            if self.data.size != 1:
                raise ValueError('Non-scalar output requires a seed')
            grad = np.ones_like(self.data)
        grad = np.asarray(grad, dtype=np.float64)
        if grad.shape != self.data.shape:
            raise ValueError('Seed shape mismatch')
        nodes = topological(self)
        for node in nodes:
            node.grad = np.zeros_like(node.data)
        self.grad = grad.copy()
        for node in reversed(nodes):
            node._backward()

def unbroadcast(grad, shape):
    """Sum a broadcast output gradient back to an original shape; return an ndarray of that shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)

def topological(root):
    """Return each reachable Tensor once, parents before children; shared edges retain contributions."""
    seen, ordered = set(), []
    def visit(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        for parent in node.parents:
            visit(parent)
        ordered.append(node)
    visit(root)
    return ordered

def add(a, b):
    """Elementwise Tensor addition with NumPy broadcasting and accumulated parent gradients."""
    out = Tensor(a.data+b.data, (a,b))
    def backward():
        a.grad += unbroadcast(out.grad, a.data.shape)
        b.grad += unbroadcast(out.grad, b.data.shape)
    out._backward = backward
    return out

def multiply(a, b):
    """Elementwise Tensor multiplication; support broadcasting and the same Tensor as both operands."""
    out = Tensor(a.data*b.data, (a,b))
    def backward():
        a.grad += unbroadcast(out.grad*b.data, a.data.shape)
        b.grad += unbroadcast(out.grad*a.data, b.data.shape)
    out._backward = backward
    return out

def matmul(a, b):
    """Rank-two matrix multiplication only. Reject non-matrices with ValueError."""
    if a.data.ndim != 2 or b.data.ndim != 2:
        raise ValueError('Only matrices are supported')
    out = Tensor(a.data@b.data, (a,b))
    def backward():
        a.grad += out.grad@b.data.T
        b.grad += a.data.T@out.grad
    out._backward = backward
    return out

def summation(x, axis=None, keepdims=False):
    """Sum all elements or one integer axis (negative axes allowed), with the corresponding VJP."""
    out = Tensor(x.data.sum(axis=axis, keepdims=keepdims),(x,))
    def backward():
        g=out.grad
        if axis is not None and not keepdims:
            g=np.expand_dims(g,axis)
        x.grad += np.broadcast_to(g,x.data.shape)
    out._backward=backward
    return out

def relu(x):
    """Elementwise max(x,0); derivative is zero at and below zero."""
    out=Tensor(np.maximum(x.data,0),(x,))
    out._backward=lambda: None
    def backward():
        x.grad += out.grad*(x.data>0)
    out._backward=backward
    return out

def exp(x):
    """Elementwise exponential with a reverse-mode derivative."""
    out=Tensor(np.exp(x.data),(x,))
    def backward():
        x.grad += out.grad*out.data
    out._backward=backward
    return out

def log(x):
    """Natural logarithm; reject any nonpositive input with ValueError."""
    if np.any(x.data<=0):
        raise ValueError('Log requires positive values')
    out=Tensor(np.log(x.data),(x,))
    def backward():
        x.grad += out.grad/x.data
    out._backward=backward
    return out

def mse(pred, target):
    """Mean squared error across every broadcast result element, composed from this engine's ops."""
    delta=add(pred,multiply(target,Tensor(-1.0)))
    square=multiply(delta,delta)
    return multiply(summation(square),Tensor(1.0/square.data.size))

def cross_entropy(logits, labels):
    """Stable mean cross-entropy for [N,C] logits and integer [N] labels. Return scalar Tensor.
    Reject wrong label shape, empty N or out-of-range labels. Backward must remain finite
    for logits of magnitude 1000. Compute derivatives yourself; do not delegate to torch.
    """
    y=np.asarray(labels)
    if logits.data.ndim!=2 or len(logits.data)==0 or y.shape!=(len(logits.data),) or y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>=logits.data.shape[1]):
        raise ValueError('Invalid labels or logits')
    shifted=logits.data-logits.data.max(axis=-1,keepdims=True)
    z=np.exp(shifted).sum(axis=-1,keepdims=True)
    out=Tensor((np.log(z[:,0])-shifted[np.arange(len(y)),y]).mean(),(logits,))
    def backward():
        g=np.exp(shifted)/z
        g[np.arange(len(y)),y]-=1
        logits.grad += out.grad*g/len(y)
    out._backward=backward
    return out

def sgd(parameters, lr):
    """Update each unique Tensor exactly once in place using data -= lr*grad; then zero grad.
    Reject negative lr. Parameters may contain repeated references to shared weights.
    """
    if lr<0:
        raise ValueError('Negative learning rate')
    seen=set()
    for p in parameters:
        if id(p) not in seen:
            p.data -= lr*p.grad
            p.grad.fill(0)
            seen.add(id(p))

def train_mlp(x, y, hidden=8, steps=300, lr=0.03, seed=0):
    """Fit [N,D] x to [N,O] y with Linear-ReLU-Linear using only this engine.
    Initialize w1,w2 using local NumPy normal(0,.3), biases zero. Return (parameters, losses),
    parameters ordered [w1,b1,w2,b2], losses containing one PRE-update MSE per step.
    Shapes are [D,H], [H], [H,O], [O]. Do not mutate inputs or use global NumPy RNG.
    """
    rng=np.random.default_rng(seed)
    w1,b1=Tensor(rng.normal(0,.3,(x.shape[1],hidden))),Tensor(np.zeros(hidden))
    w2,b2=Tensor(rng.normal(0,.3,(hidden,y.shape[1]))),Tensor(np.zeros(y.shape[1]))
    params=[w1,b1,w2,b2]
    losses=[]
    for _ in range(steps):
        pred=add(matmul(relu(add(matmul(Tensor(x),w1),b1)),w2),b2)
        loss=mse(pred,Tensor(y))
        losses.append(float(loss.data))
        loss.backward()
        sgd(params,lr)
    return params,losses
'''
tests='''import numpy as np
import pytest
import torch


def test_01(api):
    a=np.array([1.,2.]); t=api.Tensor(a); a[0]=99
    assert t.data.dtype==np.float64 and t.data[0]==1
    np.testing.assert_array_equal(t.grad,[0,0]); assert t.parents==()
    assert api.Tensor(2).data.shape==()


def test_02(api,seed):
    g=np.random.default_rng(seed).normal(size=(2,3,4))
    np.testing.assert_allclose(api.unbroadcast(g,(1,4)),g.sum((0,1),keepdims=False)[None,:])
    np.testing.assert_allclose(api.unbroadcast(g,()),g.sum())
    np.testing.assert_allclose(api.unbroadcast(g,(2,3,4)),g)


def test_03(api):
    a=api.Tensor(1); b=api.Tensor(2,(a,)); c=api.Tensor(3,(a,b)); d=api.Tensor(4,(c,c))
    nodes=api.topological(d)
    assert len(nodes)==4 and len({id(n) for n in nodes})==4
    assert nodes.index(a)<nodes.index(b)<nodes.index(c)<nodes.index(d)


def test_04(api):
    a=api.Tensor([2.,3.]); out=api.Tensor([4.,9.],(a,))
    def vjp(): a.grad.__iadd__(out.grad*2*a.data)
    out._backward=vjp
    out.backward(np.array([3.,4.])); np.testing.assert_allclose(a.grad,[12,24])
    out.backward(np.array([3.,4.])); np.testing.assert_allclose(a.grad,[12,24])
    with pytest.raises(ValueError): out.backward()
    with pytest.raises(ValueError): out.backward(np.ones(1))


def test_05(api,seed):
    rng=np.random.default_rng(seed); a=api.Tensor(rng.normal(size=(3,2))); b=api.Tensor([1.,2.])
    out=api.add(a,b); out.backward(np.ones((3,2)))
    np.testing.assert_allclose(out.data,a.data+b.data); np.testing.assert_allclose(b.grad,[3,3])
    api.add(a,a).backward(np.ones((3,2))); np.testing.assert_allclose(a.grad,2)


def test_06(api,seed):
    a=api.Tensor(np.random.default_rng(seed).normal(size=(3,2))); b=api.Tensor([2.,-1.])
    api.multiply(a,b).backward(np.ones((3,2)))
    np.testing.assert_allclose(a.grad,np.broadcast_to(b.data,(3,2))); np.testing.assert_allclose(b.grad,a.data.sum(0))
    api.multiply(a,a).backward(np.ones((3,2))); np.testing.assert_allclose(a.grad,2*a.data)


def test_07(api,seed):
    rng=np.random.default_rng(seed); a=api.Tensor(rng.normal(size=(3,4))); b=api.Tensor(rng.normal(size=(4,2)))
    g=rng.normal(size=(3,2)); out=api.matmul(a,b); out.backward(g)
    np.testing.assert_allclose(out.data,a.data@b.data); np.testing.assert_allclose(a.grad,g@b.data.T); np.testing.assert_allclose(b.grad,a.data.T@g)
    with pytest.raises(ValueError): api.matmul(api.Tensor([1,2]),b)


def test_08(api,seed):
    x=api.Tensor(np.random.default_rng(seed).normal(size=(2,3)))
    for axis,keep in [(None,False),(0,False),(-1,True)]:
        y=api.summation(x,axis,keep); np.testing.assert_allclose(y.data,x.data.sum(axis=axis,keepdims=keep))
        y.backward(np.ones_like(y.data)); np.testing.assert_allclose(x.grad,1)


def test_09(api):
    x=api.Tensor([-2.,0.,3.]); y=api.relu(x); y.backward(np.array([1.,2.,4.]))
    np.testing.assert_allclose(y.data,[0,0,3]); np.testing.assert_allclose(x.grad,[0,0,4])


def test_10(api,seed):
    x=api.Tensor(np.random.default_rng(seed).uniform(.2,2,(2,3)))
    y=api.log(api.exp(x)); y.backward(np.full((2,3),2.))
    np.testing.assert_allclose(y.data,x.data); np.testing.assert_allclose(x.grad,2)
    with pytest.raises(ValueError): api.log(api.Tensor([0.]))
    z=api.log(x); z.backward(np.ones((2,3))); np.testing.assert_allclose(x.grad,1/x.data)


def test_11(api,seed):
    rng=np.random.default_rng(seed); x=api.Tensor(rng.normal(size=(3,2))); y=api.Tensor(rng.normal(size=(3,2)))
    loss=api.mse(x,y); loss.backward()
    np.testing.assert_allclose(loss.data,((x.data-y.data)**2).mean())
    np.testing.assert_allclose(x.grad,2*(x.data-y.data)/6); np.testing.assert_allclose(y.grad,-x.grad)


def test_12(api,seed):
    data=np.random.default_rng(seed).normal(size=(4,5))*1000; labels=np.array([0,4,1,3])
    x=api.Tensor(data); loss=api.cross_entropy(x,labels); loss.backward()
    ref=torch.tensor(data,requires_grad=True); expected=torch.nn.functional.cross_entropy(ref,torch.tensor(labels)); expected.backward()
    np.testing.assert_allclose(loss.data,expected.item(),atol=1e-9); np.testing.assert_allclose(x.grad,ref.grad.numpy(),atol=1e-9)
    with pytest.raises(ValueError): api.cross_entropy(x,np.array([0,1,2,5]))


def test_13(api):
    p=api.Tensor([2.,4.]); p.grad[:]=[1,2]; api.sgd([p,p],.1)
    np.testing.assert_allclose(p.data,[1.9,3.8]); np.testing.assert_allclose(p.grad,0)
    with pytest.raises(ValueError): api.sgd([p],-.1)


def test_14(api,seed):
    x=np.linspace(-1,1,20).reshape(-1,1); y=2*x+.5; old=x.copy()
    p,loss=api.train_mlp(x,y,hidden=12,steps=400,lr=.05,seed=seed)
    assert len(loss)==400 and np.isfinite(loss).all() and loss[-1]<.03 and loss[-1]<loss[0]*.1
    assert [v.data.shape for v in p]==[(1,12),(12,),(12,1),(1,)]
    np.testing.assert_array_equal(x,old)
    _,again=api.train_mlp(x,y,hidden=12,steps=400,lr=.05,seed=seed)
    np.testing.assert_array_equal(loss,again)
    # Mechanism guard: the training loop must exercise the learner's backward and optimizer.
    calls=[]; original=api.sgd
    def tracked(*a,**kw): calls.append(1); return original(*a,**kw)
    api.sgd=tracked
    try: api.train_mlp(x,y,steps=4,seed=seed)
    finally: api.sgd=original
    assert len(calls)==4
'''
info=[
('Own your tensors','Tensor','A tensor has numerical storage and participation in a graph. Those are different responsibilities. Owning the input array prevents a caller from silently changing a forward value after the graph was constructed. Gradient storage must have the same shape even for a scalar, whose shape is an empty tuple. This engine uses float64 so numerical derivative checks are easier to interpret.','Implement the constructor now; leave backward for stage 4. The callback is a no-argument callable. Parents retain identity, including repeated references.','Why would sharing the caller’s array make an otherwise correct backward pass wrong?', ['Separate values from derivatives.','A scalar is a zero-dimensional array, not a length-one vector.','Copy the data and allocate independent zero gradient storage.']),
('Undo broadcasting','unbroadcast','Broadcasting duplicates a value conceptually across output positions. The reverse operation adds the influence of every use back into the original input. Removing extra leading dimensions and reducing singleton axes are distinct steps. A bias shaped [D] added to [B,T,D] receives contributions from both B and T; averaging those contributions would change the derivative.','Valid inputs are NumPy arrays and shapes that broadcast to their shape. Sum rather than average.','Trace a [1,4] bias expanded into [2,3,4]. How many contributions reach each bias element?', ['Reverse replication with addition.','Handle extra leading axes first.','Retain singleton dimensions when reducing axes from the original shape.']),
('Order a shared graph','topological','A graph is not necessarily a tree. One parameter can influence two branches which later join. Visiting it twice can run its local derivative before all incoming contributions have arrived. A parents-before-children order, reversed at differentiation time, resolves this dependency. Node identity matters: equal numerical values can still be different parameters.','Graphs are acyclic; cycle detection is an optional extension. Return actual Tensor objects.','Draw a diamond graph and predict the error from a tree-only traversal.', ['Track node identity.','Visit parents before appending their child.','Repeated edges do not mean repeated traversal.']),
('Run reverse mode','Tensor.backward','Reverse mode computes a vector-Jacobian product, not an entire Jacobian. The seed specifies which weighted combination of outputs is being differentiated. A scalar loss conventionally has seed one. Every node must receive all contributions before its callback executes. This course deliberately resets reachable gradients at each backward call; optimizer accumulation is a separate policy.','Implement backward according to its scaffold docstring. Reject missing seeds for multi-element outputs and mismatched seed shapes.','For y=[x²,3x], what derivative results from seed [2,4]?', ['Initialize gradients before callbacks run.','The output seed is itself a gradient.','Reverse the topological list exactly once.']),
('Differentiate addition','add','Addition preserves each input’s influence. Broadcasting changes where that influence is collected. When both inputs are the same object, the same storage must receive two contributions. Assigning a parent gradient rather than adding to it silently breaks both shared operands and branched graphs.','Return a new Tensor connected to both parents. Preserve input data.','Explain why add(x,x) has derivative two rather than one.', ['Use unbroadcast for each parent.','Callbacks add contributions.','Do not deduplicate operand contributions.']),
('Differentiate multiplication','multiply','The derivative with respect to one factor depends on the other factor’s forward value. This local rule composes with arbitrary upstream weights through multiplication. A square is an especially useful adversarial example: both operand positions refer to one object, so both local derivatives must be accumulated.','Support ordinary NumPy broadcasting; operands are Tensor instances.','Check a broadcast product with finite differences and a nonuniform output seed.', ['Hold the other operand fixed.','Multiply by the upstream gradient first.','Reduce to the parent shape after multiplying.']),
('Differentiate matrix products','matmul','Matrix multiplication combines many scalar products. For A[M,K] and B[K,N], an upstream gradient G[M,N] implies gradients shaped exactly like A and B. Deriving one entry by its summation index is safer than memorizing transposes. Restricting the first implementation to matrices makes the algebra visible before adding batched broadcasting.','Reject vectors and rank-three inputs. No torch autograd or numerical differentiation inside the implementation.','Derive dL/dA[i,k] as a sum over n.', ['Write the index equation for the forward result.','Check that the inner dimension contracts.','The two gradients use the opposite operand transposed.']),
('Differentiate reductions','summation','A sum sends the upstream derivative equally to each included input element. The output shape loses an axis unless keepdims is enabled. Backward must recover that axis before broadcasting. This is the reverse of forward broadcasting: here reduction creates a smaller output, and its derivative expands.','Support axis=None or a single integer, including negative axes.','Compare sum and mean derivatives on batches of different sizes.', ['Remember which axis was reduced.','Restore the axis only when it was removed.','Broadcast the upstream values to the input shape.']),
('Introduce a nonlinearity','relu','Without nonlinearities, composing linear layers produces another linear map. ReLU selects an active subset of coordinates. Its derivative is undefined at zero mathematically; implementations need an explicit convention. We choose zero there, and the grader checks it. Finite differences near the kink require care and should not be mistaken for smooth-function checks.','Derivative is zero for x<=0 and one for x>0, multiplied by the upstream seed.','Why can a ReLU unit stop learning if every input is negative?', ['Save or recover the active mask.','The output and derivative use related but different values.','Zero belongs to the inactive side by contract.']),
('Compose exponential and logarithm','exp, log','Exponentials and logarithms appear in normalized probabilities and likelihood losses. Their derivatives are simple, but their numerical domains matter. Taking log at zero is not an innocuous edge case. Checking log(exp(x)) on moderate inputs is a useful composition test, while extreme inputs teach why a stable fused loss is needed later.','Raise ValueError for any nonpositive logarithm input. Inputs to exp in this stage are moderate.','Why can mathematically equivalent log(exp(1000)) overflow?', ['Use the output of exp in its local derivative.','The log derivative uses the input reciprocal.','Multiply every local derivative by the upstream gradient.']),
('Construct a loss from operations','mse','A training objective reduces many predictions to a scalar. Mean squared error divides by the number of output elements, not merely the batch count. Constructing it from earlier operations tests whether graph composition actually works. Both prediction and target tensors remain differentiable in this engine, even though a normal training target is treated as fixed.','Compose existing operations; do not write a separate shortcut gradient.','How does changing mean reduction to sum affect the appropriate learning rate?', ['Build subtraction from addition and multiplication.','Square a shared intermediate.','Normalize by the number of elements after broadcasting.']),
('Stabilize cross-entropy','cross_entropy','Softmax probabilities can underflow and exponentials can overflow even when the final loss is finite. Subtracting the row maximum preserves normalized probabilities. Computing log probabilities through log-sum-exp avoids taking the log of a rounded zero. The gradient has a compact structure: predicted probability minus the target indicator, averaged across examples.','Labels are integer class indices. Large-magnitude logits must give finite losses and gradients.','Add a constant 1000 to every class score in each row. What should change?', ['Softmax ignores a common shift.','Compute the normalizer after subtracting the row maximum.','Each gradient row should sum approximately to zero.']),
('Update shared parameters once','sgd','Differentiation decides how the objective changes; optimization decides how parameters move. Parameter sharing introduces a new identity issue: several module paths can refer to one weight. The graph must accumulate every use, but the optimizer must update the unique weight once. This stage also makes gradient lifecycle explicit by clearing gradients after the step.','Mutate parameters in place; reject negative learning rates. Zero is allowed.','Why do graph edge deduplication and optimizer parameter deduplication have different rules?', ['Use object identity for optimizer membership.','Apply the gradient before clearing it.','A repeated parameter is one storage location.']),
('Train a composed network','train_mlp','A working training loop integrates ownership, graph construction, differentiation and optimization. Each iteration builds a fresh graph from current parameters. Retaining old graphs unnecessarily wastes memory and can expose stale values. A local random generator makes initialization reproducible without depending on earlier unrelated experiments. The small regression task verifies mechanics, not general language understanding.','Follow the exact initialization and return contract. Use the engine functions and sgd; external autodiff is forbidden.','Compare train and held-out MSE for a linear target and then a curved target. Predict the effect of hidden width.', ['Construct the forward expression from earlier operations.','Record the pre-update scalar loss.','Differentiate and update once per step.'])]
root=build('01-autograd','Build a tiny autodiff framework','Build the differentiation engine underneath a neural network, using NumPy rather than delegating gradients to PyTorch. Fourteen small stages culminate in a trainable network. Prerequisites: Python, arrays, derivatives and matrix multiplication.',info,source,tests,[('Deep Learning Systems','https://dlsyscourse.org/'),('Karpathy: Zero to Hero','https://karpathy.ai/zero-to-hero.html')])
