# 12 · Accelerator experiments: kernels, precision, and launch overhead

**Time:** 2–4 hours. **Prerequisite:** numerical reference implementations. **Deliverable:** a tested SDPA substitution and an optional CUDA Graph microexperiment. **Hardware:** CPU examples run everywhere; CUDA sections skip when unavailable.

A correct CPU engine teaches scheduling and memory semantics. A fast accelerator engine also needs efficient kernels, layouts, launch scheduling, and transfers. Moving every tensor to a GPU is only the beginning. Tiny operations can be slower when dispatch and synchronization cost more than the math they accelerate.

The core reference makes several host/device crossings: Python ownership checks inspect block IDs, sampling moves logits to CPU, and the runner loops over requests. These are purposeful teaching choices. They prevent this implementation from serving as evidence of GPU throughput or being captured wholesale in a CUDA Graph.

## Replace dense attention with PyTorch SDPA

`scaled_dot_product_attention` provides an implementation boundary for optimized attention. Backend dispatch depends on device, dtype, shapes, masks, and library version. Calling it does not prove that FlashAttention was selected. For this lesson we use an explicit offset mask and manually expand KV heads so the semantics match our reference.

PyTorch's SDPA boolean mask uses `True` for positions that are **allowed**. Other attention APIs can use the opposite convention. Check the exact API rather than carrying a remembered mask meaning from another function. See the [PyTorch 2.7 SDPA documentation](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html).

```python
import torch.nn.functional as F
from nanovllm_course.model import attention
q = torch.randn(3,4,16)
k, v = torch.randn(9,2,16), torch.randn(9,2,16)
start = 6
expanded_k = k.repeat_interleave(2,dim=1)
expanded_v = v.repeat_interleave(2,dim=1)
mask = torch.arange(9)[None,:] <= (start+torch.arange(3))[:,None]
sdpa = F.scaled_dot_product_attention(
    q.transpose(0,1), expanded_k.transpose(0,1), expanded_v.transpose(0,1),
    attn_mask=mask, dropout_p=0.0, is_causal=False,
).transpose(0,1)
expected = attention(q,k,v,query_start=start)
torch.testing.assert_close(sdpa,expected,atol=2e-6,rtol=2e-5)
print("SDPA matches the offset-mask reference")
```

Do not blindly pass `is_causal=True` for rectangular cached attention. The non-square causal alignment must match the intended absolute positions; an explicit mask is the easiest correctness baseline. Also pass `dropout_p=0.0` for inference—SDPA's dropout argument is not automatically disabled merely because your surrounding module is in evaluation mode.

## MPS and CUDA correctness before speed

If an accelerator is available, compare a full forward pass with a CPU copy of the same weights. We keep float32 here to isolate device differences from precision changes. The experiment does not assert exact bitwise equality. Generated greedy IDs can still diverge near a tie even when logits are numerically close.

```python
accelerator = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
if accelerator:
    cpu_model = make_model()
    accelerated = make_model(device=accelerator)
    accelerated.load_state_dict(cpu_model.state_dict())
    with torch.inference_mode():
        cpu_logits,_ = cpu_model([256,1,2,3,4])
        device_logits,_ = accelerated([256,1,2,3,4])
    error = (cpu_logits-device_logits.cpu()).abs().max().item()
    print(accelerator, "maximum float32 logit difference:", error)
    torch.testing.assert_close(device_logits.cpu(),cpu_logits,rtol=2e-4,atol=2e-4)
else:
    print("No accelerator: CPU/SDPA examples still complete the core lab.")
```

Half precision halves the K/V byte count relative to float32, but changes numerical behavior and kernel availability. BF16 has fewer mantissa bits but a wider exponent range than FP16. Quantized weights reduce weight storage; that alone does not reduce an unquantized KV cache. KV quantization requires its own representation, scales, and error validation.

## A CUDA Graph experiment with stable buffers

A CUDA Graph captures a sequence of GPU operations for replay, reducing repeated launch overhead in suitable workloads. Captured tensors need stable memory addresses and the replayed work must obey capture constraints. Dynamic admission and variable sequence lengths require staging buffers, shape buckets, or more elaborate graph management.

This example captures **only a fixed-shape linear layer**, not the engine. Inputs change by copying into the same buffer. A clone saves outputs if they must survive the next replay. The [PyTorch CUDA Graph notes](https://docs.pytorch.org/docs/2.7/notes/cuda.html#cuda-graphs) describe lifetime and stream requirements.

```python
if torch.cuda.is_available():
    linear = torch.nn.Linear(64,64,bias=False).cuda().eval()
    static_input = torch.randn(16,64,device="cuda")
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream), torch.no_grad():
        for _ in range(3):
            linear(static_input)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        static_output = linear(static_input)
    new_input = torch.randn_like(static_input)
    static_input.copy_(new_input)
    graph.replay()
    torch.cuda.synchronize()
    with torch.no_grad():
        expected = linear(new_input)
    torch.testing.assert_close(static_output,expected)
    print("CUDA Graph replay matches eager linear output")
else:
    print("CUDA Graph lab skipped: NVIDIA CUDA hardware required.")
```

## Design the kernel you would build next

For a decode kernel, one useful work unit is a request/query head. Inputs include Q, paged K/V, block tables, sequence lengths, and output storage. Map query heads to KV heads, iterate valid logical blocks, load physical pages, compute scaled scores, and update an online softmax numerator/denominator. Mask unused tail slots before the reduction.

For multi-token prefill, add the query-position dimension and offset causal mask. That is more complex than replacing one matrix multiply. Test on shuffled pages, odd sequence lengths, partial tails, GQA, extreme scores, and mixed lengths. Use the lesson 07 implementation as the oracle.

**Completion criteria for an optional Triton/CUDA kernel:** match logits or attention outputs within stated dtype-specific tolerances; exclude compilation and warmup from steady-state timing; report allocation/copy overhead separately; demonstrate a measured gain for at least one representative workload without claiming it generalizes everywhere. Triton is intentionally not a required course dependency and no custom kernel is shipped as “tested” here.

Next: [13 — Capstone](13_capstone.ipynb).
