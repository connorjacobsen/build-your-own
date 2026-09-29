# Public API and input domains

These are scaffold contracts, not implementations. Earlier stages remain dependencies of later stages.

## `chat_tokens`

Return (ids, assistant_mask) lists. Start with BOS=256 (mask false).
    For each {'role': system|user|assistant, 'content':str}, append UTF-8 bytes of
    '<|ROLE|>
' (mask false), UTF-8 content (mask true iff assistant), then EOS=257
    (mask true iff assistant). Reject empty message lists, unknown roles or non-string content.
    Literal role headers are ordinary bytes, not additional vocabulary IDs.
    

Signature: `chat_tokens(messages)`

## `collate`

examples are (ids,assistant_mask) pairs with equal nonzero lengths. Return dict
input_ids long [B,T], labels long [B,T] (original ID if mask true, else -100),
attention_mask bool [B,T]. Right-pad; padding labels=-100, attention=false.
Reject empty batches, empty examples and mismatched lengths. Do not shift labels here.

Signature: `collate(examples, pad_id=0)`

## `masked_loss`

Next-token mean CE: logits [B,T,V] at :-1 predict labels at 1:.
Ignore labels=-100, divide by total supervised tokens across batch. Raise ValueError
when no shifted target is supervised. Return differentiable scalar tensor.

Signature: `masked_loss(logits, labels)`

## `LoRALinear`

Wrap an nn.Linear as base. Require rank>=1 and alpha>0. Freeze base parameters.
A is nn.Parameter [rank,in], Kaiming-uniform initialized; B is zeros [out,rank].
Both match base dtype/device. Forward: base(x) + (x @ A.T @ B.T)*(alpha/rank).
Arbitrary leading input dimensions are supported. Do not modify base weights in forward.

### `LoRALinear.__init__(self, base, rank=2, alpha=2.0)`

Implement as specified by the owning stage.

### `LoRALinear.forward(self, x)`

Implement as specified by the owning stage.

## `inject_lora`

In-place replace exact qualified names of nn.Linear modules with LoRALinear.
Validate every name and type before changing any module. Reject duplicates, missing names,
empty targets and non-linear targets. Return the same model. Other modules stay untouched.

Signature: `inject_lora(model, targets, rank=2, alpha=2.0)`

## `freeze_except_adapters`

Freeze all parameters except actual LoRALinear A/B objects. Return unique trainable
Parameters in model.parameters() order. Reject models without adapters. Names alone
must not cause unrelated parameters named A or B to become trainable.

Signature: `freeze_except_adapters(model)`

## `sft_step`

One optimizer update using model(input_ids)->[B,T,V] logits and masked_loss.
Set train mode, clear old grads, backpropagate, step; return pre-update float loss.
The teaching model interface needs no attention mask because test fixtures are causal
positionwise networks; capstone wrappers must handle padding and document isolation.

Signature: `sft_step(model, optimizer, batch)`

## `merge_lora`

Return a new ordinary nn.Linear on the same device/dtype with W + alpha/rank * B@A
and the same bias. Preserve the input adapter and all its parameters. Merging twice
from the same original gives the same result; do not add the delta into base in place.

Signature: `merge_lora(layer)`

## `sequence_logps`

Return [B] SUM of next-token log probabilities for labels[:,1:] excluding -100.
An unsupervised row contributes zero. Keep gradients; do not length-normalize for DPO.

Signature: `sequence_logps(logits, labels)`

## `dpo_loss`

Mean -logsigmoid(beta*((pc-pr)-(rc-rr))). All arguments [B]; beta>0.
Detach reference terms so the frozen baseline receives no gradient. Stable for large margins.

Signature: `dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1)`

## `dpo_step`

chosen/rejected are collate-style batches of equal batch size. One policy update
from summed response log probabilities. Run reference in eval mode under no_grad;
clear policy gradients, leave reference parameters unchanged, return float DPO loss.

Signature: `dpo_step(policy, reference, optimizer, chosen, rejected, beta=0.1)`

## `adapter_state`

Return dict qualified_name+'.A'/'.B' -> detached CPU cloned tensors for actual
LoRALinear modules only. Reject models with no adapters. No base weights in this state.

Signature: `adapter_state(model)`

## `save_adapter`

Save trusted local torch payload format='toyadapter-v1', base_sha256 (64 hex chars),
state=adapter_state(model), config={module_name:{rank,alpha}}. Return SHA256 file digest.
Reject malformed base digest. The base digest identifies a toylm-v1 artifact externally.

Signature: `save_adapter(path, model, base_sha256)`

## `load_adapter`

Load onto an already injected matching model. Validate format, base digest, config,
all adapter keys and shapes BEFORE modifying anything; mismatch raises ValueError.
Copy tensors in place under no_grad, converting to destination dtype/device. Base stays fixed.

Signature: `load_adapter(path, model, expected_base_sha256)`
