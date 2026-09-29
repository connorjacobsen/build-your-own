# Change behavior while tracking what changed

Supervised fine-tuning optimizes selected target tokens under a task-specific input format. The serialization, supervised mask and training distribution define the behavior being rewarded. Adapting a model can teach new facts as well as formatting or behavior; whether those facts are reliably retained and retrieved is an empirical question.

## Supervision follows targets

If an input sequence is [BOS,user_tokens,assistant_header,a,b,EOS], the target-position mask marks a, b and EOS. The loss uses the logit immediately before each marked target. Prompt and header tokens can remain visible as context without contributing their own target loss. Right-padding creates neither useful context nor supervised targets.

A model-specific chat template is part of the interface. Our byte template is a deliberately explicit teaching convention. Applying it to a downloaded model that expects different role markers can cause failure even with a correct training loop.

## Low-rank updates

For base weight W[out,in], choose A[r,in] and B[out,r]. The adapted function is xW^T + (alpha/r)xA^TB^T. Its extra trainable parameter count is r*(in+out), compared with in*out for a full update. Savings are strongest when r is small relative to both dimensions.

Initialize A with small nonzero values and B with zeros. The starting function equals the base model. On the first update, B can receive a gradient through A, while A's gradient is initially zero because B is zero. This asymmetry is useful; setting both factors to zero would block learning in both.

Merging forms W' = W+(alpha/r)BA. Verify output equivalence before interpreting a speed measurement. Keep the original base artifact intact so other adapters and rollback remain meaningful.

## Relative preference optimization

Let s_theta be the summed chosen log probability minus the summed rejected log probability. Let s_ref be that same margin under a fixed reference. DPO minimizes -log sigmoid(beta*(s_theta-s_ref)). If policy and reference are identical, the margin difference is zero and loss is log(2). The gradient favors increasing the policy's chosen-relative margin.

Preference labels can be noisy, incomplete or misaligned with your actual goal. A lower preference loss is not automatically better task performance. Keep held-out task and regression evaluations, examine changes across slices, and separate improvements caused by data changes from those caused by the objective.
