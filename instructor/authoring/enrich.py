from build import ROOT,write
import ast
import json

textbooks={
'01-autograd':'''# Reverse-mode differentiation, from values to learning

A neural network is a composition of functions. Training asks how a scalar objective changes when each parameter changes. Reverse mode answers this efficiently by reusing the intermediate values computed in the forward pass.

For a node y=f(x), the local backward rule maps an upstream gradient g to J_f(x)^T g. This is a vector-Jacobian product. You almost never need to construct the full Jacobian. A matrix product's local rule uses two matrix multiplications, and an elementwise function's local rule uses elementwise multiplication.

## A shared-value example

Let a=2, b=a*a and L=b+b. The forward values are b=4 and L=8. Starting with dL/dL=1, addition contributes one to b through each edge, so dL/db=2. Multiplication contributes a*dL/db through each operand position. Both positions refer to a, yielding dL/da=8.

Two separate mistakes can yield the wrong answer. Deduplicating edges loses contributions, while failing to deduplicate node traversal can execute a callback too early or too often. Graph traversal visits a node once; local derivative rules account for every operand occurrence.

## Broadcasting is a linear map

Adding a bias [D] to activations [B,D] uses each bias element B times. Its backward gradient is the sum over B. For [B,T,D], sum over both leading axes. The correct rule depends on the original shape, not on guessing which output dimensions look like batches.

Reduction reverses the direction: summing an input axis sends one upstream value back to every element on that axis. Together, broadcasting and reduction form a useful pair of adjoint operations.

## Numerical checks

A central finite difference estimates a derivative as (f(x+h)-f(x-h))/(2h). Very large h measures curvature rather than the local derivative; very small h loses precision through cancellation. Float64 and h around 1e-5 are useful starting points for these tiny smooth examples, not universal guarantees. Avoid checking ReLU exactly at its nondifferentiable kink.

The acceptance grader also uses algebraic invariants and a separate PyTorch baseline. Independent checks reduce the chance that the implementation and its test repeat the same mistake. Explain why each test is informative before treating a green result as understanding.

The course engine intentionally omits broadcasting matmul, higher-order derivatives, mutation tracking and memory-efficient graph disposal. These are extensions after you can reason about the existing graph and its ownership rules.
''',
'02-pretraining':'''# What pretraining optimizes

An autoregressive model defines p(x_1,...,x_T)=product_t p(x_t | x_<t). Training minimizes the negative log likelihood of observed targets. Teacher forcing supplies the real preceding tokens; generation instead feeds back sampled tokens. This difference helps explain why a low training loss does not guarantee robust long continuations.

## One shifted sequence

For token IDs [BOS,a,b,EOS], the inputs [BOS,a,b] predict [a,b,EOS]. The logit at b predicts EOS. Appending EOS to a generation prompt would tell the model that the document has already ended; the training and inference boundaries therefore differ deliberately.

With batches of unequal sequences, the token objective is the sum of all target losses divided by the total target count. If one sequence contributes one token with loss 4 and another contributes three tokens with total loss 3, the correct mean is 7/4=1.75. Averaging sequence means gives (4+1)/2=2.5, a different objective.

## Decoder equations

For each layer, normalize the residual stream, project Q/K/V, rotate Q/K by absolute position, compute softmax(QK^T/sqrt(d)+causal_mask)V, project back and add the residual. Then normalize again and add W_down(SiLU(W_gate z) * W_up z). The final normalization and output projection produce raw vocabulary logits.

GQA stores fewer KV heads than query heads. It changes the size of the state reused by inference without requiring every query head to share the same attention distribution. The checkpoint contract fixes head mapping and rotary pairing, so shapes alone are insufficient for compatibility.

## Optimization and state

Adam's moments obey m_t=beta1*m_(t-1)+(1-beta1)*g_t and v_t=beta2*v_(t-1)+(1-beta2)*g_t^2. Bias correction divides by 1-beta^t. AdamW separately scales parameters by 1-lr*weight_decay before the adaptive update. A checkpoint missing moments or the update count cannot faithfully resume that optimizer.

First diagnose whether a tiny fixed batch can be learned. Then separate train, validation and final test data. Use validation for choices such as learning rate and stopping point. Repeatedly selecting on test results turns the test set into another validation set.

This model is intentionally tiny and byte-based. It teaches the mechanics of pretraining rather than reproducing the capabilities or scale of a production foundation model. Exporting its exact architecture to the inference course is a stronger systems exercise than displaying a plausible sentence.
''',
'03-data':'''# Data defines the learning problem

A model optimizes an expectation under its training distribution. Filtering, deduplication, sampling and packing all change that distribution or the conditional prediction problem. Treat them as modeling decisions rather than invisible preprocessing.

## Identity has several meanings

A record ID identifies a source entry. A normalized content hash identifies a chosen equivalence class of text. A file digest identifies exact serialized bytes. Keep these concepts separate: two records can share content but different provenance, and two differently serialized files can contain equivalent records.

Normalization is not neutral. Compatibility forms and casefolding can erase distinctions important in code, identifiers or named entities. The course keeps original text for training and normalized text for matching so you can inspect what each policy does.

## Why transitive grouping matters

Suppose shingle sets are A={ab,bc}, B={bc,cd}, C={cd,de}. Adjacent pairs have Jaccard similarity 1/3 while A and C have zero overlap. At threshold 1/3, all three belong to one connected component. Splitting that component risks leakage through the bridge document B even if A and C do not look similar directly.

Exact quadratic comparisons are reasonable for a tiny learning corpus. At large scale, approximate candidate generation such as locality-sensitive hashing reduces comparisons, but brings missed-pair and threshold behavior that require evaluation of their own.

## Packing changes information flow

Packing two independent documents into one sequence creates two hazards. A target at the second document's beginning can be incorrectly predicted from the first document, and later tokens can attend to unrelated earlier content. A loss mask solves the first problem; a segmented causal attention mask solves the second. Position IDs determine whether positional context restarts. None of those policies is implied by concatenated token IDs alone.

## Sampling and provenance

A source with mixture mass 0.5 supplies half the expected draws even if it has very few unique examples. Record both total draws and unique content counts. A reproducible local RNG fixes one sampling sequence but does not establish that its distribution is appropriate.

Use the experiment journal to inspect rejected content and compare source slices. A filtering rule is only useful in relation to a task, its error costs and its intended population. The output manifest makes experiments repeatable; it does not make a biased or contaminated dataset valid.
''',
'04-finetuning':'''# Change behavior while tracking what changed

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
''',
'05-evaluation':'''# Evaluation is an experimental protocol

A benchmark score is produced by a model, a dataset, a prompt or tokenization protocol, a decoding strategy and a metric. Changing any component can change the result. Exact artifact identities and per-case outputs make the claim inspectable.

## Mechanics versus meaning

A test can prove that continuation scoring uses the correct token offset. It cannot prove that the chosen benchmark represents real user needs. A release report can be internally consistent yet measure the wrong capability. Keep implementation checks and validity arguments separate.

For three scored tokens with probabilities .5, .25 and .125, the summed negative log likelihood is -log(.5*.25*.125). Perplexity is exp(NLL/3), the reciprocal of the geometric mean token probability. It is not the arithmetic mean of reciprocal probabilities, and per-document perplexities should not be averaged without returning to sums and counts.

## Paired comparisons

Suppose two models run on the same cases. Each case gives a score difference d_i. The mean difference estimates the observed improvement. A paired bootstrap samples case indices and averages their differences repeatedly. A confidence interval summarizes variability under resampling assumptions; it does not correct label mistakes, contamination or an unrepresentative task population.

Keeping IDs aligned is essential. If outputs are sorted by completion time, row 20 in one file may not refer to row 20 in another. Missing cases must be explained. Dropping failures or timeouts can turn reliability defects into higher scores.

## Slices and calibration

A system with 90% accuracy on a dominant group and 0% on a rare group can look acceptable in aggregate. Report counts, micro and macro scores, and task-relevant slices. Very small slices need cautious interpretation because uncertainty can be large.

Calibration asks whether confidence matches observed correctness. It differs from accuracy and discrimination. A model can be calibrated by always reporting the base success rate, yet provide little useful differentiation between easy and hard cases. Binned ECE depends on bin definitions, so preserve that protocol.

## Repeated experimentation

Choose metrics and a validation selection rule before comparing many variants. Use a final holdout for the eventual claim, and report the experimentation budget. A single lucky seed or an improvement selected from dozens of attempts should not be presented as a stable effect. In this suite the small acceptance datasets exercise software behavior; capstones ask for evidence beyond those fixtures.
''',
'07-distributed':'''# Preserve the objective while changing where work happens

Distributed training should initially compute the same update as a single-process global batch. Only after establishing that equivalence should you optimize communication, memory or throughput. Many failures are weighting and ownership errors rather than exotic network problems.

## A rank-mean trap

Rank 0 sees one example with gradient 10. Rank 1 sees three examples whose mean gradient is 2. The global mean gradient is (1*10+3*2)/4=4. Averaging rank means gives 6. Dividing by world size is valid only under equal local denominators. For language models, the denominator may be supervised tokens rather than examples.

All ranks must execute collectives in compatible order. A rank with no examples still contributes zeros and participates. If it returns early, another rank may wait forever. The CPU grader launches actual Gloo processes and includes an empty-rank case to expose this failure.

## State placement

Parameters, gradients, optimizer moments and activations are different memory categories. Replicated data parallelism duplicates parameters and optimizer state while splitting examples. The educational sharded momentum step stores only one rank's optimizer slice but still replicates parameters and gradients. Do not label that full parameter sharding.

Flattening gives tensors a shared communication layout. Gather reconstruction must use the same order, shape metadata and balanced ownership ranges. Padding communication payloads is a transport detail; padded values must not become model parameters.

## Precision and recovery

Mixed precision introduces numerical concerns independent of communication. Loss scaling can prevent small gradients from underflowing, but overflow must cause a consistent skipped update across all ranks. Momentum buffers and counters must follow the same decision as weights.

Activation checkpointing trades saved intermediates for recomputation. Disk checkpointing saves persistent training state. Their names are similar but their purposes and correctness tests differ. Sharded disk recovery requires all pieces to refer to the same model and update, with exact range coverage before repartitioning.

## Hardware claims

CPU process tests validate collective semantics. They cannot establish NCCL behavior, GPU memory savings or scaling speedups. The final gate requires two CUDA devices and compares their update against a CPU reference. A benchmark additionally needs synchronized timing, identical workloads, warmup and multiple measurements. Faster execution that changes the objective or drops work is not a valid improvement.
''',
'08-releases':'''# A release is a decision backed by artifacts

Training produces weights. Evaluation produces evidence. A release policy decides whether particular weights should become active. Keeping those responsibilities separate makes failures easier to diagnose and rollback easier to reason about.

## Three identities

A model filename is a mutable label. A byte digest identifies one serialized object. An active pointer states which digest a consumer should use now. Storing immutable objects and changing a small pointer lets you retain old releases without copying or overwriting their weights.

Digests prove equality with known bytes, not correctness or trustworthiness. A model can have a valid hash and still be poorly trained. An evaluation report can refer to the right model and still use contaminated cases. The registry validates identity and consistency while the experiment report argues for scientific validity.

## Evidence gates

An absolute quality floor prevents accepting a candidate merely because it beats a weak baseline. A regression allowance limits backward movement. A minimum case count prevents tiny samples from being treated as sufficient evidence. These are deterministic policy rules, not replacements for uncertainty estimates or task-specific risk analysis.

For example, baseline accuracy .80 and candidate .78 meet a maximum allowed drop of .02 under the stated inclusive boundary. Floating-point comparison needs an explicit tolerance at such boundaries. Comparing a candidate on easier cases to a baseline on harder ones is invalid even if both have accurate aggregate bookkeeping.

## Atomicity and concurrency

Atomic file replacement prevents a reader from observing half-written JSON. It does not prevent two writers from reading the same old pointer and overwriting each other. A lock plus compare-and-swap makes the expected prior value part of the operation. One concurrent promotion wins; the other receives a conflict and must reconsider its evidence.

Rollback uses the same mechanism. It must not blindly restore a pointer based on stale assumptions, because someone else may have promoted a newer model in the meantime. The local implementation demonstrates these semantics without affecting a public service.

## Monitoring closes the loop

Offline quality and online operation can diverge. Canary routing limits exposure while producing comparable observations. Error rates and tail latency answer different operational questions. Small cohorts require patience, but waiting itself can have costs. Record the policy and the observations that triggered a decision.

This course ends with a local release control loop. Authentication, distributed storage, live traffic routing, observability infrastructure and production rollout orchestration remain explicit extensions rather than implied capabilities of the toy registry.
'''
}
capstones={
'01-autograd':('Train and diagnose a nonlinear regressor','Use train_mlp on y=x*x over a bounded interval. Split inputs before tuning width or learning rate. Compare hidden widths 4, 8 and 16 using identical train/validation examples and at least three initialization seeds.','A gradient-check report on a smooth composed expression, train/validation MSE curves, final parameters and an explanation of one failed learning-rate choice.','Compare gradients of your network with an independently constructed PyTorch network loaded with identical arrays. Do not import the instructor engine.','Add a nonlinear classification task; implement additional operations only after documenting their derivatives and shape domains.'),
'02-pretraining':('Train a model you can later serve','Consume the JSONL training split from course 03. Start with a tiny repeated sequence to diagnose the loop, then train on a small self-authored corpus. Choose configuration and learning rate using validation loss. Export toylm-v1 and save a resume checkpoint separately.','Dataset digest, configuration, parameter count, training tokens, loss curves, held-out loss, checkpoint-resume equivalence and a portable export digest.','Interrupt training at an update boundary and compare the next update with uninterrupted execution. Load the exported artifact in course 06 and compare dense logits.','Add a budgeted scaling experiment over model width and tokens; extrapolation is an estimate, not a guarantee.'),
'03-data':('Prepare and audit a small corpus','Create at least 50 self-authored records including exact duplicates, lightly edited copies, repeated spam, Unicode variants and reserved evaluation examples. Predict each filter outcome before running preparation.','Raw records, a rejection audit, normalized-content IDs, connected duplicate groups, split assignments, token counts and toydata-v1 manifest.','Verify no duplicate component crosses splits, reserved exact content is absent, and rerunning into a new directory produces the same JSONL bytes.','Compare two normalization or shingle policies and manually inspect disagreements. Report which source types each policy disproportionately removes.'),
'04-finetuning':('Adapt the checkpoint from pretraining','Load a trusted course 02 checkpoint and wrap it in the batched interface described in ARTIFACTS.md. Build coherent instruction examples, apply LoRA to selected linears, run supervised tuning, then a small DPO experiment with a fixed reference.','Base and adapter digests, chat template, masks, trainable parameter count, optimization budget, held-out task and regression scores, and merged-model export.','Check frozen base tensors before and after training, merged/unmerged logits, wrong-base adapter rejection, and preferred-margin direction on a controlled pair.','Compare output-head adaptation with attention projection adaptation using matched budgets. Explain results with evidence rather than assuming more targets are always better.'),
'05-evaluation':('Compare models without changing the question','Define a small task with stable case IDs and multiple acceptable references where needed. Compare the base and adapted models on identical held-out cases. Select the normalization and decoding policy before collecting final results.','Per-case predictions and scores, model/dataset digests, paired interval, slice counts, failure examples and a canonical toyeval-v1 report.','Shuffle output order and confirm identical comparisons; remove a case and confirm rejection; deliberately change continuation offsets and observe a regression.','Add a second metric and explain where it disagrees with exact match. Treat any model-based judge as another fallible measurement system.'),
'06-inference':('Serve the model you trained','Complete CPU gates, load a toylm-v1 checkpoint from course 02 or merged course 04 output, then compare dense, cached and paged generation on the same requests. Add concurrent clients and cancellation.','Checkpoint digest, numerical equivalence, projected-row counts, page ownership traces, latency samples and a cancellation/resource-cleanup demonstration.','Run the cumulative CPU gate through stage 30. For GPU claims run stages 31–32 on real CUDA. Compare identical outputs and workload under each execution path.','Implement a specialized paged kernel or quantization as a separate experiment after correctness, and report error and performance rather than only speed.'),
'07-distributed':('Recover and scale a tiny training update','Match a global-batch update across two CPU ranks, then save each rank’s momentum shard plus step, range, dtype and artifact identity metadata. Reconstruct and repartition state for a different world size in a controlled experiment.','Single-process/distributed parameter comparisons, local state sizes, checkpoint manifest, missing-shard failure, and a resumed update comparison.','Inject a nonfinite gradient on one rank and verify a unanimous skipped update. Check that no rank changes weights or optimizer state when the update is skipped.','Run the two-GPU gate, then measure communication and compute separately. The supplied stages teach components; assembling a crash-recovery orchestrator is a capstone integration task.'),
'08-releases':('Release and roll back a model locally','Use actual model artifacts and toyeval-v1 reports from earlier courses. Declare a quality floor, regression allowance, minimum evidence count and monitoring policy before trying candidates. Simulate a canary workload.','Verified bundle manifest, object digests, gate reasons, active-pointer history, concurrent-promotion conflict and rollback trace.','Reject a report for another model, corrupt one byte, inject a latency spike, and race two promotions. Confirm the intended prior state survives each failed operation.','Connect a local server’s model selection to the pointer while preserving in-flight request ownership. Live deployment is a separate explicitly authorized project.')
}
for course in sorted(ROOT.glob('[0-9][0-9]-*')):
    if course.name in textbooks:
        write(course/'course/TEXTBOOK.md',textbooks[course.name])
        # Make foundational prose part of the default start rather than an unlinked extra.
        path=course/'README.md'; text=path.read_text().replace('## Start','Read [the conceptual foundation](course/TEXTBOOK.md) and [API reference](course/API.md), then work stage by stage.\n\n## Start');write(path,text)
        learner=course/'src/learner/api.py'; tree=ast.parse(learner.read_text()); sections=[]
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                sections.append(f'## `{node.name}`\n\n'+(ast.get_docstring(node) or 'See the stage-specific architecture and method contract.'))
                if isinstance(node,ast.ClassDef):
                    for method in node.body:
                        if isinstance(method,ast.FunctionDef): sections.append(f'### `{node.name}.{method.name}({ast.unparse(method.args)})`\n\n'+(ast.get_docstring(method) or 'Implement as specified by the owning stage.'))
                else: sections[-1]+='\n\nSignature: `'+node.name+'('+ast.unparse(node.args)+')`'
        write(course/'course/API.md','# Public API and input domains\n\nThese are scaffold contracts, not implementations. Earlier stages remain dependencies of later stages.\n\n'+'\n\n'.join(sections))
    title,task,evidence,checks,extension=capstones[course.name]
    write(course/'course/CAPSTONE.md',f'''# Capstone: {title}

## Build

{task}

Complete the relevant acceptance stages first. Keep implementation changes in learner-owned source and experiment drivers. Do not modify acceptance tests to fit a preferred result.

## Evidence to produce

{evidence}

Record hypothesis, configuration, seed, input identity, measurements and limitations in EXPERIMENTS.md. A failed hypothesis is a valid learning result if the experiment and explanation are sound. Separate debugging examples from validation choices and final held-out evidence.

## Integration checks

{checks}

These checks complement the automated stage gates. The capstone is an experimental report, not an automatic certification of model quality. Use the suite [artifact contract](../../ARTIFACTS.md) to connect outputs with other courses and the [lifecycle driver](../../CAPSTONE.md) for a provided smoke test.

## Extension after the core project

{extension}
''')
