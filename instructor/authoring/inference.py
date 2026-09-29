import json
import re
import shutil
from pathlib import Path
from build import ROOT,SOURCE,write
root=ROOT/'06-inference'
if root.exists(): raise RuntimeError('Refusing to overwrite')
root.mkdir()
for name in ['src','grader','instructor','course','scripts','docs','notebooks']:
    shutil.copytree(SOURCE/name,root/name,ignore=shutil.ignore_patterns('__pycache__','.ipynb_checkpoints'))
for name in ['pyproject.toml','.python-version','.gitignore','uv.lock']:
    if (SOURCE/name).exists(): shutil.copy(SOURCE/name,root/name)
(root/'instructor/tests/test_course_cli.py').unlink()
(root/'course/chapters').rename(root/'course/textbook')
write(root/'course_cli/__init__.py','')
shutil.copy(ROOT/'instructor/authoring/runner.py',root/'course_cli/main.py')
# New microstages reuse every original acceptance case, never weakening their assertions.
groups=[
(1,'UTF-8 and special tokens','tokenizer.py','utf8_bytes_and_special_ids','Encode bytes rather than Unicode code points. Establish BOS/EOS identity before constructing model inputs.'),
(1,'Round trips and invalid bytes','tokenizer.py','empty_prompt_and_invalid_utf8 unicode_roundtrip_and_no_input_mutation','Extend the tokenizer to empty text, replacement decoding and ownership preservation. Arbitrary byte sequences need not represent valid UTF-8.'),
(2,'RMS normalization','model.py: RMSNorm','rmsnorm_learned_weight_and_zero','Build learned root-mean-square normalization and make an all-zero vector finite.'),
(2,'Rotary positions','model.py: rope','rope_absolute_positions','Rotate adjacent coordinate pairs at absolute positions. Preserve vector norm and the position-zero identity.'),
(2,'Causal grouped-query attention','model.py: attention','attention_offset_gqa_and_future_mask','Map query heads to shared KV heads and mask using absolute query positions, including chunk offsets.'),
(2,'Assemble the decoder','model.py: TinyLM','dense_model_matches_fixed_checkpoint','Connect embeddings, attention residuals, SwiGLU residuals and the output head. Match the named checkpoint boundary exactly.'),
(2,'Prove causality and gradient flow','model.py: TinyLM.forward','causality_and_varied_sequences dense_gradients_reach_the_parameters','Test that future input changes cannot alter earlier logits and that the dense forward remains trainable.'),
(3,'Greedy generation and EOS','sampling.py: generate_naive','greedy_and_length_zero eos_is_included_and_stops','Use the final prompt logit to produce the first output. Zero output tokens must not execute the model; EOS is included before stopping.'),
(3,'Nucleus sampling','sampling.py: sample','nucleus_crossing_token_and_logit_immutability','Keep the crossing token in top-p filtering. Sampling must not alter the caller’s score tensor.'),
(3,'Own each request random stream','sampling.py: generate_naive','temperature_filters_and_request_rng','Initialize request-local randomness once and advance it only on emissions. Unrelated random draws must not perturb generation.'),
(4,'Append a contiguous KV cache','model.py: TinyLM.forward','cached_chunks_match_dense_without_mutation cached_values_are_actually_consumed','Append only new K/V rows, preserve earlier caller storage, and use modified cached values as the actual source of attention context.'),
(4,'Generate with real cache reuse','sampling.py: generate_cached','cache_reuses_work_not_just_outputs cached_sampling_matches_seeded_baseline','Feed one new token after prefill. The gate counts projection rows, so recomputing a whole prefix cannot masquerade as caching.'),
(5,'Batch isolated contexts','model.py: forward_batch','unequal_sequences_are_independent mixed_cached_lengths_and_empty_batch','Support unequal lengths, unequal cache offsets and an empty batch while keeping requests independent.'),
(5,'Pack the linear work','model.py: forward_batch','projections_are_packed_not_sequential_calls','Project all new rows once per layer. Keep attention segmented and reconstruct outputs in request order.'),
(6,'Allocate pages atomically','cache.py: BlockPool','atomic_allocation_and_refcounts','Separate ownership counts from token counts. Failed allocation must leave every existing owner and free block unchanged.'),
(6,'Map logical slots to physical storage','cache.py: BlockPool','slot_mapping_storage_and_shared_write_guard','Use table lookup for block identity and modulo for the within-block offset. Refuse writes to shared storage.'),
(6,'Execute from real paged KV','model.py: forward_paged_batch','real_paged_execution_with_shuffled_blocks physical_pages_are_the_source_of_truth paged_projection_work_only_new_rows','Connect packed projections with actual page writes and reads. Shuffled tables and modified physical values must affect execution correctly.'),
(7,'Schedule partial prefills','engine.py: Engine','partial_prefill_and_measured_token_budget','Advance the computed frontier without emitting during an incomplete prompt. Budget actual projected tokens, not invented trace counters.'),
(7,'Handle terminal request states','engine.py: Engine','zero_eos_cancellation_and_ids','Make cancellation idempotent and release both active and waiting ownership. Enforce request identity across the engine lifetime.'),
(7,'Preserve stochastic request semantics','engine.py: Engine','scheduling_preserves_per_request_random_stream','Vary batching and scheduling without changing any request’s seeded output stream.'),
(7,'Make progress under capacity pressure','engine.py: Engine','impossible_requests_rejected_and_pressure_makes_progress randomized_arrivals_cancel_and_progress','Reject impossible requests, reserve enough capacity and prove finite progress across arrivals and cancellations.'),
(7,'Emit truthful token events','engine.py: Engine.step','step_events_correspond_to_actual_new_outputs','Return exactly one event per actual new output. Partial work must not pretend to have produced a token.'),
(8,'Retain and evict full prefixes','cache.py: PrefixCache','full_prefix_identity_last_logit_and_active_eviction only_computed_full_blocks_publish_and_clear_preserves_owners','Prefix keys include preceding context. Retain work for final logits and never evict storage still owned by an active request.'),
(8,'Copy a shared tail before writing','cache.py: BlockPool.copy_on_write','copy_on_write_and_failure_atomicity','Allocate and copy a private block before changing ownership. Out-of-memory must preserve both the old mapping and its reference counts.'),
(8,'Reuse prefixes in the engine','engine.py: Engine','prefix_reuse_skips_real_projection_work prefix_pressure_recycles_capacity','A hit must skip real projection work. Repeated requests under memory pressure must still reclaim capacity and finish.'),
(9,'Expose a shared HTTP engine','server.py: create_app','http_contract_validation_and_shared_engine','Validate requests, return distinct IDs and send generation through one engine whose lifetime belongs to the application.'),
(9,'Serve concurrent callers','server.py: create_app','concurrent_clients_use_one_engine','Concurrent requests must join the same scheduler and preserve their own outputs. Private per-route generators miss the batching contract.'),
(9,'Bound pending work and cancellation','server.py: create_app','backpressure_and_cancelled_caller_release_resources','Reject excess pending work and clean up cancelled callers on every exit path. A cancelled request cannot leave an orphaned engine job.'),
(10,'Measure completed device work','metrics.py: benchmark','benchmark_warmup_sample_count_and_validation device_timing_brackets_work_with_synchronization','Separate warmup from samples and bracket accelerator work with synchronization. Timing queued launches alone measures the wrong operation.'),
]
old_hints=json.loads((root/'course/hints.json').read_text()); manifest=[]
old_paths={i:next((root/'grader/stages').glob(f'test_{i:02}_*.py')).relative_to(root).as_posix() for i in range(1,11)}
for number,(old,title,symbols,tests,lesson) in enumerate(groups,1):
    chapter=f'course/chapters/{number:02}_stage.md'; textbook=next((root/'course/textbook').glob(f'{old:02}_*.md')).relative_to(root).as_posix()
    manifest.append(dict(number=number,title=title,symbols=symbols,chapter=chapter,textbook=textbook,tests=[old_paths[old]+'::test_'+t for t in tests.split()],hints=old_hints[str(old)]))
    write(root/chapter,f'''# {number:02}. {title}

{lesson}

## Scope for this gate

Work in `src/toyvllm/{symbols}`. Implement only the behavior described here and in the corresponding technical contract. Earlier cumulative gates must keep passing. This smaller gate is part of original textbook milestone {old}; messages inside the preserved learner scaffold still use those original milestone numbers.

Read [the full technical contract](../textbook/{Path(textbook).name}) for exact shapes, lifecycle rules and examples. `course read {number}` also prints that contract below this stage introduction.

## Verify

```bash
uv run course check {number} --only
uv run course check {number}
uv run course hint {number} --level 1
```

The grader checks observable invariants, including resource ownership and actual execution work where appropriate. Output agreement alone does not establish that an optimization is implemented. Before changing code, draw a minimal trace and predict what the gate should observe. Record the result and one remaining limitation in your experiment journal.

## Explain before moving on

Which invariant would a plausible but incorrect shortcut violate? Give a concrete input exposing that failure. Identify which state belongs to the caller, the request, or the shared engine. Explain why the next stage can safely build on this one.
''')
# Add a real training-to-serving artifact handoff before CUDA-only gates.
loader='''"""Portable artifact loading: learner stage 30."""
import torch
from .model import TinyLM, TinyConfig

def load_export(path, device='cpu'):
    """Load trusted toylm-v1 export with weights_only=True. Validate format and exact byte
    tokenizer metadata, config.vocab_size=258, then construct TinyLM and strict-load state_dict.
    Return eval-mode model on requested device. Preserve caller CPU RNG state.
    Incompatible metadata raises ValueError; incompatible parameter keys raise RuntimeError.
    """
    payload=torch.load(path,weights_only=True,map_location='cpu')
    if payload.get('format')!='toylm-v1' or payload.get('tokenizer')!=dict(kind='utf8-byte',bos_id=256,eos_id=257,vocab_size=258) or payload['config']['vocab_size']!=258:
        raise ValueError('Incompatible artifact')
    with torch.random.fork_rng(devices=[]):
        model=TinyLM(TinyConfig(**payload['config']))
    model.load_state_dict(payload['state_dict'],strict=True)
    return model.to(device).eval()
'''
write(root/'grader/oracle/artifacts.py',loader)
write(root/'src/toyvllm/artifacts.py',loader[:loader.index('    payload=')]+"    raise NotImplementedError('Stage 30: load a trained artifact')\n")
conf=(root/'grader/conftest.py').read_text().replace('"server", "metrics"','"server", "metrics", "artifacts"');write(root/'grader/conftest.py',conf)
write(root/'grader/stages/test_11_artifacts.py','''import torch
import pytest

def test_portable_training_checkpoint(api,oracle,tmp_path):
    payload=dict(format='toylm-v1',config=vars(oracle.cfg),tokenizer=dict(kind='utf8-byte',bos_id=256,eos_id=257,vocab_size=258),state_dict=oracle.state_dict(),provenance={'seed':101})
    path=tmp_path/'model.pt'; torch.save(payload,path); before=torch.get_rng_state().clone()
    model=api.artifacts.load_export(path)
    torch.testing.assert_close(torch.get_rng_state(),before)
    assert not model.training
    torch.testing.assert_close(model([256,1,2])[0],oracle([256,1,2])[0])
    payload['tokenizer']['bos_id']=0; torch.save(payload,path)
    with pytest.raises(ValueError): api.artifacts.load_export(path)
''')
write(root/'course/chapters/30_artifact.md','''# 30. Serve a checkpoint you trained

Implement `src/toyvllm/artifacts.py: load_export`. Training and serving must agree on more than tensor shapes: tokenizer identity, normalization, rotary convention, parameter orientation and architecture all matter. The shared `toylm-v1` format records those choices.

Load the trusted local file using `torch.load(weights_only=True, map_location="cpu")`. Require format `toylm-v1` and tokenizer metadata exactly `{kind: utf8-byte, bos_id: 256, eos_id: 257, vocab_size: 258}`. Configuration vocabulary must also be 258. Construct `TinyLM(TinyConfig(**config))` without altering caller CPU RNG state, strict-load `state_dict`, move to the requested device and return in eval mode. See the scaffold signature for exception conventions.

Run `uv run course check 30`. After completing pretraining, export your trained checkpoint and verify dense logits before turning on cache reuse or prefix sharing. The suite integration check compares the same artifact across both courses. Do not assume an arbitrary external Hugging Face checkpoint shares this layout.

Explain: which metadata could be wrong even when every parameter shape matches? Why is inference mode a caller execution choice whereas eval mode is model state?
''')
manifest.append(dict(number=30,title='Load a trained portable artifact',symbols='artifacts.py: load_export',chapter='course/chapters/30_artifact.md',tests=['grader/stages/test_11_artifacts.py::test_portable_training_checkpoint'],hints=['Validate metadata before constructing a model.','Use strict state-dict loading.','Isolate initialization RNG even though trained weights replace the initial values.']))
for number,title,test in [(31,'Run causal attention on CUDA','cuda_attention_gqa_and_offset'),(32,'Run the complete engine on CUDA','cuda_engine_matches_cpu_and_keeps_kv_on_device')]:
    write(root/f'course/chapters/{number}_cuda.md',f'''# {number}. {title}

This is a real NVIDIA CUDA hardware gate. Implement device-preserving behavior in the same model and cache code already tested on CPU. Tensor factories for positions, masks and physical KV storage must use the model's device. CPU numerical correctness remains a prerequisite.

Run `uv run course check {number}` on a CUDA host or explicitly invoke the supplied Modal runner. Missing CUDA fails the gate; it cannot produce completion by skipping. Gate 31 checks attention with grouped KV heads and nonzero query offsets. Gate 32 runs the complete engine, compares output tokens with a CPU model, and verifies KV storage remains on the GPU.

Read [the measurement contract](../textbook/10_acceleration.md). Record hardware, dtype, model dimensions and repeated synchronized timing samples. A passing gate establishes numerical/device behavior, not a speedup guarantee. The toy gathered-attention implementation is deliberately easier to understand than a production paged kernel.
''')
    manifest.append(dict(number=number,title=title,symbols='model.py, cache.py',chapter=f'course/chapters/{number}_cuda.md',tests=[old_paths[10]+'::test_'+test],hints=old_hints['10']))
write(root/'course/stages.json',json.dumps(manifest,indent=2))
# Keep textbook milestone numbering explicit; remove obsolete command instructions from its heading.
for p in (root/'course/textbook').glob('*.md'):
    content=p.read_text();content=re.sub(r'^\*\*Gate:\*\*.*$', '**Gate:** use the microstage number shown by `uv run course list`.',content,flags=re.M)
    write(p,'> This is a technical textbook chapter using original milestone numbers. The current course has 32 smaller gates; use `course list` for their numbering.\n\n'+content)
write(root/'AGENTS.md',(SOURCE/'AGENTS.md').read_text().replace('through the ten course stages','through the 32 course microstages'))
rows='\n'.join(f"| {s['number']} | [{s['title']}]({s['chapter']}) |" for s in manifest)
write(root/'README.md',f'''# Build a tiny vLLM inference engine

The original course is preserved at `~/Code/ai/nanovllm`. This independent copy preserves its learner code and acceptance cases, and splits ten large milestones into **32 smaller gates**. It adds a portable trained-checkpoint loader shared with course 02. Nothing was moved out of the original directory.

```bash
cd ~/Code/ai/courses/06-inference
uv sync --locked
uv run course list
uv run course read 1
uv run course check 1
```

Implement `src/toyvllm/`. The chapters explain the concepts; the grader supplies expected behavior and tests real execution work. The local grader is inspectable, not a hidden-test service. Complete reference implementations remain under `grader/oracle/` for instructor validation; do not use them as learner implementations.

Stages 1–30 run on CPU. Stages 31–32 require actual NVIDIA CUDA. The optional Modal runner is explicit and incurs provider compute charges when invoked. See [GPU instructions](../GPU.md). Existing scaffold error messages and archived notebooks retain original milestone numbers; follow `course list` for the new gates.

| Stage | Build / verify |
|---|---|
{rows}

The ten [textbook chapters](course/textbook) retain the detailed mathematical and systems contracts. `course read` prints the relevant textbook alongside each microstage. The architecture uses a tiny random-weight decoder, and becomes a trained model when you import course 02's artifact. Paging gathers ordinary PyTorch attention; it is a correctness-oriented teaching implementation, not vLLM performance equivalence.

Read [the capstone](course/CAPSTONE.md) and [artifact contract](../ARTIFACTS.md) to connect training, evaluation and serving. The original optional notebooks remain supporting visual experiments. The default learning path is code, contracts and tests.
''')
# Distinguish the relocated distribution in uv metadata.
p=root/'pyproject.toml';p.write_text(p.read_text().replace('name = "nanovllm-course"','name = "model-course-06-inference"'))
