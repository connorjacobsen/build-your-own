from pathlib import Path
import json
import shutil
from build import ROOT, write
# Improve the shared reader to print complete technical context for split inference gates.
p=ROOT/'instructor/authoring/runner.py'
s=p.read_text().replace("elif args.command=='read': print((ROOT/stages[args.stage-1]['chapter']).read_text())", "elif args.command=='read':\n        stage=stages[args.stage-1]\n        print((ROOT/stage['chapter']).read_text())\n        if stage.get('textbook'): print('\\n---\\n'+(ROOT/stage['textbook']).read_text())")
p.write_text(s)
for course in sorted(ROOT.glob('[0-9][0-9]-*')):
    shutil.copy(p,course/'course_cli/main.py')
    if course.name=='06-inference':
        write(course/'course/EXPERIMENTS.md',(ROOT/'01-autograd/course/EXPERIMENTS.md').read_text())
    # Package explicit remote grading only where actual hardware is a required gate.
    if course.name in ['06-inference','07-distributed']:
        gpu='L4' if course.name=='06-inference' else 'L4:2'
        count=32 if course.name=='06-inference' else 14
        write(course/'scripts/modal_grade.py',f'''"""Explicit paid GPU execution; importing this file never launches a remote job."""
from pathlib import Path
import json
import modal

ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv==0.7.15")
    .add_local_file(ROOT/"pyproject.toml","/course/pyproject.toml",copy=True)
    .add_local_file(ROOT/"uv.lock","/course/uv.lock",copy=True)
    .add_local_file(ROOT/"README.md","/course/README.md",copy=True)
    .run_commands("cd /course && uv sync --locked --no-install-project --no-default-groups --group dev")
    .add_local_dir(ROOT/"src","/course/src",ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT/"grader","/course/grader",ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT/"course_cli","/course/course_cli",ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT/"course","/course/course",ignore=["**/__pycache__/**"]))
app=modal.App("model-course-{course.name}",image=image)

@app.function(gpu="{gpu}",timeout=600,max_containers=1)
def grade(stage:int):
    import os
    import subprocess
    env=os.environ.copy(); env['PYTHONPATH']='/course/src:/course'
    result=subprocess.run(['/course/.venv/bin/python','-m','course_cli.main','check',str(stage)],cwd='/course',env=env,capture_output=True,text=True,timeout=360)
    path=Path('/course/.course/latest.json'); report=json.loads(path.read_text()) if path.exists() else None
    junit=(Path('/course')/report['junit']).read_text() if report else None
    hardware=json.loads(subprocess.check_output(['/course/.venv/bin/python','-c','import torch,json; print(json.dumps({{"torch":torch.__version__,"cuda":torch.version.cuda,"devices":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}}))'],text=True))
    return dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,report=report,junit=junit,hardware=hardware)

@app.local_entrypoint()
def main(stage:int={count}):
    if not 1<=stage<={count}: raise ValueError('Invalid stage')
    # The returned remote fingerprint must match this local source snapshot to count locally.
    import sys
    sys.path.insert(0,str(ROOT))
    from course_cli.main import fingerprint
    before=fingerprint()
    result=grade.remote(stage)
    print(result['stdout']); print(result['stderr'])
    report=result['report']
    if report:
        report['backend']='modal'; report['hardware']=result['hardware']
        if report['fingerprint']!=before or before!=fingerprint():
            report['complete']=False
            print('Source changed or differed remotely; receipt is not a completion claim.')
        dest=ROOT/'.course'; dest.mkdir(exist_ok=True)
        (dest/'latest.json').write_text(json.dumps(report,indent=2)+'\\n')
        if result['junit']:
            path=ROOT/report['junit']; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(result['junit'])
    if result['exit_code']: raise SystemExit(result['exit_code'])
''')

write(ROOT/'README.md','''# Build the model stack

Eight independent, build-first courses teach the model lifecycle through **131 small stages**. You implement learner code; instructor-owned tests verify it. Each course has its own `uv` project, lockfile, chapters, conceptual hints, grader, experiment journal and capstone. Nothing completes itself by importing a reference implementation.

## Choose a starting point

| Course | Stages | Main artifact | Hardware |
|---|---:|---|---|
| [01 — Autodiff](01-autograd/README.md) | 14 | A trainable NumPy neural-network engine | CPU |
| [02 — Pretraining](02-pretraining/README.md) | 16 | A trained byte-level transformer and portable checkpoint | CPU; optional larger GPU experiments |
| [03 — Data](03-data/README.md) | 13 | Deduplicated, split, versioned JSONL dataset | CPU |
| [04 — Fine-tuning](04-finetuning/README.md) | 14 | SFT/DPO training code and portable LoRA adapters | CPU; optional larger GPU experiments |
| [05 — Evaluation](05-evaluation/README.md) | 14 | Reproducible per-case evidence and comparisons | CPU |
| [06 — Inference](06-inference/README.md) | 32 | Cached, paged, continuously batched serving engine | CPU through 30; NVIDIA GPU for 31–32 |
| [07 — Distributed training](07-distributed/README.md) | 14 | Real process collectives and sharded optimizer state | CPU through 13; two NVIDIA GPUs for 14 |
| [08 — Releases](08-releases/README.md) | 14 | Local artifact registry, release gates and rollback | CPU |

A useful first pass is **01 → 03 → 02 → 05 → 04 → 06 → 07 → 08**. Read evaluation before interpreting fine-tuning improvements. If you are already comfortable with gradients, begin with the inference course you originally requested and return to the foundations as needed. Prerequisites are solid Python, basic arrays, matrix multiplication, derivatives and elementary probability. The textbook introductions connect those ideas to each implementation.

## The learning loop

```bash
cd ~/Code/ai/courses/01-autograd
uv sync --locked
uv run course list
uv run course read 1
uv run course check 1
```

The first check intentionally fails at an unfinished learner function. Implement it in `src/learner/api.py` (or `src/toyvllm/` for inference), then rerun. Work through as many small stages as you need; there is no requirement to finish a whole course in one sitting.

- `uv run course check N` verifies stages 1–N cumulatively.
- `uv run course check N --only` isolates a stage for debugging and cannot establish cumulative completion.
- `uv run course check N --seed 91` varies seeded inputs where a gate uses randomized cases.
- `uv run course hint N --level 1` reveals one conceptual hint; levels 2 and 3 narrow the reasoning.
- `uv run course status` checks the latest receipt against current source, grader, catalog and lockfile.

No global installation is required. Each course is editable and self-contained. Use its own working directory; the `learner` package name is intentionally scoped to that course’s virtual environment. `uv sync` downloads dependencies but never starts a GPU job or downloads model weights.

## What gets verified

Acceptance tests use independent formulas, PyTorch baselines, adversarial examples and lifecycle invariants. They test gradients and parameter changes as well as outputs. Systems gates inspect actual cache reuse, packed projection work, process communication and cleanup. Every stage has a runnable gate; later stages integrate earlier components.

Correct mechanics, statistical model quality and hardware performance are separate claims. Controlled training examples prove a training path works. They do not prove useful generalization. Capstones require held-out comparisons and experiment records. GPU gates fail on missing hardware; they cannot claim completion by skipping. See [GRADING.md](GRADING.md) and [VALIDATION.md](VALIDATION.md).

The local grader is inspectable, not a secure hidden-test service. Treat tests as instructor-owned acceptance criteria. Complete references are deliberately outside learner source, under each course’s `instructor/` (inference retains `grader/oracle/`). Do not read these during the default path. [AGENTS.md](AGENTS.md) tells coding assistants to coach unless you request a worked solution.

## Connect the courses

[ARTIFACTS.md](ARTIFACTS.md) defines the shared dataset, model, adapter, evaluation and release contracts. [CAPSTONE.md](CAPSTONE.md) describes the end-to-end project. `tools/lifecycle.py` is a provided integration driver that invokes your completed APIs; it is not a replacement for implementing them. [GPU.md](GPU.md) explains optional remote execution and its limits.

The original `~/Code/ai/nanovllm` remains untouched. `06-inference` is a copy with preserved learner files, a new artifact-loader exercise and smaller grading gates. Work in one copy at a time; changes are not automatically synchronized between them.

These are substantial courses, not a promise of expertise after a fixed number of hours. Early stages may take under an hour; graph differentiation, decoder assembly, scheduling and distributed recovery can take several sessions. Follow the contracts, keep an experiment journal and explain a failure before reaching for another hint.
''')
write(ROOT/'AGENTS.md','''# Model engineering learning suite

The learner owns each course's src/. Default to coaching, failure explanation and graduated
conceptual hints. Do not solve learner stages unless explicitly asked for implementation help.
Preserve learner edits. User instructions take precedence.

The instructor owns grader/, instructor/, course contracts and course_cli/. Never weaken an
acceptance gate to make a submission pass. For course maintenance, fix demonstrated contract
or grader defects and explain them. Full solutions stay outside the default learning path.
Local tests are inspectable, not a secure hidden service.

Each course has an independent uv environment. Never import a sibling learner package by
accident. Use tools/lifecycle.py's isolated module loading for cross-course integration.
Do not launch paid remote compute without explicit authorization for the run. Adding or
validating runner code is not authorization to start a GPU job.

Instructor authoring scripts refuse to overwrite learner source. Do not bypass that guard.
''')
write(ROOT/'GRADING.md','''# Grading and feedback

Every stage in `course/stages.json` points to exact pytest node IDs. Cumulative checks include the boundary check and all gates through the selected stage. Inference reuses all of its original acceptance tests without weakening assertions. Its old broad milestones are now a textbook organization, not completion units.

The runner always selects learner code. It removes ambient pytest options/plugins, loads the timeout plugin explicitly, fixes a case seed and limits the total subprocess runtime. A receipt includes timestamp, counts, test selection, seed, environment and a SHA256 fingerprint of source, grader, CLI, project metadata, lockfile and stage catalog. Changing these inputs makes an earlier receipt stale. A skip, failure, collection error, empty selection or timed-out run cannot establish completion. An isolated gate cannot establish cumulative completion.

Receipts are convenience records, not tamper-proof credentials. Local instructors and learners can inspect or edit the grader. Boundary checks catch accidental direct instructor imports, not every possible attempt to bypass an exercise. The intended use is deliberate practice with independent expected behavior.

Course maintenance uses `uv run python -m pytest grader --implementation instructor.reference -m "not gpu"` (inference uses `grader.oracle`). This validates the course and creates no learner achievement. `tools/validate.py` runs that maintenance sweep. Fault-injection checks in `instructor/test_suite.py` demonstrate that representative plausible defects are rejected. Read [VALIDATION.md](VALIDATION.md) for work actually verified on this host.

Most functions have narrow input domains. Do not infer support for arbitrary shapes, malformed schemas, distributed topology changes or concurrent writes beyond the stated contract. Chapter contracts and scaffold docstrings are normative. If a test contradicts a contract, treat it as a course defect and explain the correction rather than learning an undocumented special case.

For learning: predict a small example; implement the contract; read the first failure; ask for hint level 1; isolate the broken invariant; run the cumulative gate. After passing, complete the chapter investigation and capstone evidence. Passing tests is necessary for mechanics but insufficient for scientific claims.
''')
write(ROOT/'GPU.md','''# CPU first, explicit GPU runs

All early exercises run locally with tiny inputs. No course imports trigger model downloads, account setup or paid compute. The distributed course uses real Gloo worker processes on CPU, which verify collective semantics but do not measure GPU scaling.

Two final gates require hardware:

```bash
cd ~/Code/ai/courses/06-inference
uv sync --locked --group gpu
uv run --group gpu modal setup
uv run --group gpu modal run scripts/modal_grade.py --stage 32
```

This requests one NVIDIA L4. For the distributed course:

```bash
cd ~/Code/ai/courses/07-distributed
uv sync --locked --group gpu
uv run --group gpu modal run scripts/modal_grade.py --stage 14
```

This requests TWO NVIDIA L4 GPUs in one container. Modal authentication is user-managed. Explicitly running these commands starts provider-billed compute; check your provider account and prices before using them. The runners use bounded container timeouts and one container maximum. They are not deployed services.

Only course source, grader, CLI, chapters and project metadata are uploaded. The runners do not upload home directories, credentials, original notebooks or instructor references for the seven new courses. Inference includes its oracle because fixed-checkpoint acceptance tests use it. Remote receipts include GPU identity and must match the local source fingerprint to support local completion.

The image installs the lockfile in a dedicated virtual environment. Tests execute with that environment’s Python. Multi-process workers inherit the same module path. No tests silently skip when the required CUDA hardware is absent. You can run the same `course check` command directly on a suitable local or rented CUDA host without Modal.

No remote image build or paid GPU run was performed while authoring these courses. Local SDK import and image definition checks do not establish remote runtime compatibility. GPU throughput, distributed scaling, numerical behavior on accelerators and cloud costs must be measured on actual hardware.
''')
write(ROOT/'ARTIFACTS.md','''# Shared artifact contracts

These formats keep the courses connected while preserving their independent environments. All examples are trusted local artifacts; do not load arbitrary third-party pickle payloads. Torch readers use `weights_only=True`. Record byte digests rather than relying on filenames.

## Dataset: toydata-v1

A directory contains `records.jsonl` and `manifest.json`. JSONL uses UTF-8, sorted keys, compact separators and one final newline per record. The manifest stores `format`, relative `file`, `sha256` and `records` count. Prepared records preserve `id` and original `text`, and add `split` (`train` or `validation`) and `content_sha256`. Train only on the training split. Reserved final evaluation cases must be separately identified; never repeatedly tune to them.

Course 03 produces this format. Course 02 consumes each document as `[BOS] + UTF-8 bytes + [EOS]`. The simple trainer takes separate sequences; packing metadata is a separate exercise and must be accompanied by an isolation mask before it is used by a transformer.

## Model: toylm-v1

A torch file contains `format='toylm-v1'`, `config`, `tokenizer`, `state_dict`, and `provenance`. Tokenizer metadata is exactly `{'kind':'utf8-byte','bos_id':256,'eos_id':257,'vocab_size':258}`. Byte IDs are 0–255. Training documents include EOS; ordinary generation prompts do not append EOS.

Configuration fields: vocab_size, dim, n_layers, n_heads, n_kv_heads, hidden_dim, max_seq_len. The architecture is the same pre-normalized RMSNorm / adjacent-pair RoPE / grouped-query attention / SwiGLU decoder in courses 02 and 06. Embedding and output weights are untied. There are no linear biases. Checkpoint keys:

- `embedding.weight`: [vocab,dim]
- `layers.i.attn_norm.weight`, `layers.i.ffn_norm.weight`: [dim]
- `layers.i.q.weight`: [n_heads*head_dim,dim]
- `layers.i.k.weight`, `layers.i.v.weight`: [n_kv_heads*head_dim,dim]
- `layers.i.o.weight`: [dim,dim]
- `layers.i.gate.weight`, `layers.i.up.weight`: [hidden_dim,dim]
- `layers.i.down.weight`: [dim,hidden_dim]
- `norm.weight`: [dim]; `lm_head.weight`: [vocab,dim]

Course 02 exports detached CPU tensors. Course 06 stage 30 loads them strictly and verifies metadata. Equality of dense logits is the first handoff test; only then compare cached/paged generation. This format is not an arbitrary pretrained-model loader.

## Adapter: toyadapter-v1

Course 04 stores `format`, `base_sha256`, `state` and `config`. State contains only qualified `A`/`B` tensors from LoRALinear modules. Per-module configuration contains `rank` and `alpha`. Base identity, target module paths, dimensions and scale must match before any restore changes a model. Merging adapters into ordinary linears yields a model that can be exported again as toylm-v1.

The fine-tuning API expects batched logits. The provided integration driver wraps the single-sequence training model by stacking per-sequence forward calls. This is an interface adapter, not a packed-training optimization. Right-padding does not influence preceding causal logits; padded targets are masked. Do not claim padding is computationally free.

## Evaluation: toyeval-v1

Course 05 returns `format`, `model_sha256`, `dataset_sha256`, `metrics={accuracy,n}` and `cases=[{id,prediction,score}]`. IDs are unique, n is the case count and accuracy is the mean score. The dataset digest identifies exact evaluation records, not the training corpus. Canonical JSON gives an evaluation file digest.

The score is a specified normalized exact-match metric; it is not a general semantic evaluator. More sophisticated metrics should receive a new explicit protocol version and retain per-case evidence. Course 08 verifies identity and aggregate consistency, not scientific validity or evaluator honesty.

## Release: toyrelease-v1 and local pointers

A release manifest contains selected relative paths, byte sizes, hashes and copied metadata. A content-addressed object store holds exact model/report bytes. An active pointer is `{'current': model_digest, 'previous': prior_digest_or_null}`. Compare-and-swap promotion prevents stale concurrent decisions. The local registry is a teaching substitute for a deployment control plane, not a live server deployment.

A release record returned by the integration step includes model and report digests plus the pointer state. Preserve it beside the experiment journal. Model bytes, evidence and policy decisions are distinct artifacts.
''')
