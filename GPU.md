# CPU first, explicit GPU runs

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
