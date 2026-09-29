"""Explicit paid GPU execution; importing this file never launches a remote job."""

from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv==0.7.15")
    .add_local_file(ROOT / "pyproject.toml", "/course/pyproject.toml", copy=True)
    .add_local_file(ROOT / "uv.lock", "/course/uv.lock", copy=True)
    .add_local_file(ROOT / "README.md", "/course/README.md", copy=True)
    .run_commands(
        "cd /course && uv sync --locked --no-install-project --no-default-groups --group dev"
    )
    .add_local_dir(ROOT / "src", "/course/src", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "grader", "/course/grader", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "course_cli", "/course/course_cli", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "course", "/course/course", ignore=["**/__pycache__/**"])
)
app = modal.App("model-course-07-distributed", image=image)


@app.function(gpu="L4:2", timeout=600, max_containers=1)
def grade(stage: int):
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = "/course/src:/course"
    result = subprocess.run(
        ["/course/.venv/bin/python", "-m", "course_cli.main", "check", str(stage)],
        cwd="/course",
        env=env,
        capture_output=True,
        text=True,
        timeout=360,
    )
    path = Path("/course/.course/latest.json")
    report = json.loads(path.read_text()) if path.exists() else None
    junit = (Path("/course") / report["junit"]).read_text() if report else None
    hardware = json.loads(
        subprocess.check_output(
            [
                "/course/.venv/bin/python",
                "-c",
                'import torch,json; print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"devices":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))',
            ],
            text=True,
        )
    )
    return dict(
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        report=report,
        junit=junit,
        hardware=hardware,
    )


@app.local_entrypoint()
def main(stage: int = 14):
    if not 1 <= stage <= 14:
        raise ValueError("Invalid stage")
    # The returned remote fingerprint must match this local source snapshot to count locally.
    import sys

    sys.path.insert(0, str(ROOT))
    from course_cli.main import fingerprint

    before = fingerprint()
    result = grade.remote(stage)
    print(result["stdout"])
    print(result["stderr"])
    report = result["report"]
    if report:
        report["backend"] = "modal"
        report["hardware"] = result["hardware"]
        if report["fingerprint"] != before or before != fingerprint():
            report["complete"] = False
            print("Source changed or differed remotely; receipt is not a completion claim.")
        dest = ROOT / ".course"
        dest.mkdir(exist_ok=True)
        (dest / "latest.json").write_text(json.dumps(report, indent=2) + "\n")
        if result["junit"]:
            path = ROOT / report["junit"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(result["junit"])
    if result["exit_code"]:
        raise SystemExit(result["exit_code"])
