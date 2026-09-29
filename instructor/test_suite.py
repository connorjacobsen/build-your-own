"""Instructor-only fault injection and grader bookkeeping tests. Contains deliberately bad code."""

import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pair(course):
    slug = course.replace("-", "_")
    return load(ROOT / course / "instructor/reference.py", slug + "_ref"), load(
        ROOT / course / "grader/test_stages.py", slug + "_checks"
    )


def test_rejects_missing_shared_gradient():
    api, tests = pair("01-autograd")

    def bad(a, b):
        return api.Tensor(a.data * b.data)

    api.multiply = bad
    with pytest.raises(AssertionError):
        tests.test_06(api, 1729)


def test_rejects_missing_weight_decay():
    api, tests = pair("02-pretraining")
    original = api.adamw_step

    def bad(*args, **kwargs):
        kwargs["weight_decay"] = 0.0
        return original(*args, **kwargs)

    api.adamw_step = bad
    with pytest.raises(AssertionError):
        tests.test_09(api, 1729)


def test_rejects_sequence_weighted_training():
    api, tests = pair("02-pretraining")

    def bad(model, optimizer, sequences):
        optimizer.zero_grad()
        loss = sum(api.token_loss(model(s[:-1])[0], torch.tensor(s[1:])) for s in sequences) / len(
            sequences
        )
        loss.backward()
        optimizer.step()
        return loss.item()

    api.train_step = bad
    with pytest.raises(AssertionError):
        tests.test_12(api)


def test_rejects_pairwise_groups_without_transitive_closure():
    api, tests = pair("03-data")
    api.duplicate_components = lambda *a, **kw: [["a", "b"], ["c"], ["d"], ["e"]]
    with pytest.raises(AssertionError):
        tests.test_06(api)


def test_rejects_cross_document_attention():
    api, tests = pair("03-data")
    api.attention_mask = lambda segments: [
        [k <= q for k in range(len(segments))] for q in range(len(segments))
    ]
    with pytest.raises(AssertionError):
        tests.test_11(api)


def test_rejects_unshifted_sft_loss():
    api, tests = pair("04-finetuning")
    api.masked_loss = lambda x, y: torch.nn.functional.cross_entropy(
        x.flatten(0, 1), y.flatten(), ignore_index=-100
    )
    with pytest.raises(AssertionError):
        tests.test_03(api, 1729)


def test_rejects_lora_merge_without_scale():
    api, tests = pair("04-finetuning")

    def bad(layer):
        out = torch.nn.Linear(layer.base.in_features, layer.base.out_features)
        out.weight.data.copy_(layer.base.weight + layer.B @ layer.A)
        out.bias.data.copy_(layer.base.bias)
        return out

    api.merge_lora = bad
    with pytest.raises(AssertionError):
        tests.test_08(api, 1729)


def test_rejects_mean_of_document_perplexities():
    api, tests = pair("05-evaluation")
    api.perplexity = lambda a, b: sum(math.exp(x / n) for x, n in zip(a, b)) / len(a)
    with pytest.raises(AssertionError):
        tests.test_05(api)


def test_rejects_unpaired_bootstrap():
    api, tests = pair("05-evaluation")

    def bad(a, b, seed=0, resamples=2000, confidence=0.95):
        rng = np.random.default_rng(seed)
        a = np.array(a)
        b = np.array(b)
        values = b[rng.integers(len(b), size=(resamples, len(b)))].mean(1) - a[
            rng.integers(len(a), size=(resamples, len(a)))
        ].mean(1)
        return dict(
            delta=float((b - a).mean()),
            low=float(np.quantile(values, 0.025)),
            high=float(np.quantile(values, 0.975)),
        )

    api.paired_bootstrap = bad
    with pytest.raises(AssertionError):
        tests.test_08(api, 1729)


def test_rejects_averaging_microbatch_means():
    api, tests = pair("07-distributed")

    def bad(model, batches, loss_fn):
        model.zero_grad()
        losses = [loss_fn(model(x), y) / len(y) for x, y in batches]
        loss = sum(losses) / len(losses)
        loss.backward()
        return loss.item()

    api.accumulate_gradients = bad
    with pytest.raises(AssertionError):
        tests.test_05(api, 1729)


def test_rejects_partial_unscale_on_overflow():
    api, tests = pair("07-distributed")

    def bad(parameters, scale):
        for p in parameters:
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    return False
                p.grad.div_(scale)
        return True

    api.unscale_gradients = bad
    with pytest.raises(AssertionError):
        tests.test_10(api)


def test_rejects_length_only_integrity(tmp_path):
    api, tests = pair("08-releases")

    def bad(root, manifest):
        for row in manifest["files"]:
            assert (Path(root) / row["path"]).stat().st_size == row["size_bytes"]
        return True

    api.verify_manifest = bad
    with pytest.raises(pytest.fail.Exception):
        tests.test_05(api, tmp_path)


def test_rejects_unconditional_release():
    api, tests = pair("08-releases")
    api.release_gate = lambda *a, **kw: dict(eligible=True, reasons=[])
    with pytest.raises(AssertionError):
        tests.test_08(api)


def mini_course(tmp_path, second="assert True"):
    for folder in ["course_cli", "course", "grader", "src/learner"]:
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "01-autograd/course_cli/main.py", tmp_path / "course_cli/main.py")
    (tmp_path / "course_cli/__init__.py").write_text("")
    (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\npythonpath=[".","src"]\n')
    (tmp_path / "uv.lock").write_text("fixture")
    (tmp_path / "src/learner/api.py").write_text("value=1\n")
    (tmp_path / "grader/test_boundary.py").write_text("def test_boundary(): assert True\n")
    (tmp_path / "grader/conftest.py").write_text(
        'def pytest_addoption(parser):\n    parser.addoption("--implementation")\n    parser.addoption("--case-seed")\n'
    )
    (tmp_path / "grader/test_cases.py").write_text(
        "import pytest\ndef test_one(): assert True\ndef test_two(): " + second + "\n"
    )
    stages = [
        dict(
            number=i,
            title=str(i),
            symbols="value",
            chapter="course/x.md",
            hints=["a", "b", "c"],
            tests=[f"grader/test_cases.py::test_{name}"],
        )
        for i, name in [(1, "one"), (2, "two")]
    ]
    (tmp_path / "course/stages.json").write_text(json.dumps(stages))
    return tmp_path


def run_cli(root, *args):
    return subprocess.run(
        [sys.executable, "-m", "course_cli.main", *args],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cumulative_pass_and_nested_source_staleness(tmp_path):
    root = mini_course(tmp_path)
    assert run_cli(root, "check", "2").returncode == 0
    r = json.loads((root / ".course/latest.json").read_text())
    assert r["complete"] and r["passed"] == 3
    nested = root / "src/learner/nested"
    nested.mkdir()
    (nested / "helper.py").write_text("x=2")
    status = run_cli(root, "status")
    assert "STALE" in status.stdout and "No current cumulative" in status.stdout


def test_isolated_gate_never_completes(tmp_path):
    root = mini_course(tmp_path)
    assert run_cli(root, "check", "2", "--only").returncode == 0
    assert not json.loads((root / ".course/latest.json").read_text())["complete"]


def test_skips_cannot_complete(tmp_path):
    root = mini_course(tmp_path, 'pytest.skip("missing hardware")')
    result = run_cli(root, "check", "2")
    r = json.loads((root / ".course/latest.json").read_text())
    assert result.returncode != 0 and not r["complete"] and r["skipped"] == 1


def test_failed_case_cannot_complete(tmp_path):
    root = mini_course(tmp_path, "assert False")
    result = run_cli(root, "check", "2")
    r = json.loads((root / ".course/latest.json").read_text())
    assert result.returncode != 0 and not r["complete"] and r["failed"] == 1


def test_every_stage_has_existing_gate_and_nonempty_chapter():
    for course in ROOT.glob("[0-9][0-9]-*"):
        stages = json.loads((course / "course/stages.json").read_text())
        assert [s["number"] for s in stages] == list(range(1, len(stages) + 1))
        for s in stages:
            assert len((course / s["chapter"]).read_text().split()) > 100 and len(s["hints"]) == 3
            for node in s["tests"]:
                path, name = node.split("::")
                assert ("def " + name + "(") in (course / path).read_text()


def test_inference_preserves_original_acceptance_cases_and_learner_edits():
    old = Path("/Users/connor/Code/ai/nanovllm")
    new = ROOT / "06-inference"
    if not old.exists():
        pytest.skip("Original workspace not available on this host")
    for path in (old / "src").rglob("*.py"):
        assert path.read_bytes() == (new / path.relative_to(old)).read_bytes()
    for path in (old / "grader/stages").glob("test_*.py"):
        assert path.read_bytes() == (new / path.relative_to(old)).read_bytes()
