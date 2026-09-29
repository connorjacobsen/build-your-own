"""Provided cross-course integration driver. Default: unfinished learner APIs.

Run from a course environment: uv run python ../tools/lifecycle.py --output /tmp/my-lifecycle
Instructor-only maintenance: add --reference. That creates no learner completion receipts.
"""

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load_course(slug, reference=False):
    path = ROOT / slug / ("instructor/reference.py" if reference else "src/learner/api.py")
    name = "lifecycle_" + slug.replace("-", "_") + ("_reference" if reference else "_learner")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class BatchInterface(nn.Module):
    """Provided interface adapter; batching optimization remains in the inference course."""

    def __init__(self, single):
        super().__init__()
        self.single = single

    def forward(self, ids):
        return torch.stack([self.single(row.tolist())[0] for row in ids])


def run(output, reference=False):
    torch.manual_seed(17)  # Reproducible adapter initialization and integration fixture.
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory to preserve prior evidence")
    output.mkdir(parents=True)
    data = load_course("03-data", reference)
    train = load_course("02-pretraining", reference)
    tune = load_course("04-finetuning", reference)
    evaluate = load_course("05-evaluation", reference)
    release = load_course("08-releases", reference)
    # Controlled, tiny corpus: this is integration evidence, not a meaningful language benchmark.
    raw = [dict(id=f"doc-{i:03}", text=f"item {i} gives {i % 3}") for i in range(24)]
    manifest = data.prepare_dataset(raw, [], output / "dataset", salt="lifecycle")
    records = [
        json.loads(line) for line in (output / "dataset/records.jsonl").read_text().splitlines()
    ]
    train_rows = [r for r in records if r["split"] == "train"]
    heldout = [r for r in records if r["split"] == "validation"]
    if not train_rows or not heldout:
        raise RuntimeError("Fixture split must include training and held-out cases")
    sequences = [train.encode_bytes(r["text"]) for r in train_rows]
    model, losses = train.train_run(sequences, steps=8, seed=17)
    base_path = output / "base.pt"
    base_digest = train.export_model(
        base_path, model, dict(dataset_sha256=manifest["sha256"], seed=17, steps=8)
    )
    ids = [256] + list(b"item 1 gives ")
    baseline_logits = model(ids)[0].detach().clone()
    inference_root = ROOT / "06-inference"
    sys.path.insert(0, str(inference_root if reference else inference_root / "src"))
    loader = importlib.import_module(
        "grader.oracle.artifacts" if reference else "toyvllm.artifacts"
    )
    baseline = loader.load_export(base_path)
    torch.testing.assert_close(baseline(ids)[0], baseline_logits)
    tune.inject_lora(model, ["lm_head"], rank=2, alpha=2.0)
    parameters = tune.freeze_except_adapters(model)
    optimizer = torch.optim.AdamW(parameters, lr=0.01)
    examples = [
        tune.chat_tokens(
            [
                dict(role="user", content=r["text"][:-1]),
                dict(role="assistant", content=r["text"][-1]),
            ]
        )
        for r in train_rows[:4]
    ]
    batch = tune.collate(examples)
    wrapper = BatchInterface(model)
    sft_losses = [tune.sft_step(wrapper, optimizer, batch) for _ in range(3)]
    adapter_digest = tune.save_adapter(output / "adapter.pt", model, base_digest)
    before = model(ids)[0].detach().clone()
    model.lm_head = tune.merge_lora(model.lm_head)
    torch.testing.assert_close(model(ids)[0], before, atol=2e-6, rtol=2e-5)
    candidate_path = output / "candidate.pt"
    candidate_digest = train.export_model(
        candidate_path, model, dict(base_sha256=base_digest, adapter_sha256=adapter_digest)
    )
    candidate = loader.load_export(candidate_path)
    torch.testing.assert_close(candidate(ids)[0], model(ids)[0], atol=2e-6, rtol=2e-5)
    # Verify an optimized serving path also consumes these actual trained parameters.
    sampling = importlib.import_module(
        "grader.oracle.sampling" if reference else "toyvllm.sampling"
    )
    engine_module = importlib.import_module(
        "grader.oracle.engine" if reference else "toyvllm.engine"
    )
    expected = sampling.generate_naive(candidate, ids, 3)
    engine = engine_module.Engine(
        candidate, block_size=4, num_blocks=16, token_budget=5, prefill_chunk=3
    )
    engine.add_request("handoff", ids, 3)
    assert engine.run()["handoff"] == expected
    engine.clear_prefix_cache()
    assert engine.pool.n_free == engine.pool.num_blocks
    cases = [dict(id=r["id"], prompt=r["text"][:-1], acceptable=[r["text"][-1]]) for r in heldout]
    eval_manifest = data.write_dataset(output / "eval-cases", cases)

    def predictor(m):
        def predict(prompt):
            with torch.inference_mode():
                logits, _ = m([256] + list(prompt.encode()))
                token = int(logits[-1].argmax())
            return bytes([token]).decode("utf-8", errors="replace") if token < 256 else ""

        return predict

    base_report = evaluate.evaluate(
        predictor(baseline), cases, base_digest, eval_manifest["sha256"]
    )
    candidate_report = evaluate.evaluate(
        predictor(candidate), cases, candidate_digest, eval_manifest["sha256"]
    )
    evaluate.write_report(output / "baseline-eval.json", base_report)
    evaluate.write_report(output / "candidate-eval.json", candidate_report)
    pointer = output / "registry/active.json"
    store = output / "registry/objects"
    release.register_artifact(store, base_path)
    release.promote(pointer, base_digest, None)
    # Permissive policy is intentional for an integration smoke test, NEVER a quality claim.
    record = release.release_model(
        candidate_path,
        output / "candidate-eval.json",
        base_report,
        store,
        pointer,
        base_digest,
        min_accuracy=0,
        max_drop=1,
        min_cases=1,
    )
    rolled = release.rollback(pointer, candidate_digest)
    assert rolled["current"] == base_digest
    summary = dict(
        mode="instructor reference" if reference else "learner",
        training_documents=len(train_rows),
        heldout_cases=len(cases),
        pretraining_loss=losses,
        sft_loss=sft_losses,
        baseline_accuracy=base_report["metrics"]["accuracy"],
        candidate_accuracy=candidate_report["metrics"]["accuracy"],
        release=record,
        rollback=rolled,
        quality_claim=False,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Instructor maintenance only; reads complete solutions",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = run(args.output, args.reference)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
