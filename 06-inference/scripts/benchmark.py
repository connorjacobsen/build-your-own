"""After stage 10: benchmark your code; correctness and timing are separate."""

import argparse
import json
from pathlib import Path
import platform
import torch
from toyvllm import ByteTokenizer, Engine, make_model
from toyvllm.sampling import generate_naive, generate_cached
from toyvllm.metrics import benchmark

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
args = parser.parse_args()
torch.set_num_threads(1)
model = make_model(device=args.device)
prompt = ByteTokenizer().encode("Benchmark identical useful work. " * 2)
count = 16


def paged():
    engine = Engine(model, prefix_caching=False)
    engine.add_request("one", prompt, count)
    return engine.run()["one"]


expected = generate_naive(model, prompt, count)
assert expected == generate_cached(model, prompt, count) == paged()
methods = {
    "naive": lambda: generate_naive(model, prompt, count),
    "contiguous_cache": lambda: generate_cached(model, prompt, count),
    "paged_cold": paged,
}
report = {
    "implementation": "toyvllm",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "platform": platform.platform(),
    "device": args.device,
    "dtype": "float32",
    "model": vars(model.cfg),
    "prompt_tokens": len(prompt),
    "output_tokens": count,
    "warmup": 2,
    "repeats": 5,
    "outputs_equal": True,
    "prefix_cache": False,
    "boundary": "whole generation; paged includes engine/pool construction; model construction excluded",
    "timings": {
        name: benchmark(fn, device=args.device, warmup=2, repeats=5) for name, fn in methods.items()
    },
}
if args.device == "cuda":
    report["gpu"] = torch.cuda.get_device_name(0)
path = Path(__file__).resolve().parents[1] / "artifacts" / f"learner-benchmark-{args.device}.json"
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
