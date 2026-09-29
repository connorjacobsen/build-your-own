"""Small CPU comparison; saves raw timings and explicit metadata."""

import json
import platform
from pathlib import Path
import torch
from nanovllm_course import make_model, ByteTokenizer, Engine
from nanovllm_course.sampling import generate_naive, generate_cached
from nanovllm_course.metrics import benchmark

torch.set_num_threads(1)
model = make_model()
prompt = ByteTokenizer().encode("Benchmark a fixed workload. " * 2)
output_length = 16


def paged():
    engine = Engine(model, prefix_caching=False)
    engine.add_request("one", prompt, output_length)
    return engine.run()["one"]


expected = generate_naive(model, prompt, output_length)
assert expected == generate_cached(model, prompt, output_length) == paged()
report = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "platform": platform.platform(),
    "device": "cpu",
    "dtype": "float32",
    "torch_threads": 1,
    "model": vars(model.cfg),
    "prompt_tokens": len(prompt),
    "output_tokens": output_length,
    "sampling": "greedy; no EOS",
    "boundary": "whole generation call; paged case includes pool/engine construction; model pre-created",
    "prefix_cache": False,
    "warmup": 2,
    "repeats": 5,
    "outputs_equal": True,
    "timings": {
        "naive": benchmark(lambda: generate_naive(model, prompt, output_length)),
        "contiguous_cache": benchmark(lambda: generate_cached(model, prompt, output_length)),
        "paged_engine_cold": benchmark(paged),
    },
}
path = Path(__file__).resolve().parents[1] / "artifacts/benchmark.json"
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
