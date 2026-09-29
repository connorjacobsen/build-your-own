"""Fetch a small pinned vLLM reading set, not its dependencies or entire repository."""

from pathlib import Path
from urllib.request import urlopen
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
REVISION = "b8b302cde434df8c9289a2b465406b47ebab1c2d"
PATHS = [
    "vllm/v1/request.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/core/kv_cache_manager.py",
    "vllm/v1/core/block_pool.py",
    "vllm/v1/engine/core.py",
    "vllm/v1/worker/gpu_model_runner.py",
    "vllm/v1/sample/sampler.py",
    "vllm/model_executor/models/llama.py",
    "docs/design/prefix_caching.md",
    "docs/design/arch_overview.md",
]
manifest = {
    "repository": "https://github.com/vllm-project/vllm",
    "tag": "v0.11.0",
    "revision": REVISION,
    "files": {},
}
for relative in PATHS:
    url = f"https://raw.githubusercontent.com/vllm-project/vllm/{REVISION}/{relative}"
    data = urlopen(url, timeout=30).read()
    path = ROOT / ".reference" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    manifest["files"][relative] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "url": url,
    }
    print(relative)
(ROOT / "docs/upstream_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
