"""Execute each lesson in a fresh kernel; save evidence without editing learner notebooks."""

import argparse
import json
from pathlib import Path
from time import perf_counter
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("patterns", nargs="*", default=["*.ipynb"])
args = parser.parse_args()
paths = sorted({p for pattern in args.patterns for p in (ROOT / "notebooks").glob(pattern)})
if not paths:
    parser.error("no notebooks matched")
out = ROOT / "artifacts/executed"
out.mkdir(parents=True, exist_ok=True)
report = []
for path in paths:
    start = perf_counter()
    nb = nbformat.read(path, as_version=4)
    try:
        NotebookClient(
            nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
        ).execute()
    except Exception:
        nbformat.write(nb, out / path.name)
        print(f"FAILED {path.name}", flush=True)
        raise
    nbformat.write(nb, out / path.name)
    record = {
        "notebook": path.name,
        "seconds": round(perf_counter() - start, 2),
        "status": "passed",
    }
    report.append(record)
    print(record, flush=True)
(ROOT / "artifacts/notebook-report.json").write_text(json.dumps(report, indent=2) + "\n")
