"""Author visualization companions; no grader or solution imports. Overwrites these three files."""

from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)
books = {
    "01_attention_sandbox": [
        (
            "markdown",
            "# Attention visibility sandbox\n\nOptional companion to stages 02 and 04. This visualizes a small offset mask; it does not implement your attention function or grant stage credit. Change `cached` and `new_tokens` before predicting the visible cells.",
        ),
        (
            "code",
            'import numpy as np\nimport matplotlib.pyplot as plt\ncached, new_tokens = 4, 3\nqueries = cached + np.arange(new_tokens)\nkeys = np.arange(cached + new_tokens)\nvisible = keys[None, :] <= queries[:, None]\nplt.figure(figsize=(7, 3))\nplt.imshow(visible, cmap="Blues", aspect="auto", vmin=0, vmax=1)\nplt.yticks(range(new_tokens), queries)\nplt.xticks(keys)\nplt.xlabel("Absolute key position")\nplt.ylabel("Absolute query position")\nplt.title("Allowed attention positions")\nplt.tight_layout()\nplt.show()',
        ),
        (
            "markdown",
            "Why would resetting the query position to zero be wrong after a cached prefix? Why does a multi-token append need a mask even though all old keys are visible? Return to the stage contract and implement the operation in `src/toyvllm/model.py`.",
        ),
    ],
    "02_memory_sandbox": [
        (
            "markdown",
            "# Page-size and slack sandbox\n\nOptional companion to stage 06. The chart models on-demand tail slack. Reservation-based admission adds another kind of slack, and resident pool bytes remain allocated even when blocks are free.",
        ),
        (
            "code",
            'import math\nimport matplotlib.pyplot as plt\nlengths = [1, 7, 9, 17, 31]\nblock_sizes = [1, 2, 4, 8, 16]\nslack = [sum(math.ceil(n/b)*b-n for n in lengths) for b in block_sizes]\nplt.figure(figsize=(7, 3))\nplt.bar([str(b) for b in block_sizes], slack, color="#2563eb")\nplt.xlabel("Block size (tokens)")\nplt.ylabel("Unused allocated tail slots")\nplt.title("One source of cache fragmentation")\nplt.tight_layout()\nplt.show()',
        ),
        (
            "markdown",
            "Choose a prompt length exactly at a block boundary and one just beyond it. How does the number of pages change? Which logical positions use the next page? Use your own allocator to verify after stage 06 passes.",
        ),
    ],
    "03_trace_viewer": [
        (
            "markdown",
            "# View your scheduler trace\n\nOptional companion to stage 07. Export your own completed `engine.history` as `artifacts/learner-trace.json` using `json.dumps`. This notebook reads that artifact; it does not run the instructor engine. Missing trace data is reported explicitly.",
        ),
        (
            "code",
            'from pathlib import Path\nimport json\nimport matplotlib.pyplot as plt\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())\npath = root / "artifacts/learner-trace.json"\nif not path.exists():\n    print("No learner trace yet. Build stage 07 and export engine.history first.")\nelse:\n    history = json.loads(path.read_text())\n    ids = sorted({row["request_id"] for step in history for row in step["scheduled"]})\n    fig, ax = plt.subplots(figsize=(10, max(3, len(ids)*0.6)))\n    for i, step in enumerate(history):\n        for row in step["scheduled"]:\n            y = ids.index(row["request_id"])\n            color = "#2563eb" if row["phase"] == "prefill" else "#16a34a"\n            ax.barh(y, .85, left=i, color=color)\n            ax.text(i+.42, y, str(row["count"]), ha="center", va="center", color="white")\n    ax.set_yticks(range(len(ids)), ids)\n    ax.set_xlabel("Execution step (not elapsed time)")\n    ax.set_title("Your schedule: blue prefill, green decode, labels are token counts")\n    plt.tight_layout()\n    plt.show()',
        ),
        (
            "markdown",
            "Look for late admissions, prompt chunks without emissions, and capacity freed by completion. A step index is not a hardware utilization or latency measurement. Explain each row using your request state transitions.",
        ),
    ],
}
for name, cells in books.items():
    nb = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(s) if kind == "markdown" else nbformat.v4.new_code_cell(s)
            for kind, s in cells
        ],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )
    for i, cell in enumerate(nb.cells):
        cell.id = f"{name}-{i}"
    nbformat.validate(nb)
    nbformat.write(nb, OUT / f"{name}.ipynb")
    print(OUT / f"{name}.ipynb")
