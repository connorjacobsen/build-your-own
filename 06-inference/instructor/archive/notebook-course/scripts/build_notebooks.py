"""Convert course/lessons/*.md to notebooks; Python fences become executable cells.

Do not regenerate over edited notebooks without saving your work. Markdown files
are the authoring source; notebooks are the learner-facing, independent artifacts.
"""

from pathlib import Path
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1]
SETUP = """from pathlib import Path
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from nanovllm_course import TinyConfig, TinyLM, make_model, ByteTokenizer, Engine, SamplingParams

# Locate the course from either the project root or notebooks/.
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)
torch.manual_seed(7)
DEVICE = "cpu"  # The tested core path; optional accelerator experiments are in lesson 12.
RUN_EXERCISES = False  # Set True only after implementing this lesson's named exercise.
plt.rcParams.update({"figure.figsize": (8, 3.5), "axes.spines.top": False, "axes.spines.right": False})
"""

for path in sorted((ROOT / "course/lessons").glob("*.md")):
    parts = re.split(r"^```python\s*\n(.*?)^```\s*$", path.read_text(), flags=re.M | re.S)
    cells = []
    for i, part in enumerate(parts):
        if not part.strip():
            continue
        cells.append(
            nbformat.v4.new_code_cell(part.strip())
            if i % 2
            else nbformat.v4.new_markdown_cell(part.strip())
        )
        if i == 0:
            cells.append(nbformat.v4.new_code_cell(SETUP))
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "course": {"source": str(path.relative_to(ROOT)), "independent": True},
        },
    )
    for i, cell in enumerate(notebook.cells):
        cell.id = f"{path.stem[:40]}-{i:03}"
    nbformat.validate(notebook)
    dest = ROOT / "notebooks" / (path.stem + ".ipynb")
    nbformat.write(notebook, dest)
    print(f"{dest.name}: {len(cells)} cells")
