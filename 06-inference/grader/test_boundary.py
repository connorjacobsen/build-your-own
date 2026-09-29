"""A learning boundary, not an anti-cheat sandbox: implement your own behavior."""

import ast
from pathlib import Path
import pytest


@pytest.mark.examples
@pytest.mark.extended
def test_learner_package_does_not_import_instructor_code():
    root = Path(__file__).resolve().parents[1] / "src/toyvllm"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] not in {"grader", "instructor", "nanovllm_course"}, (
                    f"{path.name}:{node.lineno}: learner code must not import the instructor implementation; "
                    "build this behavior in toyvllm instead"
                )
