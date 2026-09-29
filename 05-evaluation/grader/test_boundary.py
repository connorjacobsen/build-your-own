import ast
from pathlib import Path


def test_no_instructor_imports():
    for path in (Path(__file__).resolve().parents[1] / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n.split(".")[0] in {"grader", "instructor"} for n in names), (
                f"{path}: keep instructor code outside your implementation"
            )
