"""Instructor maintenance generator. Never rerun over learner edits without reviewing a diff."""
import ast
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/Users/connor/Code/ai/nanovllm')

def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + '\n')

COURSES = []
def build(slug, title, intro, stages, source, tests, sources, provided=()):
    root = ROOT / slug
    if (root / 'src').exists():
        raise RuntimeError(f'Refusing to overwrite learner work: {root}')
    root.mkdir(exist_ok=True)
    COURSES.append((slug, title, len(stages)))
    write(root/'instructor/reference.py', source)
    tree = ast.parse(source)
    symbols = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            symbols[node.name] = ast.get_docstring(node) or ''
            if node.name in provided:
                continue
            targets = [node] if isinstance(node, ast.FunctionDef) else [n for n in node.body if isinstance(n, ast.FunctionDef)]
            for fn in targets:
                doc = ast.get_docstring(fn)
                fn.body = ([ast.Expr(value=ast.Constant(doc))] if doc else []) + [ast.Raise(exc=ast.Call(func=ast.Name(id='NotImplementedError', ctx=ast.Load()), args=[ast.Constant(f'Implement {node.name}' + (f'.{fn.name}' if fn is not node else '') + '; see course list/read')], keywords=[]))]
    write(root/'src/learner/api.py', ast.unparse(ast.fix_missing_locations(tree)))
    for p in ['src/learner/__init__.py', 'instructor/__init__.py', 'grader/__init__.py', 'course_cli/__init__.py']:
        write(root/p, '"""Course package."""')
    write(root/'grader/test_stages.py', tests)
    shutil.copy(ROOT/'instructor/authoring/runner.py',root/'course_cli/main.py')
    write(root/'grader/conftest.py', '''import importlib
import os
import pytest
import torch

def pytest_addoption(parser):
    parser.addoption('--implementation', default='learner.api')
    parser.addoption('--case-seed', type=int, default=1729)

def pytest_configure(config):
    torch.set_num_threads(1)

@pytest.fixture
def api(request):
    return importlib.import_module(request.config.getoption('--implementation'))

@pytest.fixture(params=[0, 1, 2], ids=['case-a', 'case-b', 'case-c'])
def seed(request):
    return request.config.getoption('--case-seed') + request.param * 101
''')
    write(root/'grader/test_boundary.py', '''import ast
from pathlib import Path

def test_no_instructor_imports():
    for path in (Path(__file__).resolve().parents[2] / 'src').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            assert not any(n.split('.')[0] in {'grader', 'instructor'} for n in names), f'{path}: keep instructor code outside your implementation'
''')
    manifest = []
    links = '\n'.join(f'- [{name}]({url})' for name,url in sources)
    for i, s in enumerate(stages,1):
        # title, symbols, explanatory prose, contract details, experiment, three hints
        name, names, lesson, contract, exercise, hints = s
        slugpart = names.split(',')[0].strip().lower().replace('.','-').replace('_','-')
        chapter = f'course/chapters/{i:02}_{slugpart}.md'
        manifest.append({'number':i, 'title':name, 'symbols':names, 'chapter':chapter, 'tests':[f'grader/test_stages.py::test_{i:02}'], 'hints':hints})
        docs = '\n\n'.join(f'**`{n.strip()}`**\n\n{symbols.get(n.strip(), "See the method signatures in the learner scaffold.")}' for n in names.split(','))
        write(root/chapter, f'''# {i:02}. {name}

{lesson}

## Implementation contract

Work in `src/learner/api.py`. Implement **{names}**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

{docs}

{contract}

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check {i} --only
uv run course check {i}
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint {i} --level 1`.

## Explain and investigate

{exercise}

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

{links}
''')
    write(root/'course/stages.json', json.dumps(manifest,indent=2))
    write(root/'pyproject.toml', f'''[project]
name = "model-course-{slug}"
version = "0.1.0"
description = "{title}: a build-first model engineering course"
requires-python = ">=3.11,<3.13"
dependencies = ["torch>=2.6,<2.8", "numpy>=2.1,<3"]

[project.scripts]
course = "course_cli.main:main"

[dependency-groups]
dev = ["pytest>=8.3,<9", "pytest-timeout>=2.3,<3", "ruff>=0.11,<0.12"]
gpu = ["modal>=1.0,<2"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/learner", "course_cli", "grader"]

[tool.pytest.ini_options]
pythonpath = [".", "src"]
testpaths = ["grader"]
timeout = 30
markers = ["gpu: requires actual CUDA hardware", "distributed: launches real worker processes"]

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F"]
''')
    write(root/'.python-version','3.11')
    write(root/'.gitignore','.venv/\n.course/\n__pycache__/\n.pytest_cache/\n.ruff_cache/\nartifacts/\n*.egg-info/')
    write(root/'AGENTS.md', '''# Learning course
The learner owns src/learner/. Default to coaching and graduated conceptual hints.
Do not implement a learner solution unless explicitly asked. Preserve learner edits.
The instructor owns grader/, instructor/, course contracts and course_cli/.
Do not weaken tests to make a submission pass. Course maintenance may fix demonstrated
contract or grader defects; explain the correction. Explicit user instructions take precedence.
Complete solutions belong only under instructor/. The local grader is inspectable,
not a secure hidden-test service. Do not consult reference solutions during routine coaching.
''')
    rows='\n'.join(f'| {i} | [{s[0]}]({manifest[i-1]["chapter"]}) | `{s[1]}` |' for i,s in enumerate(stages,1))
    write(root/'README.md',f'''# {title}

{intro}

This is a build-first course: you implement the unfinished `src/learner/api.py`, read the chapters, and run independently authored tests. There is no default reference fallback. Read [the suite guide](../README.md) for the learning order and artifact handoffs.

## Start

```bash
cd ~/Code/ai/courses/{slug}
uv sync --locked
uv run course list
uv run course read 1
uv run course check 1
```

The initial failure is intentional. Implement one small stage, use conceptual hints when stuck, then run the cumulative check. Foundations use CPU and tiny deterministic data; nothing downloads a model or starts paid compute. See [the capstone](course/CAPSTONE.md) for an applied experiment after the acceptance stages.

## Stages

| Stage | Chapter | Implement |
|---|---|---|
{rows}

## Learning loop

`course check N` runs stages 1–N. `--only` is a debugging aid; `--seed 91` varies seeded cases. `course status` reports the latest run and detects source changes. Any skipped test prevents completion. The grader owns expected answers and tests invariants, edge cases, and integration. Local tests are readable, not secure hidden tests. Keep `instructor/` out of your normal reading path; it contains worked references for course maintenance.

Use [EXPERIMENTS.md](course/EXPERIMENTS.md) to record predictions, measurements and limitations. Keep holdout data untouched while making choices. GPU timing and model quality are separate from unit correctness. A passing toy exercise does not establish production readiness.

## Sources

{links}

The curriculum and tests here are original teaching material inspired by these concepts; they are not university assignments or official product certifications.
''')
    write(root/'course/EXPERIMENTS.md','''# Experiment journal

For each stage, record: hypothesis; exact source/seed/environment; input and expected behavior; observed result; explanation; next question.

For training experiments also record dataset identity, train/validation/test split, parameter count, token count, optimizer and schedule, compute budget, training and held-out metrics, and variation across seeds. Select using validation data; use the test set once for the final comparison. Report unsuccessful experiments too.

For systems experiments record hardware, dtype, warmup, synchronization, workload distribution, measurement count and dispersion. Compare identical work. A shorter run that drops requests or tokens is not an optimization.

For release experiments record artifact digests, gate decisions and rollback evidence. Never describe a CPU simulation as a multi-GPU measurement.
''')
    write(root/'instructor/README.md',f'''# Instructor maintenance — contains solutions

Do not read this directory during the default learning path. Acceptance tests never fall back to this code.

```bash
uv run python -m pytest grader --implementation instructor.reference -q
```

This validates the reference against the independent grader. It does not complete learner stages and creates no learner completion receipt. Use a temporary copy for fault injection; never replace learner files. See the suite validation report for checks actually run and hardware limitations.
''')
    return root
