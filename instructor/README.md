# Instructor maintenance — not the learner path

This directory contains course authoring sources, validation evidence and deliberately broken
implementations used to test the graders. Do not read authoring/reference code while solving stages.

Each course has its own isolated runtime. Run the suite checks after `uv sync --locked` in each:

```bash
01-autograd/.venv/bin/python tools/validate.py
01-autograd/.venv/bin/python -m pytest instructor/test_suite.py -q
```

`tools/lifecycle.py --reference` tests artifact compatibility using complete references and never
creates learner completion receipts. Hardware gates require explicit separate GPU runs.

Authoring scripts are archival maintenance inputs. Their builders refuse to overwrite learner
source. Do not remove that guard or regenerate over someone’s work. Generated course files are
normal editable source; revise contracts and graders deliberately and keep references consistent.
