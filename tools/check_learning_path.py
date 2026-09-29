"""Instructor smoke check for unfinished courses and preserved inference progress."""

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
results = []
for course in sorted(ROOT.glob("[0-9][0-9]-*")):
    python = course / ".venv/bin/python"

    def run(*args):
        return subprocess.run(
            [str(python), "-m", "course_cli.main", *args],
            cwd=course,
            capture_output=True,
            text=True,
            timeout=30,
        )

    catalog = json.loads((course / "course/stages.json").read_text())
    listed, chapter, hint = run("list"), run("read", "1"), run("hint", "1")
    assert listed.returncode == chapter.returncode == hint.returncode == 0
    assert catalog[-1]["title"] in listed.stdout
    number = 1
    checked = run("check", str(number))
    # The original learner has tokenizer work. Preserve and verify it rather than assuming stubs.
    if course.name == "06-inference":
        while checked.returncode == 0 and number < 30:
            number += 1
            checked = run("check", str(number))
        assert checked.returncode != 0, (
            "Update smoke expectations if all CPU gates are now complete"
        )
    else:
        assert checked.returncode != 0 and "NotImplementedError" in checked.stdout, checked.stdout
    record = json.loads((course / ".course/latest.json").read_text())
    assert record["failed"] == 1 and not record["complete"] and record["skipped"] == 0
    status = run("status")
    assert status.returncode == 0 and "Source unchanged" in status.stdout
    (ROOT / "instructor/validation" / f"{course.name}-first-incomplete-gate.txt").write_text(
        checked.stdout + checked.stderr
    )
    results.append(
        dict(
            course=course.name,
            stages=len(catalog),
            first_incomplete_gate=number,
            preserved_prior_gates=number - 1,
            receipt_complete=record["complete"],
        )
    )
(ROOT / "instructor/validation/learning-path.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
