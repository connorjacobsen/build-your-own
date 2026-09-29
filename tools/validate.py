"""Instructor validation; never records learner completion or starts paid compute."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    report = []
    failed = False
    destination = ROOT / "instructor/validation"
    destination.mkdir(parents=True, exist_ok=True)
    for course in sorted(ROOT.glob("[0-9][0-9]-*")):
        python = course / ".venv/bin/python"
        if not python.exists():
            raise SystemExit(f"Run uv sync --locked in {course}")
        impl = "grader.oracle" if course.name == "06-inference" else "instructor.reference"
        junit = destination / f"{course.name}.xml"
        command = [
            str(python),
            "-m",
            "pytest",
            "grader",
            "--implementation",
            impl,
            "--case-seed",
            str(args.seed),
            "-m",
            "not gpu",
            "-q",
            f"--junitxml={junit}",
        ]
        result = subprocess.run(command, cwd=course, timeout=240)
        counts = dict(passed=0, failed=0, skipped=0)
        if junit.exists():
            for case in ET.parse(junit).iter("testcase"):
                kind = (
                    "failed"
                    if case.find("failure") is not None or case.find("error") is not None
                    else "skipped"
                    if case.find("skipped") is not None
                    else "passed"
                )
                counts[kind] += 1
        report.append(
            dict(course=course.name, seed=args.seed, exit_code=result.returncode, **counts)
        )
        failed |= result.returncode != 0 or counts["skipped"] > 0
    (destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
