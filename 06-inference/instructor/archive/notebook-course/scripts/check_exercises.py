"""Run learner checks explicitly. --reference validates supplied worked solutions."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exercises.checks import NAMES, check

parser = argparse.ArgumentParser()
parser.add_argument("names", nargs="*", default=NAMES)
parser.add_argument("--reference", action="store_true")
args = parser.parse_args()
module = None
if args.reference:
    from exercises import reference_solutions as module
for name in args.names:
    if name not in NAMES:
        parser.error(f"Unknown exercise {name}; choose from {', '.join(NAMES)}")
    try:
        check(name, module)
    except NotImplementedError as exc:
        print(f"UNFINISHED {name}: {exc}")
        raise SystemExit(1)
