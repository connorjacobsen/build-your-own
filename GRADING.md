# Grading and feedback

Every stage in `course/stages.json` points to exact pytest node IDs. Cumulative checks include the boundary check and all gates through the selected stage. Inference reuses all of its original acceptance tests without weakening assertions. Its old broad milestones are now a textbook organization, not completion units.

The runner always selects learner code. It removes ambient pytest options/plugins, loads the timeout plugin explicitly, fixes a case seed and limits the total subprocess runtime. A receipt includes timestamp, counts, test selection, seed, environment and a SHA256 fingerprint of source, grader, CLI, project metadata, lockfile and stage catalog. Changing these inputs makes an earlier receipt stale. A skip, failure, collection error, empty selection or timed-out run cannot establish completion. An isolated gate cannot establish cumulative completion.

Receipts are convenience records, not tamper-proof credentials. Local instructors and learners can inspect or edit the grader. Boundary checks catch accidental direct instructor imports, not every possible attempt to bypass an exercise. The intended use is deliberate practice with independent expected behavior.

Course maintenance uses `uv run python -m pytest grader --implementation instructor.reference -m "not gpu"` (inference uses `grader.oracle`). This validates the course and creates no learner achievement. `tools/validate.py` runs that maintenance sweep. Fault-injection checks in `instructor/test_suite.py` demonstrate that representative plausible defects are rejected. Read [VALIDATION.md](VALIDATION.md) for work actually verified on this host.

Most functions have narrow input domains. Do not infer support for arbitrary shapes, malformed schemas, distributed topology changes or concurrent writes beyond the stated contract. Chapter contracts and scaffold docstrings are normative. If a test contradicts a contract, treat it as a course defect and explain the correction rather than learning an undocumented special case.

For learning: predict a small example; implement the contract; read the first failure; ask for hint level 1; isolate the broken invariant; run the cumulative gate. After passing, complete the chapter investigation and capstone evidence. Passing tests is necessary for mechanics but insufficient for scientific claims.
