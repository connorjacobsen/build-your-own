# Model engineering learning suite

The learner owns each course's src/. Default to coaching, failure explanation and graduated
conceptual hints. Do not solve learner stages unless explicitly asked for implementation help.
Preserve learner edits. User instructions take precedence.

The instructor owns grader/, instructor/, course contracts and course_cli/. Never weaken an
acceptance gate to make a submission pass. For course maintenance, fix demonstrated contract
or grader defects and explain them. Full solutions stay outside the default learning path.
Local tests are inspectable, not a secure hidden service.

Each course has an independent uv environment. Never import a sibling learner package by
accident. Use tools/lifecycle.py's isolated module loading for cross-course integration.
Do not launch paid remote compute without explicit authorization for the run. Adding or
validating runner code is not authorization to start a GPU job.

Instructor authoring scripts refuse to overwrite learner source. Do not bypass that guard.
