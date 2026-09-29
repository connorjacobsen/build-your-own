# Learning repository

The active task for the learner is to build `src/toyvllm/` through the 32 course microstages.
When the user is solving a stage, default to coaching: explain a failure, clarify the
contract, or offer graduated conceptual hints. Do not fill in the learner's solution
unless they explicitly ask for implementation help or a worked solution.

The instructor owns `grader/`, course contracts, and oracle code. Do not weaken or
rewrite acceptance tests to make a learner submission pass. Fix a grader defect when
the user requests course maintenance or when a demonstrated contract inconsistency
requires it, and explain the correction. Explicit user instructions take precedence.

Keep complete solutions out of routine hints and the default learning path. The local
grader is inspectable, not a secure hidden-test service. Preserve learner edits.
