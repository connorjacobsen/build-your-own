# A release is a decision backed by artifacts

Training produces weights. Evaluation produces evidence. A release policy decides whether particular weights should become active. Keeping those responsibilities separate makes failures easier to diagnose and rollback easier to reason about.

## Three identities

A model filename is a mutable label. A byte digest identifies one serialized object. An active pointer states which digest a consumer should use now. Storing immutable objects and changing a small pointer lets you retain old releases without copying or overwriting their weights.

Digests prove equality with known bytes, not correctness or trustworthiness. A model can have a valid hash and still be poorly trained. An evaluation report can refer to the right model and still use contaminated cases. The registry validates identity and consistency while the experiment report argues for scientific validity.

## Evidence gates

An absolute quality floor prevents accepting a candidate merely because it beats a weak baseline. A regression allowance limits backward movement. A minimum case count prevents tiny samples from being treated as sufficient evidence. These are deterministic policy rules, not replacements for uncertainty estimates or task-specific risk analysis.

For example, baseline accuracy .80 and candidate .78 meet a maximum allowed drop of .02 under the stated inclusive boundary. Floating-point comparison needs an explicit tolerance at such boundaries. Comparing a candidate on easier cases to a baseline on harder ones is invalid even if both have accurate aggregate bookkeeping.

## Atomicity and concurrency

Atomic file replacement prevents a reader from observing half-written JSON. It does not prevent two writers from reading the same old pointer and overwriting each other. A lock plus compare-and-swap makes the expected prior value part of the operation. One concurrent promotion wins; the other receives a conflict and must reconsider its evidence.

Rollback uses the same mechanism. It must not blindly restore a pointer based on stale assumptions, because someone else may have promoted a newer model in the meantime. The local implementation demonstrates these semantics without affecting a public service.

## Monitoring closes the loop

Offline quality and online operation can diverge. Canary routing limits exposure while producing comparable observations. Error rates and tail latency answer different operational questions. Small cohorts require patience, but waiting itself can have costs. Record the policy and the observations that triggered a decision.

This course ends with a local release control loop. Authentication, distributed storage, live traffic routing, observability infrastructure and production rollout orchestration remain explicit extensions rather than implied capabilities of the toy registry.
