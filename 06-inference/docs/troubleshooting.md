# Troubleshooting the build course

## Stage 01 fails immediately

That is expected. The learner package contains unfinished methods. Implement the named function in `src/toyvllm/tokenizer.py` and rerun `uv run course check 1`. No instructor implementation is used as a fallback. Do not edit the test to accept an empty result.

## Imports or the `course` command fail

Run `uv sync --locked` from the repository root. Use `uv run course ...`, which selects the editable project installation and environment. Python 3.11 is selected in `.python-version`. If using a notebook, restart its kernel after changing imported source files.

## Later stages fail while earlier ones are still unfinished

The full command is cumulative. Finish earlier stages first. `--only` is a debugging convenience, not permission to bypass the dependencies or claim completion. Stage 02's model constructor and checkpoint contract are prerequisites for most later tests.

## Checkpoint loading reports missing or unexpected keys

Follow the exact schema in stage 02. The grader fixes weights to separate model math from random initialization. Keep the named linear modules callable; later work-count tests attach hooks to them. Do not work around the error by changing the grader's checkpoint.

## Numerical outputs match, but a work-count test fails

This is intentional. Full-history recomputation can return the same tokens as cached generation, and sequential calls can return the same results as a packed batch. The course requires the optimization itself. Read the failure's observed input/projection sizes and the stage's execution contract.

## Test results change after unrelated random operations

Use request-local generators and fixed model weights. Do not consume a global sampling stream shared by requests. For stochastic cases, repeat with the recorded `--seed` and inspect the earliest differing logits before comparing entire outputs.

## Progress says STALE

A relevant source, grader, CLI, or environment file changed. Rerun the cumulative gate. Receipts describe a particular source snapshot; they are not permanently granted badges. Example-only, single-stage, and skipped runs do not produce a cumulative completion claim.

## Modal command is missing or authentication fails

Install the optional group with `uv sync --locked --group gpu`, then use `uv run --group gpu modal setup` for your own account. The default local environment does not require credentials. No credentials are placed in the course source. Remote GPU calls are explicit and may incur account charges.

## GPU tests fail on this Mac

Apple MPS is not CUDA. Keep developing CPU stages locally and use the Modal command or a CUDA machine for the final hardware gate. The course deliberately refuses to count skipped GPU tests as success. You can run stage 10's CPU-only timing tests separately as described in the grading guide.

## An engine test times out

Check whether the waiting head can ever fit, whether running requests have enough reserved capacity to finish, and whether terminal requests release ownership and queue membership. Do not raise the timeout to conceal a progress bug. Tests have bounded waits and the CLI has a whole-run limit.

## Server queue or cancellation tests fail

The injected engine in the test can be paused deliberately. Respect it instead of constructing a second engine. Clean up pending-call accounting when a route is cancelled as well as when it returns normally. An async route does not make synchronous compute nonblocking.

## Generated text is nonsense

The tiny model has random weights. Token IDs, logits, cache invariants, and execution behavior are the correctness signals. Pretrained checkpoint loading is a separate extension; the byte tokenizer and positional conventions are not automatically compatible with external models.

## Where did the old notebooks go?

The previous course, scripts, documents, exercises, and notebook edits are preserved under `instructor/archive/notebook-course/`. New notebooks are optional visualization companions. Archived commands assume the old layout and are not the current learner workflow.
