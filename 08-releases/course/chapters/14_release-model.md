# 14. Release from verified evidence

The capstone-sized integration combines artifact identity, evidence validation, quality policy, immutable storage and promotion. A failed quality gate must not activate a model or register it as a successful release. Hashing the actual model file binds evaluation metadata to the bytes being promoted. This local workflow provides a concrete foundation for later deployment integrations.

## Implementation contract

Work in `src/learner/api.py`. Implement **release_model**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`release_model`**

Integrated local release: compute model file digest, parse candidate report, validate
its model/dataset identity against actual bytes and baseline, apply release_gate.
If ineligible raise ValueError BEFORE storing artifacts or changing pointer. Otherwise
register model and report, promote with expected_current, return {model_sha256,
report_sha256,pointer}. Report integrity and model quality remain separate claims.

This updates only a local registry pointer. It never deploys to a public endpoint. Failed concurrency checks may leave harmless immutable objects in the store.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 14 --only
uv run course check 14
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 14 --level 1`.

## Explain and investigate

Trace every artifact and decision from training output to a rollback.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)
- [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)
- [Python file locks](https://docs.python.org/3/library/fcntl.html)
