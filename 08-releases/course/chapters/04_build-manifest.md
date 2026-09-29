# 04. Describe an explicit bundle

A model release can include weights, tokenizer metadata, configuration and evaluation evidence. Explicit file selection makes that boundary reviewable. A manifest records size and digest for each file and preserves experiment metadata independently of mutable caller dictionaries. Sorted entries make the result deterministic.

## Implementation contract

Work in `src/learner/api.py`. Implement **build_manifest**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`build_manifest`**

Require nonempty unique relative paths. Return format='toyrelease-v1', metadata copied
via canonical JSON roundtrip, files sorted by path with {path,sha256,size_bytes}.
Validate each path using safe_artifact_path. Caller selects explicit files, no recursive scan.

The manifest describes files but does not certify their quality.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 4 --only
uv run course check 4
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 4 --level 1`.

## Explain and investigate

Which files are necessary to reproduce inference, and which are useful only for training?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)
- [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)
- [Python file locks](https://docs.python.org/3/library/fcntl.html)
