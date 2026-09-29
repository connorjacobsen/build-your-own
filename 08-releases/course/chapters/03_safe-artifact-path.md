# 03. Bound artifact paths

A bundle should refer only to its own files. Checking string prefixes is insufficient because paths can contain parent traversal or symlinks. Resolve the path and verify containment in the resolved bundle root. This also prevents a manifest from accidentally including a local file outside the intended model package.

## Implementation contract

Work in `src/learner/api.py`. Implement **safe_artifact_path**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`safe_artifact_path`**

Return resolved existing regular file inside resolved root. Reject absolute paths,
empty paths, any '..' component and symlink resolution outside root, with ValueError.
Internal symlinks to regular files are permitted. Missing/nonfiles raise ValueError.

Internal symlinks are allowed, escaping symlinks are not. Existing regular files only.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 3 --only
uv run course check 3
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 3 --level 1`.

## Explain and investigate

Why does a path starting with the text of a directory name not establish containment?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)
- [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)
- [Python file locks](https://docs.python.org/3/library/fcntl.html)
