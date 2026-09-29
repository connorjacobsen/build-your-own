# 13. Run an auditable evaluation

The harness should control references and expose only prompts to the predictor. Carrying model and dataset identities into the result makes claims traceable. Every case must produce an output or an explicit failure: dropping failures changes the denominator and often rewards unreliable systems. This harness keeps the execution path intentionally small and inspectable.

## Implementation contract

Work in `src/learner/api.py`. Implement **evaluate**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`evaluate`**

Each case has unique id, prompt:str, acceptable:list[str]. Call predict(prompt) once
per case, preserving order; never expose reference answers to predict. Require nonempty
cases and valid 64-lowercase-hex digests. Prediction must be str, else ValueError.
Return toyeval-v1 dict with model_sha256,dataset_sha256, metrics={accuracy,n}, and
cases=[{id,prediction,score}]. No silent exception swallowing or dropping failed cases.

Exceptions propagate. Case outputs preserve input order, even though later comparisons align by ID.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 13 --only
uv run course check 13
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 13 --level 1`.

## Explain and investigate

What metadata would you add for stochastic generation or a model-based judge?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 evaluation curriculum](https://cs336.stanford.edu/)
- [ARENA evaluations](https://www.arena.education/curriculum)
