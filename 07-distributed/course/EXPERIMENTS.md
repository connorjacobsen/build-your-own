# Experiment journal

For each stage, record: hypothesis; exact source/seed/environment; input and expected behavior; observed result; explanation; next question.

For training experiments also record dataset identity, train/validation/test split, parameter count, token count, optimizer and schedule, compute budget, training and held-out metrics, and variation across seeds. Select using validation data; use the test set once for the final comparison. Report unsuccessful experiments too.

For systems experiments record hardware, dtype, warmup, synchronization, workload distribution, measurement count and dispersion. Compare identical work. A shorter run that drops requests or tokens is not an optimization.

For release experiments record artifact digests, gate decisions and rollback evidence. Never describe a CPU simulation as a multi-GPU measurement.
