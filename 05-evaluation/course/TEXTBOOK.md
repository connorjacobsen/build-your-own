# Evaluation is an experimental protocol

A benchmark score is produced by a model, a dataset, a prompt or tokenization protocol, a decoding strategy and a metric. Changing any component can change the result. Exact artifact identities and per-case outputs make the claim inspectable.

## Mechanics versus meaning

A test can prove that continuation scoring uses the correct token offset. It cannot prove that the chosen benchmark represents real user needs. A release report can be internally consistent yet measure the wrong capability. Keep implementation checks and validity arguments separate.

For three scored tokens with probabilities .5, .25 and .125, the summed negative log likelihood is -log(.5*.25*.125). Perplexity is exp(NLL/3), the reciprocal of the geometric mean token probability. It is not the arithmetic mean of reciprocal probabilities, and per-document perplexities should not be averaged without returning to sums and counts.

## Paired comparisons

Suppose two models run on the same cases. Each case gives a score difference d_i. The mean difference estimates the observed improvement. A paired bootstrap samples case indices and averages their differences repeatedly. A confidence interval summarizes variability under resampling assumptions; it does not correct label mistakes, contamination or an unrepresentative task population.

Keeping IDs aligned is essential. If outputs are sorted by completion time, row 20 in one file may not refer to row 20 in another. Missing cases must be explained. Dropping failures or timeouts can turn reliability defects into higher scores.

## Slices and calibration

A system with 90% accuracy on a dominant group and 0% on a rare group can look acceptable in aggregate. Report counts, micro and macro scores, and task-relevant slices. Very small slices need cautious interpretation because uncertainty can be large.

Calibration asks whether confidence matches observed correctness. It differs from accuracy and discrimination. A model can be calibrated by always reporting the base success rate, yet provide little useful differentiation between easy and hard cases. Binned ECE depends on bin definitions, so preserve that protocol.

## Repeated experimentation

Choose metrics and a validation selection rule before comparing many variants. Use a final holdout for the eventual claim, and report the experimentation budget. A single lucky seed or an improvement selected from dozens of attempts should not be presented as a stable effect. In this suite the small acceptance datasets exercise software behavior; capstones ask for evidence beyond those fixtures.
