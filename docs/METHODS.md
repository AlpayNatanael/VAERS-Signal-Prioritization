# Methods and evaluation

This document describes the implemented retrospective experiment. Model settings
and evaluation windows are also recorded in [protocol.json](../analysis/protocol.json).

## Unit, candidate set, and outcome

An observation is one vaccine type and MedDRA event term in one receipt quarter.
The raw pipeline retains primary report sequences (`ORDER=1`), joins the three
VAERS tables by `(VAERS_ID, ORDER)`, and counts a report once per pair. The
available history begins in 2022; the last included receipt quarter is 2026Q2.

The quarterly 2-by-2 counts are:

| Count | Definition |
|---|---|
| `a` | Reports containing both the vaccine type and event term |
| `b` | Reports containing the vaccine type without the event term |
| `c` | Reports containing the event term without the vaccine type |
| `d` | Remaining reports in the analysis population |

Add 0.5 to every cell when computing ratios. With corrected counts denoted
`A`, `B`, `C`, and `D`, the proportional reporting ratio is
`PRR = [A / (A+B)] / [C / (C+D)]`. Its lower 95% bound is
`exp(log(PRR) - 1.96 * sqrt(1/A - 1/(A+B) + 1/C - 1/(C+D)))`.

A pair meets the screening rule when its **uncorrected** count is at least 3,
its corrected PRR is at least 2, and the lower bound is greater than 1.
Candidates are observed pairs that do not meet the rule in quarter `t`.
The binary outcome is whether the pair meets the rule in exact quarter `t+1`.
A pair absent in `t+1` remains a negative; it is not dropped from evaluation.
The final available quarter cannot itself be a candidate quarter because its
next-quarter outcome is not observed.

## Features and models

The corrected pipeline defines 31 features: five volume/expected-count features,
five current statistical scores, ten report-context features, and eleven
history features. Counts and positive ratios use `log1p` transforms. History
matches exact calendar quarters rather than the previous available observation.
`new_pair` means absent in the previous quarter, not first-ever observed.

Missing mean age is filled using the median learned only in the fitting window;
that value is then reused for calibration and test rows. Missing female share
uses a fixed value of 0.5. Missingness indicators are included.

| Model | Fixed settings |
|---|---|
| Logistic regression | StandardScaler fitted on training data; balanced class weights; `lbfgs`; `max_iter=3000`; default L2 regularization |
| Histogram gradient boosting | Balanced class weights; learning rate 0.06; 250 iterations; 15 leaf nodes; minimum 50 samples per leaf; L2 regularization 1.0; early stopping disabled |

Both models use seed 42 and sigmoid probability calibration on a distinct later
quarter. No new hyperparameter search is part of this release.

## Chronological evaluation

For each candidate test quarter `t`, fitting uses candidate quarters strictly
before `t-1`. Calibration uses candidates from `t-1`, whose outcomes occur in
`t`. The model ranks candidates in `t` and is evaluated against outcomes in
`t+1`. Thus the intended forecast is after quarter `t` ends.

There are nine test quarters, 2024Q1 through 2026Q1. In the final window, fitting
ends with 2025Q3 candidates, calibration uses 2025Q4 candidates, and 2026Q1
candidates are evaluated against 2026Q2. Receipt dates do **not** establish when
reports became publicly available. This is not a reconstruction of historical
real-time availability.

## Baselines, metrics, and ties

Threshold proximity is `min(a/3, PRR/2, PRR_lower_bound/1)`. The history baseline
places pairs with any earlier observed crossing ahead of the rest, then orders
by current proximity. History is strictly before the candidate quarter.
Verification also evaluates current/previous counts, a previous-quarter
crossing rule, PRR, and the project's simple shrinkage proxy. That proxy is not
a full MGPS/EBGM implementation.

Average precision (AP) is computed at distinct score cutoffs and independently
cross-checked against scikit-learn. It is not trapezoidal area under a plotted
precision-recall curve. Some legacy column names say `AUPRC` but contain AP.
Top-k yield and pairs examined to reach 50%/80% capture are also reported.

For a tied block of `n` candidates containing `p` positives, if `r` additional
positives are needed to reach a target, the expected position of the `r`-th
positive under uniform random ordering is `r * (n+1) / (p+1)`. Add the number
of candidates above that block. Bounds from the best/worst ordering are tie
bounds, not confidence intervals. Expected capture at a fixed budget and
expected stopping time are different quantities.

## Interpretation

The final population contains 6,926 candidates and 294 positives. Of those
positives, 245 had crossed before and 49 had no earlier crossing in the available
history. The overall model therefore emphasizes recurrence. The notebook also
reports first-observed-crossing performance and coverage outside the eligible
candidate set.

These periods have already been examined, and repeated pairs and overlapping
training data create dependence. The repository does not demonstrate causal
vaccine effects, clinical usefulness, staff-time savings, or independent
confirmatory significance. Later untouched periods, stronger temporal baselines,
dependence-aware uncertainty, actual data-availability dates, and expert review
are priorities for further work. Ensemble and simulation studies are not yet
implemented here.
