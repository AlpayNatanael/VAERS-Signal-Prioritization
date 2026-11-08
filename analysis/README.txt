VAERS NUMERICAL VERIFICATION — 13 OCTOBER 2026

Purpose
Verify the exploratory recurrence findings and history-aware baseline using
fresh chronological model fits and independent numerical calculations.
This is a separate verification package. The poster and original code were
not modified.

Scope
The input is the previously audited quarterly panel (2022Q1–2026Q2).
This run did not repeat ingestion of all raw VAERS reports. It rebuilt
200,220 candidate-quarter rows from the 228,652-row panel, recomputed PRR,
its lower confidence bound, the strong-signal rule, exact next-quarter labels,
and past-only history membership independently. It fit logistic regression
and histogram gradient boosting from scratch in each of nine existing test
windows, with the original settings and seed 42. All five original scores'
AP values were reproduced in all nine quarters. Final fitted predictions
also agree with the saved predictions to within 1e-12.

Status
These data had already been examined. This is exploratory retrospective
verification, not a new untouched validation set, a prospective trial,
a clinical assessment, or evidence of vaccine causality.
History begins in 2022; no earlier crossing means none in available history.

Main checks
{
  "final_n_candidates": 6926,
  "final_n_crossings": 294,
  "returning_crossings": 245,
  "first_observed_crossings": 49,
  "top100_previous_crossing_candidates": 100,
  "top100_total_crossings": 32,
  "top100_returning_crossings": 32,
  "top100_first_crossings": 0,
  "top806_total_crossings": 147,
  "top806_returning_crossings": 142,
  "top806_first_crossings": 5,
  "history_baseline_AP_legacy_float_encoding": 0.13408443464332218,
  "history_baseline_AP_exact_lexicographic": 0.13408443464332218
}

All 437 numerical verification checks passed.

Final candidate quarter: 2026Q1; outcome quarter: 2026Q2
AP means sklearn average precision, not trapezoidal PR-curve area.
quarter          population                  model  n_candidates  n_positive  base_rate  average_precision
 2026Q1        All eligible      Gradient boosting          6926         294   0.042449           0.204706
 2026Q1        All eligible    Logistic regression          6926         294   0.042449           0.156018
 2026Q1        All eligible    Threshold proximity          6926         294   0.042449           0.086218
 2026Q1        All eligible History then proximity          6926         294   0.042449           0.134084
 2026Q1 No earlier crossing      Gradient boosting          4581          49   0.010696           0.060449
 2026Q1 No earlier crossing    Logistic regression          4581          49   0.010696           0.025587
 2026Q1 No earlier crossing    Threshold proximity          4581          49   0.010696           0.011549
 2026Q1 No earlier crossing History then proximity          4581          49   0.010696           0.011549
 2026Q1    Earlier crossing      Gradient boosting          2345         245   0.104478           0.238070
 2026Q1    Earlier crossing    Logistic regression          2345         245   0.104478           0.195345
 2026Q1    Earlier crossing    Threshold proximity          2345         245   0.104478           0.149610
 2026Q1    Earlier crossing History then proximity          2345         245   0.104478           0.149610

Nine-quarter results
These are arithmetic means across quarters, not pooled AP values.
         population                  model  mean_quarter_AP  median_quarter_AP  minimum_quarter_AP  maximum_quarter_AP
       All eligible      Gradient boosting         0.190151           0.198324            0.150373            0.216035
       All eligible    Logistic regression         0.171309           0.160446            0.153862            0.236131
       All eligible    Threshold proximity         0.106575           0.101708            0.086218            0.142654
       All eligible History then proximity         0.147023           0.142387            0.126373            0.187462
No earlier crossing      Gradient boosting         0.067603           0.061815            0.049785            0.103575
No earlier crossing    Logistic regression         0.058001           0.058032            0.025587            0.077945
No earlier crossing    Threshold proximity         0.039998           0.039548            0.011549            0.062941
No earlier crossing History then proximity         0.039998           0.039548            0.011549            0.062941
   Earlier crossing      Gradient boosting         0.235858           0.239313            0.186450            0.264857
   Earlier crossing    Logistic regression         0.215147           0.209730            0.191925            0.279606
   Earlier crossing    Threshold proximity         0.178029           0.173277            0.149610            0.250191
   Earlier crossing History then proximity         0.178029           0.173277            0.149610            0.250191

Candidate coverage in final quarter
quarter  next_quarter_total_strong  already_strong_current  eligible_crossings  absent_current_seen_earlier  absent_current_never_seen_earlier
 2026Q1                        727                     294                 294                          126                                 13
The 139 future-strong pairs absent in the current quarter were not ranked:
126 had appeared previously; 13 had not appeared in the available history.
The claimed recall is conditional on eligible candidates.

Ties and the new baseline
History then proximity means exact lexicographic ordering: first place
pairs with any earlier crossing, then sort by current threshold proximity.
The implementation uses integer ranks to avoid floating-point addition
creating artificial ties. The original exploratory floating-point encoding
and the exact encoding give the same final AP (0.13408443464332218).

In the final overall ranking, GB needs 806 reviews and original threshold
proximity needs 1,531 to obtain 147 positives. Both cutoffs have unique scores.
The history baseline reaches 147 with 992 reviews under the original stable
row order, but the cutoff is inside a tie of 876 pairs containing 53 positives.
Under uniform random tie ordering, expected reviews are 994.2407407407.
The possible extreme counts are 979 through 1,802; these are tie-order bounds,
not a confidence interval. Do not treat 992 as a unique intrinsic workload.
Top-k tables include stable-order results, tie-aware expected results, and
bounds. verify_ties.py independently enumerates all 24 permutations of a
small tied-score example to check these formulas.

Interpretation
GB exceeded threshold proximity and the simple history baseline in all nine
existing test quarters. Among no-earlier-crossing candidates, it exceeded
threshold proximity in all nine quarters and logistic regression in seven.
Its no-earlier-crossing final AP was 0.06045 versus 0.01155 for proximity;
there were only 49 positives among 4,581 candidates. Ranking those candidates
separately gives 4 positives among GB's top 100. The full combined top 100
contains 32 positives, all returning crossings. Thus first-crossing prediction
has some ranking value but remains much harder in absolute terms.
Wins across overlapping historical windows are descriptive, not nine
independent significance tests. This package does not supply dependence-aware
confidence intervals or an independent clinical endpoint.

Reproduce
Use Python 3.12.14 and the dependencies in requirements.txt.
  python -m pip install -r requirements.txt
  python verify_results.py
  python verify_ties.py
Fresh fits took about 35 seconds in the supplied environment (four threads).
The included pickle files are the project's own numerical data; load only
trusted pickle files. CSV outputs are readable without loading pickle.

Contents
  verify_results.py: fresh fits, independent checks, metrics, and figure
  verify_ties.py: exhaustive tie-formula verification
  protocol.json: fixed analysis scope and choices, written before the fits
  source/vaers_pipeline.py: unchanged corrected project module
  inputs/quarterly_panel.pkl: audited input panel
  inputs/reference_final_scored.pkl: reference predictions for agreement check
  inputs/reference_rolling.csv: original quarter-level metrics
  results/: metrics, subgroup results, budget results, coverage, predictions,
             checks, logs, source hashes, and chart
  FILE_SHA256.json: hashes of package files

The graph uses a separate vertical scale in each panel. Positive-outcome
rates differ between populations, so raw AP is not a direct comparison of
intrinsic difficulty without also considering prevalence.
