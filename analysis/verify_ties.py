"""Exhaustive independent check of tied-score evaluation formulas."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from verify_results import topk_stats, workload_stats

frame = pd.DataFrame({
    "score": [2., 1., 1., 1., 1.],
    "emerging_next_q": [1, 1, 0, 1, 0],
    "ever_strong_before": [0, 1, 0, 0, 1],
})
counts, workloads = [], []
for order in itertools.permutations([1, 2, 3, 4]):
    ranked = frame.iloc[[0, *order]]
    counts.append(int(ranked.head(3).emerging_next_q.sum()))
    workloads.append(int(np.flatnonzero(ranked.emerging_next_q.cumsum().to_numpy() >= 2)[0]) + 1)
top = topk_stats(frame, "score", 3)
work = workload_stats(frame, "score", 2 / 3)
assert np.isclose(np.mean(counts), top["tie_expected_positives"])
assert min(counts) == top["tie_min_positives"] and max(counts) == top["tie_max_positives"]
assert np.isclose(np.mean(workloads), work["tie_expected_reviews"])
assert min(workloads) == work["tie_min_reviews"] and max(workloads) == work["tie_max_reviews"]
result = {"all_passed": True, "exhaustive_permutations": len(counts),
          "expected_top3_positives": float(np.mean(counts)),
          "expected_reviews_for_2_positives": float(np.mean(workloads))}
destination = Path(__file__).resolve().parent / "results/tie_permutation_check.json"
destination.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
