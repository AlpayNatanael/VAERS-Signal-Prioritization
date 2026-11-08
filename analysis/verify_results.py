#!/usr/bin/env python3
"""Reproduce the exploratory VAERS checks with fresh chronological model fits.

This starts from the previously audited quarterly panel, not from raw reports.
It independently checks scores, outcomes, history flags, AP, and tied rankings.
Run: python verify_results.py
"""
from __future__ import annotations

import os
for setting in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[setting] = "4"
os.environ["MPLBACKEND"] = "Agg"

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
import vaers_pipeline as project

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
CHECKS = []
METRICS = []
TOPK = []
WORKLOAD = []
COVERAGE = []
REFITS = []
QUARTERS = json.loads((ROOT / "protocol.json").read_text())["quarters"]
SCORES = {
    "Gradient boosting": "score_hgb",
    "Logistic regression": "score_lr",
    "Threshold proximity": "score_proximity",
    "History then proximity": "score_history",
    "Previous crossing then proximity": "score_prev_crossing",
    "Current count": "score_count",
    "Previous count": "score_previous_count",
    "PRR": "score_prr",
    "Shrinkage proxy": "score_gp",
}


def log(message):
    print(message, flush=True)
    with (RESULTS / "execution.log").open("a") as f:
        f.write(message + "\n")


def check(name, condition, detail=""):
    CHECKS.append({"check": name, "passed": bool(condition), "detail": str(detail)})
    if not condition:
        pd.DataFrame(CHECKS).to_csv(RESULTS / "verification_checks.csv", index=False)
        raise AssertionError(f"{name}: {detail}")


def independent_ap(y, score):
    """AP evaluated at the end of each distinct-score block (ties grouped)."""
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    yy = np.asarray(y, dtype=int)[order]
    ss = np.asarray(score, dtype=float)[order]
    if yy.sum() == 0:
        return np.nan
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    cumulative = np.cumsum(yy)[ends]
    increments = np.diff(np.r_[0, cumulative])
    return float(np.sum(increments * cumulative / (ends + 1)) / yy.sum())


def topk_stats(frame, column, k):
    ranked = frame.sort_values(column, ascending=False, kind="mergesort")
    k = min(k, len(ranked))
    cut = ranked.iloc[k - 1][column]
    above = ranked[ranked[column] > cut]
    tied = ranked[ranked[column] == cut]
    remaining = k - len(above)
    p = int(tied.emerging_next_q.sum())
    a = int(above.emerging_next_q.sum())
    stable = ranked.head(k)
    return {
        "k": k,
        "stable_order_positives": int(stable.emerging_next_q.sum()),
        "stable_order_first_crossings": int((stable.emerging_next_q * (1 - stable.ever_strong_before)).sum()),
        "stable_order_returning_crossings": int((stable.emerging_next_q * stable.ever_strong_before).sum()),
        "tie_expected_positives": a + remaining * p / len(tied),
        "tie_min_positives": a + max(0, remaining - (len(tied) - p)),
        "tie_max_positives": a + min(remaining, p),
        "score_tie_size_at_k": len(tied),
    }


def workload_stats(frame, column, fraction):
    ranked = frame.sort_values(column, ascending=False, kind="mergesort")
    target = int(np.ceil(fraction * ranked.emerging_next_q.sum()))
    cumulative = ranked.emerging_next_q.cumsum().to_numpy()
    stable_k = int(np.flatnonzero(cumulative >= target)[0]) + 1
    cut = ranked.iloc[stable_k - 1][column]
    above = ranked[ranked[column] > cut]
    tied = ranked[ranked[column] == cut]
    p = int(tied.emerging_next_q.sum())
    remaining = target - int(above.emerging_next_q.sum())
    # Expected position of the r-th success under a uniform ordering of the tie.
    expected_k = len(above) + remaining * (len(tied) + 1) / (p + 1)
    return {
        "capture_fraction": fraction,
        "target_positives": target,
        "stable_order_reviews": stable_k,
        "tie_expected_reviews": expected_k,
        "tie_min_reviews": len(above) + remaining,
        "tie_max_reviews": len(above) + len(tied) - p + remaining,
        "score_tie_size_at_target": len(tied),
    }


def independent_panel_check(panel):
    check("panel_keys_unique", not panel.duplicated(["vax_type", "meddra_pt", "q_int"]).any())
    check("complete_panel_ends_2026Q2", panel.q_int.max() == 8106)
    a, b, c, d = [panel[col].to_numpy(float) + 0.5 for col in ("a", "b", "c", "d")]
    ratio = (a / (a + b)) / (c / (c + d))
    se = np.sqrt(np.maximum(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d), 1e-12))
    lower = np.exp(np.log(ratio) - 1.96 * se)
    strong = (panel.a.to_numpy() >= 3) & (ratio >= 2) & (lower > 1)
    proximity = np.minimum.reduce([panel.a.to_numpy(float) / 3, ratio / 2, lower])
    for name, independent, stored in [
        ("panel_prr", ratio, panel.prr),
        ("panel_prr_lower_bound", lower, panel.prr_lcb95),
        ("panel_proximity", proximity, panel.threshold_proximity),
    ]:
        check(name, np.allclose(independent, stored, rtol=1e-12, atol=1e-12))
    check("panel_strong_rule", np.array_equal(strong.astype(int), panel.strong_signal.to_numpy()))
    return pd.Series(strong.astype(int), index=panel.index)


def attach_independent_history(frame, panel, quarter_int):
    f = frame.copy()
    keys = ["vax_type", "meddra_pt"]
    idx = pd.MultiIndex.from_frame(f[keys])
    # Explicitly truncate history before the current candidate quarter.
    past = panel[panel.q_int < quarter_int]
    prior = past.groupby(keys).strong_signal.max()
    f["ever_strong_before"] = idx.map(prior).fillna(0).astype(int)
    # Independent construction: earliest strong quarter compared with cutoff.
    first_strong = panel[panel.strong_signal.eq(1)].groupby(keys).q_int.min()
    earliest = idx.map(first_strong).to_numpy(float)
    check(f"{quarter_int}_history_two_methods", np.array_equal(f.ever_strong_before, earliest < quarter_int))
    previous = panel[panel.q_int.eq(quarter_int - 1)].set_index(keys)
    previous_count = idx.map(previous.a).fillna(0).to_numpy(float)
    previous_strong = idx.map(previous.strong_signal).fillna(0).to_numpy(int)
    check(f"{quarter_int}_previous_counts", np.array_equal(previous_count, f.a_prev.to_numpy()))
    check(f"{quarter_int}_previous_strong", np.array_equal(previous_strong, f.strong_prev.to_numpy()))
    nxt = panel[panel.q_int.eq(quarter_int + 1)].set_index(keys)
    next_label = idx.map(nxt.strong_signal).fillna(0).to_numpy(int)
    check(f"{quarter_int}_future_labels", np.array_equal(next_label, f.emerging_next_q.to_numpy()))
    check(f"{quarter_int}_current_eligibility", bool((f.strong_signal.eq(0) & f.a.ge(1)).all()))
    check(f"{quarter_int}_proximity_below_one", bool(f.score_proximity.le(1).all()))
    # Integer encoding gives exact lexicographic ordering. Adding 2 to a float
    # can merge extremely close proximity values into artificial score ties.
    proximity_rank, unique_values = pd.factorize(f.score_proximity, sort=True)
    gap = len(unique_values) + 1
    f["score_history"] = gap * f.ever_strong_before + proximity_rank
    f["score_prev_crossing"] = gap * previous_strong + proximity_rank
    check(f"{quarter_int}_lexicographic_order",
          f.sort_values("score_history", ascending=False, kind="mergesort").index.equals(
              f.sort_values(["ever_strong_before", "score_proximity"], ascending=[False, False], kind="mergesort").index))
    f["score_count"] = f.a.to_numpy(float)
    f["score_previous_count"] = previous_count
    return f


def summarize_coverage(panel, quarter, quarter_int, scored):
    keys = ["vax_type", "meddra_pt"]
    current = panel[panel.q_int.eq(quarter_int)].set_index(keys)
    future = panel[panel.q_int.eq(quarter_int + 1) & panel.strong_signal.eq(1)].copy()
    idx = pd.MultiIndex.from_frame(future[keys])
    current_a = idx.map(current.a)
    current_strong = idx.map(current.strong_signal).fillna(0).to_numpy(int)
    past_pairs = pd.MultiIndex.from_frame(panel[panel.q_int < quarter_int][keys].drop_duplicates())
    absent = current_a.isna()
    categories = {
        "already_strong_current": int(current_strong.sum()),
        "eligible_crossings": int(((~absent) & (current_strong == 0)).sum()),
        "absent_current_seen_earlier": int((absent & idx.isin(past_pairs)).sum()),
        "absent_current_never_seen_earlier": int((absent & ~idx.isin(past_pairs)).sum()),
    }
    check(f"{quarter}_coverage_partition", sum(categories.values()) == len(future))
    check(f"{quarter}_coverage_matches_targets", categories["eligible_crossings"] == int(scored.emerging_next_q.sum()))
    COVERAGE.append({"quarter": quarter, "next_quarter_total_strong": len(future), **categories})


def evaluate(frame, quarter):
    populations = {
        "All eligible": frame,
        "No earlier crossing": frame[frame.ever_strong_before.eq(0)],
        "Earlier crossing": frame[frame.ever_strong_before.eq(1)],
    }
    for group, f in populations.items():
        positives = int(f.emerging_next_q.sum())
        check(f"{quarter}_{group}_both_classes", 0 < positives < len(f))
        for model, column in SCORES.items():
            ap = independent_ap(f.emerging_next_q, f[column])
            ref_ap = average_precision_score(f.emerging_next_q, f[column])
            check(f"{quarter}_{group}_{model}_AP", np.isclose(ap, ref_ap, atol=1e-12, rtol=1e-12))
            METRICS.append({"quarter": quarter, "population": group, "model": model,
                            "n_candidates": len(f), "n_positive": positives,
                            "base_rate": positives / len(f), "average_precision": ap})
            for k in (100, 250, 500):
                TOPK.append({"quarter": quarter, "population": group, "model": model,
                             **topk_stats(f, column, k)})
            for fraction in (0.5, 0.8):
                WORKLOAD.append({"quarter": quarter, "population": group, "model": model,
                                 **workload_stats(f, column, fraction)})


def make_figure(metrics):
    colors = {"Gradient boosting": "#006D8F", "Logistic regression": "#995600",
              "History then proximity": "#7A5195", "Threshold proximity": "#727B86"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), layout="constrained")
    for ax, population in zip(axes, ("All eligible", "No earlier crossing", "Earlier crossing")):
        sub = metrics[metrics.population.eq(population)]
        for model, color in colors.items():
            # History+proximity equals proximity within each history subgroup.
            if population != "All eligible" and model == "History then proximity":
                continue
            g = sub[sub.model.eq(model)].set_index("quarter").loc[QUARTERS]
            ax.plot(np.arange(len(QUARTERS)), g.average_precision, marker="o", ms=4,
                    lw=2, color=color, label=model)
        g = sub[sub.model.eq("Gradient boosting")].set_index("quarter").loc[QUARTERS]
        ax.plot(np.arange(len(QUARTERS)), g.base_rate, color="#BAC1C8", ls="--", label="Positive-outcome rate")
        ax.set_title(population, fontsize=12, weight="bold")
        ax.set_xticks(np.arange(len(QUARTERS)), [q.replace("Q", "\nQ") for q in QUARTERS], fontsize=8)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Candidate quarter", fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Average precision (higher is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=9)
    fig.suptitle("Fresh chronological fits: recurrence and first observed crossings", fontsize=15, weight="bold")
    fig.savefig(RESULTS / "VAERS_Numerical_Verification.png", dpi=180)
    plt.close(fig)


def main():
    (RESULTS / "execution.log").write_text("")
    started = time.monotonic()
    panel = pd.read_pickle(ROOT / "inputs/quarterly_panel.pkl")
    independent_panel_check(panel)
    candidates = project.build_emerging_signal_candidates(panel)
    check("candidate_count", len(candidates) == 200220, len(candidates))
    reference_rolling = pd.read_csv(ROOT / "inputs/reference_rolling.csv").set_index("test_quarter")
    reference_final = pd.read_pickle(ROOT / "inputs/reference_final_scored.pkl")
    check("project_features_exclude_future_columns", not any("next" in c for c in project.FEATURES))
    log(f"Panel: {len(panel):,} rows; rebuilt candidates: {len(candidates):,}; models are fit from scratch.")
    final = None
    for quarter in QUARTERS:
        t = project.quarter_to_int(quarter)
        fit = candidates[candidates.q_int < t - 1].copy()
        cal = candidates[candidates.q_int.eq(t - 1)].copy()
        test = candidates[candidates.q_int.eq(t)].copy()
        check(f"{quarter}_ordered_splits", int(fit.q_int.max()) < int(cal.q_int.min()) < int(test.q_int.min()))
        window_started = time.monotonic()
        with threadpool_limits(limits=4):
            bundle = project.fit_model_bundle(fit, cal)
            scored = project.add_model_scores(test, bundle)
        check(f"{quarter}_training_only_age_fill", bundle.age_fill_value == float(fit.mean_age.median()))
        scored = attach_independent_history(scored, panel, t)
        summarize_coverage(panel, quarter, t, scored)
        for label, col in [("Gradient Boosting", "score_hgb"), ("Logistic Regression", "score_lr"),
                           ("Threshold proximity", "score_proximity"), ("PRR", "score_prr"),
                           ("Gamma-Poisson shrinkage", "score_gp")]:
            ap = independent_ap(scored.emerging_next_q, scored[col])
            old = reference_rolling.loc[quarter, "AUPRC_" + label]
            check(f"{quarter}_{label}_matches_existing", abs(ap - old) < 1e-10, f"fresh={ap}; original={old}")
        evaluate(scored, quarter)
        columns = ["quarter", "vax_type", "meddra_pt", "emerging_next_q", "ever_strong_before"] + list(SCORES.values())
        scored[columns].to_csv(RESULTS / f"predictions_{quarter}.csv.gz", index=False)
        REFITS.append({"quarter": quarter, "fit_n": len(fit), "calibration_n": len(cal), "test_n": len(test),
                       "age_fill": bundle.age_fill_value, "seconds": time.monotonic() - window_started})
        ap_gb = independent_ap(scored.emerging_next_q, scored.score_hgb)
        ap_hist = independent_ap(scored.emerging_next_q, scored.score_history)
        log(f"{quarter}: GB AP={ap_gb:.6f}; history baseline AP={ap_hist:.6f}; {REFITS[-1]['seconds']:.1f}s; original metrics reproduced.")
        if quarter == "2026Q1":
            final = scored
            keys = ["vax_type", "meddra_pt", "q_int"]
            fresh = final.set_index(keys).sort_index()
            old = reference_final.set_index(keys).sort_index()
            check("final_prediction_keys", fresh.index.equals(old.index))
            for col in ("score_hgb", "score_lr"):
                diff = float(np.max(np.abs(fresh[col].to_numpy() - old[col].to_numpy())))
                check(f"final_{col}_predictions_match", diff < 1e-12, diff)

    assert final is not None
    metrics = pd.DataFrame(METRICS)
    topk = pd.DataFrame(TOPK)
    workload = pd.DataFrame(WORKLOAD)
    for name, frame in [("all_metrics", metrics), ("topk_yield", topk), ("capture_workload", workload),
                        ("candidate_coverage", pd.DataFrame(COVERAGE)), ("fresh_fit_windows", pd.DataFrame(REFITS))]:
        frame.to_csv(RESULTS / f"{name}.csv", index=False)

    ranked = final.sort_values("score_hgb", ascending=False, kind="mergesort")
    top100 = ranked.head(100)
    top806 = ranked.head(806)
    claims = {
        "final_n_candidates": len(final),
        "final_n_crossings": int(final.emerging_next_q.sum()),
        "returning_crossings": int((final.emerging_next_q * final.ever_strong_before).sum()),
        "first_observed_crossings": int((final.emerging_next_q * (1 - final.ever_strong_before)).sum()),
        "top100_previous_crossing_candidates": int(top100.ever_strong_before.sum()),
        "top100_total_crossings": int(top100.emerging_next_q.sum()),
        "top100_returning_crossings": int((top100.emerging_next_q * top100.ever_strong_before).sum()),
        "top100_first_crossings": int((top100.emerging_next_q * (1 - top100.ever_strong_before)).sum()),
        "top806_total_crossings": int(top806.emerging_next_q.sum()),
        "top806_returning_crossings": int((top806.emerging_next_q * top806.ever_strong_before).sum()),
        "top806_first_crossings": int((top806.emerging_next_q * (1 - top806.ever_strong_before)).sum()),
    }
    expected = dict(final_n_candidates=6926, final_n_crossings=294, returning_crossings=245,
                    first_observed_crossings=49, top100_previous_crossing_candidates=100,
                    top100_total_crossings=32, top100_returning_crossings=32, top100_first_crossings=0,
                    top806_total_crossings=147, top806_returning_crossings=142, top806_first_crossings=5)
    for name, expected_value in expected.items():
        check(name, claims[name] == expected_value, claims[name])
    legacy_ap = independent_ap(final.emerging_next_q, 2 * final.ever_strong_before + final.score_proximity)
    exact_ap = independent_ap(final.emerging_next_q, final.score_history)
    check("legacy_exploratory_history_AP", abs(legacy_ap - 0.13408443464332218) < 1e-12, legacy_ap)
    claims["history_baseline_AP_legacy_float_encoding"] = legacy_ap
    claims["history_baseline_AP_exact_lexicographic"] = exact_ap
    summary_rows = []
    wins = []
    for group, sub in metrics.groupby("population", sort=False):
        pivot = sub.pivot(index="quarter", columns="model", values="average_precision")
        for model in SCORES:
            summary_rows.append({"population": group, "model": model,
                                 "mean_quarter_AP": float(pivot[model].mean()),
                                 "median_quarter_AP": float(pivot[model].median()),
                                 "minimum_quarter_AP": float(pivot[model].min()),
                                 "maximum_quarter_AP": float(pivot[model].max())})
        for baseline in ["Threshold proximity", "History then proximity", "Logistic regression"]:
            difference = pivot["Gradient boosting"] - pivot[baseline]
            wins.append({"population": group, "comparison": "Gradient boosting vs " + baseline,
                         "quarters_higher_AP": int((difference > 1e-12).sum()),
                         "quarters_lower_AP": int((difference < -1e-12).sum()),
                         "mean_AP_difference": float(difference.mean())})
    pd.DataFrame(summary_rows).to_csv(RESULTS / "quarterly_summary.csv", index=False)
    pd.DataFrame(wins).to_csv(RESULTS / "quarterly_comparisons.csv", index=False)
    final_metrics = metrics[metrics.quarter.eq("2026Q1")]
    final_metrics.to_csv(RESULTS / "final_subgroup_metrics.csv", index=False)
    make_figure(metrics)
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
                "matplotlib": matplotlib.__version__, "joblib": joblib.__version__}
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [ROOT / "inputs/quarterly_panel.pkl", ROOT / "inputs/reference_final_scored.pkl",
                        ROOT / "inputs/reference_rolling.csv", ROOT / "source/vaers_pipeline.py",
                        ROOT / "verify_results.py", ROOT / "protocol.json"]}
    pd.DataFrame(CHECKS).to_csv(RESULTS / "verification_checks.csv", index=False)
    summary = {"all_checks_passed": all(c["passed"] for c in CHECKS), "checks": len(CHECKS),
               "claims": claims, "quarterly_comparisons": wins, "versions": versions,
               "source_hashes": hashes, "elapsed_seconds": time.monotonic() - started,
               "scope": "Fresh model fits from previously audited quarterly panel. Exploratory retrospective checks; no new raw-data rebuild, real-time snapshots, expert review, or causal validation."}
    (RESULTS / "verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "requirements.txt").write_text("\n".join(f"{name}=={version}" for name, version in versions.items() if name != "python") + "\nthreadpoolctl>=3\n")
    log(f"PASS: {len(CHECKS)} checks. Total time {summary['elapsed_seconds']:.1f}s.")
    log(json.dumps({"claims": claims, "quarterly_comparisons": wins}, indent=2))


if __name__ == "__main__":
    main()
