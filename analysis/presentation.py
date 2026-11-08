"""Readable tables and figures for the VAERS presentation notebook.

The scientific pipeline and independent verification scripts are preserved.
This module runs those scripts and builds presentation figures from their
fresh prediction files. It does not change the models or select new features.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
MODELS = ["Gradient boosting", "Logistic regression", "History then proximity", "Threshold proximity"]
SCORES = dict(zip(MODELS, ["score_hgb", "score_lr", "score_history", "score_proximity"]))
COLORS = {
    "Gradient boosting": "#0067A8",
    "Logistic regression": "#178078",
    "History then proximity": "#B78B17",
    "Threshold proximity": "#778397",
}
SHORT = {
    "Gradient boosting": "Gradient boosting",
    "Logistic regression": "Logistic regression",
    "History then proximity": "History + proximity",
    "Threshold proximity": "Threshold proximity",
}
NAVY, MUTED, PALE = "#17364D", "#536477", "#E5EBF0"


def _configure_plots():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelcolor": NAVY, "text.color": NAVY,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#C9D3DE", "axes.titleweight": "semibold",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "figure.dpi": 120,
    })


def run_analysis():
    """Refit both models in all nine windows, then independently verify outputs."""
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    # Only immutable source/input files are checked against the source archive.
    # A fresh run intentionally regenerates result tables and execution logs.
    manifest = json.loads((ANALYSIS / "FILE_SHA256.json").read_text())
    immutable = ["inputs/quarterly_panel.pkl", "inputs/reference_final_scored.pkl",
                 "inputs/reference_rolling.csv", "source/vaers_pipeline.py",
                 "verify_results.py", "verify_ties.py", "protocol.json"]
    for name in immutable:
        actual = hashlib.sha256((ANALYSIS / name).read_bytes()).hexdigest()
        if actual != manifest[name]:
            raise ValueError(f"The archived source/input checksum changed: {name}")

    started = time.monotonic()
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", MPLBACKEND="Agg")
    for script in ["verify_results.py", "verify_ties.py"]:
        result = subprocess.run([sys.executable, str(ANALYSIS / script)],
                                cwd=ANALYSIS, env=environment, text=True,
                                capture_output=True, check=False)
        (ANALYSIS / "results" / f"{Path(script).stem}_console.log").write_text(
            result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{script} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")

    results = ANALYSIS / "results"
    study = {
        "summary": json.loads((results / "verification_summary.json").read_text()),
        "ties": json.loads((results / "tie_permutation_check.json").read_text()),
        "metrics": pd.read_csv(results / "all_metrics.csv"),
        "topk": pd.read_csv(results / "topk_yield.csv"),
        "workload": pd.read_csv(results / "capture_workload.csv"),
        "coverage": pd.read_csv(results / "candidate_coverage.csv"),
        "windows": pd.read_csv(results / "fresh_fit_windows.csv"),
        "predictions": pd.read_csv(results / "predictions_2026Q1.csv.gz"),
        "seconds": time.monotonic() - started,
    }
    assert study["summary"]["all_checks_passed"]
    assert study["ties"]["all_passed"]
    presentation_checks = _validate_presentation(study)
    report = {
        "executed_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "fresh_fit_windows": len(study["windows"]),
        "fresh_fitted_models": 2 * len(study["windows"]),
        "numerical_checks_passed": study["summary"]["checks"],
        "tie_permutations_checked": study["ties"]["exhaustive_permutations"],
        "presentation_checks": presentation_checks,
        "elapsed_seconds": study["seconds"],
        "versions": study["summary"]["versions"],
        "scope": study["summary"]["scope"],
    }
    (ROOT / "execution_report.json").write_text(json.dumps(report, indent=2) + "\n")
    _configure_plots()
    print(f"Refit logistic regression and gradient boosting in {len(study['windows'])} quarters.")
    print(f"Passed {study['summary']['checks']} numerical checks and the 24-permutation tie check.")
    print(f"Completed in {study['seconds']:.1f} seconds using the frozen processed panel.")
    return study


def _validate_presentation(study):
    """Check the graph inputs directly against fresh predictions and saved metrics."""
    p = study["predictions"]
    y = p.emerging_next_q.to_numpy(dtype=int)
    h = p.ever_strong_before.to_numpy(dtype=int)
    checks = {}
    checks["candidate_and_positive_counts"] = len(p) == 6926 and y.sum() == 294
    checks["first_and_returning_counts"] = int((y * (1-h)).sum()) == 49 and int((y*h).sum()) == 245
    for population, sub in [("All eligible", p), ("No earlier crossing", p[p.ever_strong_before.eq(0)])]:
        for model, score in SCORES.items():
            row = study["metrics"].query("quarter == '2026Q1' and population == @population and model == @model").iloc[0]
            checks[f"{population}_{model}_AP"] = bool(np.isclose(
                average_precision_score(sub.emerging_next_q, sub[score]), row.average_precision,
                atol=1e-12, rtol=0))
    for model, score in SCORES.items():
        x, captured = expected_capture(p, score)
        checks[f"{model}_capture_endpoints"] = bool(x[0] == 0 and x[-1] == len(p)
            and captured[0] == 0 and np.isclose(captured[-1], 100)
            and np.all(np.diff(captured) >= -1e-10))
    for model, expected in [("Gradient boosting",806),("Threshold proximity",1531)]:
        ranked = p.sort_values(SCORES[model], ascending=False, kind="mergesort")
        count = int(np.searchsorted(ranked.emerging_next_q.cumsum().to_numpy(), 147) + 1)
        checks[f"{model}_half_capture"] = count == expected
    top_all = p.sort_values("score_hgb", ascending=False, kind="mergesort").head(100)
    first = p[p.ever_strong_before.eq(0)].sort_values("score_hgb", ascending=False, kind="mergesort").head(100)
    checks["top100_strata"] = bool(top_all.emerging_next_q.sum() == 32
        and top_all.ever_strong_before.sum() == 100 and first.emerging_next_q.sum() == 4)
    coverage = study["coverage"].query("quarter == '2026Q1'").iloc[0]
    checks["coverage_reconciles"] = int(coverage.iloc[2:].sum()) == int(coverage.next_quarter_total_strong)
    if not all(checks.values()):
        raise AssertionError({k:v for k,v in checks.items() if not v})
    return {name:bool(passed) for name,passed in checks.items()}


def study_overview(study):
    p = study["predictions"]
    items = [("Candidates ranked",len(p)), ("Future crossings",int(p.emerging_next_q.sum())),
             ("Returning crossings",int((p.emerging_next_q*p.ever_strong_before).sum())),
             ("First observed crossings",int((p.emerging_next_q*(1-p.ever_strong_before)).sum()))]
    cards = "".join(
        '<div style="flex:1;min-width:150px;padding:18px;border:1px solid #dbe5ee;border-radius:9px;background:#f6f9fc">'
        f'<div style="font-size:28px;font-weight:700;color:{NAVY}">{value:,}</div>'
        f'<div style="font-size:13px;color:{MUTED};margin-top:5px">{label}</div></div>'
        for label,value in items)
    display(HTML('<div style="display:flex;gap:12px;flex-wrap:wrap;margin:14px 0">'+cards+'</div>'))


def final_results_table(study):
    frame = study["metrics"].query("quarter == '2026Q1'").pivot(
        index="model",columns="population",values="average_precision").loc[MODELS,
        ["All eligible","No earlier crossing","Earlier crossing"]]
    frame.columns = ["All eligible pairs", "No earlier crossing", "Earlier crossing"]
    frame.index = [SHORT[x] for x in frame.index]
    frame.index.name = "Method"
    frame.to_csv(TABLES / "final_average_precision.csv")
    display(frame.style.format("{:.4f}").set_caption("Average precision in the final test; higher is better."))


def coverage_table(study):
    r = study["coverage"].query("quarter == '2026Q1'").iloc[0]
    frame = pd.DataFrame({
        "Status in the preceding quarter": ["Already above the threshold", "Eligible for our ranking",
            "Absent this quarter, but observed earlier", "Never observed in the available history"],
        "Pairs above the threshold in 2026Q2": [int(r.already_strong_current),int(r.eligible_crossings),
            int(r.absent_current_seen_earlier),int(r.absent_current_never_seen_earlier)],
    })
    frame.to_csv(TABLES / "candidate_coverage.csv",index=False)
    display(frame.style.hide(axis="index"))


def expected_capture(predictions, score):
    """Expected cumulative capture under uniform ordering within tied score blocks.

    The straight segment inside a tied block gives expected capture at each
    review budget. Its inverse is not the expected stopping time to a target;
    that distinct quantity is calculated by the verification script.
    """
    grouped = predictions.groupby(score,sort=True).emerging_next_q.agg(["size","sum"]).sort_index(ascending=False)
    x = np.r_[0,grouped["size"].cumsum().to_numpy()]
    captured = np.r_[0,grouped["sum"].cumsum().to_numpy()] / predictions.emerging_next_q.sum()*100
    return x,captured


def _save(fig,name):
    fig.savefig(FIGURES/f"{name}.png",dpi=180,bbox_inches="tight")
    fig.savefig(FIGURES/f"{name}.pdf",bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_review_efficiency(study):
    fig = plt.figure(figsize=(12,6.8))
    ax = fig.add_axes([.095,.22,.865,.60])
    fig.text(.095,.95,"Half of the future crossings in 806 reviews",fontsize=21,fontweight="bold")
    fig.text(.095,.895,"Final test: 2026Q1 candidates → 2026Q2 outcomes · 294 eligible future crossings",fontsize=11,color=MUTED)
    for model,score in SCORES.items():
        x,capture = expected_capture(study["predictions"],score)
        ax.plot(x,capture,color=COLORS[model],lw=2.7 if model=="Gradient boosting" else 1.9,
                ls="--" if model=="Threshold proximity" else "-",label=SHORT[model])
    ax.axhline(50,color=NAVY,ls=(0,(3,3)),lw=.9,alpha=.65)
    ax.text(30,51.7,"50% capture",fontsize=10,color=MUTED)
    for x,model,xytext in [(806,"Gradient boosting",(390,69)),(1531,"Threshold proximity",(1370,24))]:
        ax.plot(x,50,"o",color=COLORS[model],ms=7,zorder=5)
        ax.vlines(x,0,50,colors=COLORS[model],linestyles=":",lw=1)
        ax.annotate(f"{SHORT[model]}\n{x:,} pairs",xy=(x,50),xytext=xytext,fontsize=11,
                    color=COLORS[model],fontweight="semibold",
                    arrowprops=dict(arrowstyle="-",color=COLORS[model],lw=1))
    ax.set(xlim=(0,2000),ylim=(0,90),xlabel="Number of candidate pairs examined",
           ylabel="Future threshold crossings captured (%)")
    ax.xaxis.set_major_locator(MultipleLocator(400))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.grid(axis="y",color=PALE,lw=.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left",bbox_to_anchor=(0,1.01),ncol=4,frameon=False,fontsize=9)
    fig.text(.095,.105,"806 versus 1,531 is 47.4% fewer candidate pairs than threshold proximity.",fontsize=11,fontweight="semibold")
    fig.text(.095,.062,"The history-aware baseline is stronger. Tied-score segments show expected capture under random tie ordering.\nThis graph shows the first 2,000 positions; it measures candidate counts, not staff time or clinical benefit.",fontsize=9,color=MUTED)
    _save(fig,"01_review_efficiency")


def plot_final_performance(study):
    fig,axes = plt.subplots(1,2,figsize=(12,6.1),sharex=True,sharey=True)
    fig.subplots_adjust(left=.19,right=.96,bottom=.22,top=.72,wspace=.16)
    fig.text(.055,.95,"Does the ranking improve?",fontsize=21,fontweight="bold")
    fig.text(.055,.89,"Average precision measures ranking quality. Compare methods within each population.",fontsize=11,color=MUTED)
    groups = [("All eligible","All eligible pairs\n6,926 candidates · 294 crossings"),
              ("No earlier crossing","No earlier crossing since 2022\n4,581 candidates · 49 crossings")]
    for ax,(population,title) in zip(axes,groups):
        sub = study["metrics"].query("quarter == '2026Q1' and population == @population").set_index("model").loc[MODELS]
        positions = np.arange(len(MODELS))[::-1]
        ax.barh(positions,sub.average_precision,color=[COLORS[m] for m in MODELS],height=.58)
        for y,value in zip(positions,sub.average_precision):
            ax.text(value+.005,y,f"{value:.3f}",va="center",fontsize=11,fontweight="semibold")
        ax.axvline(float(sub.base_rate.iloc[0]),color=NAVY,ls=(0,(2,3)),lw=1,alpha=.75)
        ax.set_title(title,fontsize=11,loc="left",pad=17)
        ax.set_xlim(0,.24)
        ax.set_xlabel("Average precision (AP)",labelpad=10)
        ax.set_yticks(positions,[SHORT[m] for m in MODELS])
        ax.xaxis.set_major_locator(MultipleLocator(.05))
        ax.grid(axis="x",color=PALE,lw=.8)
        ax.set_axisbelow(True)
    fig.text(.055,.09,"Both panels use the same scale. Dashed lines mark the positive-outcome rates: 4.24% and 1.07%.",fontsize=10,color=MUTED)
    fig.text(.055,.045,"Within the no-earlier-crossing group, the history flag is constant, so history + proximity reduces to proximity.",fontsize=9,color=MUTED)
    _save(fig,"02_ranking_quality")


def plot_first_and_returning(study):
    p = study["predictions"]
    combined = p.sort_values("score_hgb",ascending=False,kind="mergesort").head(100)
    first_only = p[p.ever_strong_before.eq(0)].sort_values("score_hgb",ascending=False,kind="mergesort").head(100)
    selected = [combined,first_only]
    returning = np.array([int((x.emerging_next_q*x.ever_strong_before).sum()) for x in selected])
    first = np.array([int((x.emerging_next_q*(1-x.ever_strong_before)).sum()) for x in selected])
    other = 100-returning-first
    fig = plt.figure(figsize=(12,5.6))
    ax = fig.add_axes([.22,.30,.73,.40])
    fig.text(.055,.94,"Returning alerts explain much of the top-ranked yield",fontsize=20,fontweight="bold")
    fig.text(.055,.875,"Gradient boosting · two separate rankings of 100 candidate pairs",fontsize=11,color=MUTED)
    left = np.zeros(2)
    for values,label,color in [(returning,"Returning crossing",COLORS["Logistic regression"]),
                               (first,"First observed crossing",COLORS["History then proximity"]),
                               (other,"No next-quarter crossing",PALE)]:
        ax.barh([1,0],values,left=left,height=.49,color=color,label=label)
        for y,n,l in zip([1,0],values,left):
            if n:
                ax.text(l+n/2,y,str(n),ha="center",va="center",fontsize=13,fontweight="semibold",
                        color="white" if label=="Returning crossing" else NAVY)
        left += values
    ax.set_yticks([1,0],["All eligible pairs\nranked together","No earlier crossing\nranked separately"])
    ax.set_xlim(0,100)
    ax.set_ylim(-.55,1.55)
    ax.set_xlabel("Candidate pairs among the top 100",labelpad=10)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y",length=0,pad=12)
    ax.legend(loc="lower center",bbox_to_anchor=(.41,1.10),ncol=3,frameon=False,fontsize=9)
    fig.text(.055,.105,"The combined top 100 captured 32 returning crossings and 0 first observed crossings.",fontsize=11,fontweight="semibold")
    fig.text(.055,.055,"A separate first-crossing list captured 4 positives. Each row uses its own 100-pair budget.\n“First observed” means no earlier threshold crossing in the available history beginning in 2022.",fontsize=9,color=MUTED)
    _save(fig,"03_first_and_returning")


def plot_quarterly_performance(study):
    quarter_order = study["windows"].quarter.tolist()
    frame = study["metrics"].query("population == 'All eligible'").pivot(
        index="quarter",columns="model",values="average_precision").loc[quarter_order,MODELS]
    frame.to_csv(TABLES/"quarterly_average_precision.csv")
    fig,ax = plt.subplots(figsize=(12,5.7))
    fig.subplots_adjust(left=.095,right=.965,bottom=.27,top=.76)
    fig.text(.095,.95,"The comparison holds across several test quarters",fontsize=20,fontweight="bold")
    fig.text(.095,.89,"All eligible candidates · each point predicts the following quarter",fontsize=11,color=MUTED)
    for model in MODELS:
        ax.plot(range(len(frame)),frame[model],color=COLORS[model],marker="o",ms=5,
                lw=2.5 if model=="Gradient boosting" else 1.8,
                ls="--" if model=="Threshold proximity" else "-",label=SHORT[model])
    ax.set_xticks(range(len(frame)),frame.index,rotation=35,ha="right")
    ax.set(ylabel="Average precision (AP)",ylim=(0,.28),xlim=(-.15,8.15))
    ax.yaxis.set_major_locator(MultipleLocator(.05))
    ax.grid(axis="y",color=PALE,lw=.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left",bbox_to_anchor=(0,1.04),ncol=4,frameon=False,fontsize=9)
    fig.text(.095,.065,"Gradient boosting led proximity and the history rule in 9/9 quarters, and logistic regression in 7/9.",fontsize=10,fontweight="semibold")
    fig.text(.095,.025,"These quarters were already examined and share historical data. These are descriptive comparisons, not independent significance tests.",fontsize=9,color=MUTED)
    _save(fig,"04_quarterly_performance")


def workload_table(study):
    frame = study["workload"].query("quarter == '2026Q1' and population == 'All eligible' and capture_fraction == 0.5")
    frame = frame.set_index("model").loc[MODELS]
    output = pd.DataFrame({"Method":[SHORT[m] for m in MODELS],
        "Reviews to capture 147 positives":[f"{r.tie_expected_reviews:,.0f}" if r.score_tie_size_at_target==1
            else f"{r.tie_expected_reviews:,.2f} (tie expectation)" for _,r in frame.iterrows()]})
    output.to_csv(TABLES/"reviews_for_half_capture.csv",index=False)
    display(output.style.hide(axis="index"))


def execution_summary(study):
    report = json.loads((ROOT/"execution_report.json").read_text())
    display(pd.DataFrame({"Check":["Fresh fitted models","Forecast quarters","Numerical checks passed",
        "Tie permutations checked","Presentation consistency checks","Executed (Los Angeles time)"],
        "Result":[report["fresh_fitted_models"],report["fresh_fit_windows"],report["numerical_checks_passed"],
        report["tie_permutations_checked"],f"{sum(report['presentation_checks'].values())}/{len(report['presentation_checks'])}",
        report["executed_at"]]}).style.hide(axis="index"))
