"""Reusable functions for the AIPM VAERS emerging-signal project.

The project forecasts which vaccine-type × MedDRA-symptom pairs that are
*not* strong signals in quarter t will become strong signals in the exact
next calendar quarter t+1.

Important: this is a prioritization/forecasting workflow. It does not infer
causality between a vaccine and a reported symptom.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import math
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42

DATA_COLS = [
    "VAERS_ID", "RECVDATE", "AGE_YRS", "SEX", "V_ADMINBY",
    "DIED", "L_THREAT", "HOSPITAL", "X_STAY", "DISABLE", "BIRTH_DEFECT",
    "ER_ED_VISIT", "OFC_VISIT", "ORDER",
]
VAX_COLS = ["VAERS_ID", "VAX_TYPE", "VAX_NAME", "ORDER"]
SYM_COLS = [
    "VAERS_ID", "ORDER", "SYMPTOM1", "SYMPTOM2", "SYMPTOM3",
    "SYMPTOM4", "SYMPTOM5",
]

STRONG_A_MIN = 3
STRONG_PRR_MIN = 2.0
STRONG_PRR_LCB_MIN = 1.0

FEATURES = [
    # Volume / expected counts
    "log_a", "log_n_vax", "log_n_symptom", "log_N", "log_expected",
    # Current classical signal information
    "log_prr", "log_ror", "ic", "log_prr_lcb95", "log_gp",
    # Current context
    "serious_share", "er_urgent_share", "office_visit_share",
    "mean_age_filled", "mean_age_missing",
    "female_share_filled", "female_share_missing",
    "pharmacy_share", "senior_facility_share", "unknown_admin_share",
    # Exact prior-quarter information
    "log_a_prev", "log_prr_prev", "log_gp_prev",
    "log_count_change", "log_prr_change",
    "observed_prev", "observed_prev2", "new_pair",
    "strong_prev", "strong_prev2", "persistence_2q",
]

TEMPORAL_FEATURES = [
    "log_a_prev", "log_prr_prev", "log_gp_prev",
    "log_count_change", "log_prr_change",
    "observed_prev", "observed_prev2", "new_pair",
    "strong_prev", "strong_prev2", "persistence_2q",
]

CONTEXT_FEATURES = [
    "serious_share", "er_urgent_share", "office_visit_share",
    "mean_age_filled", "mean_age_missing",
    "female_share_filled", "female_share_missing",
    "pharmacy_share", "senior_facility_share", "unknown_admin_share",
]

CLASSICAL_FEATURES = [
    "log_prr", "log_ror", "ic", "log_prr_lcb95", "log_gp",
]


@dataclass
class ModelBundle:
    logistic_base: object
    logistic_calibrated: object
    gradient_boosting_base: object
    gradient_boosting_calibrated: object
    age_fill_value: float


def quarter_to_int(q: str) -> int:
    """Convert '2025Q3' to a sortable integer."""
    q = str(q)
    return int(q[:4]) * 4 + int(q[-1])


def int_to_quarter(q_int: int) -> str:
    """Inverse of quarter_to_int."""
    year, rem = divmod(int(q_int) - 1, 4)
    return f"{year}Q{rem + 1}"


def _yes_indicator(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper().eq("Y").astype("int8")


def _female_share(series: pd.Series) -> float:
    values = series.fillna("").astype(str).str.upper()
    known = values.isin(["F", "M"])
    if not known.any():
        return np.nan
    return float(values[known].eq("F").mean())


def _assert_files(data_dir: Path, years: Sequence[int]) -> None:
    missing = []
    for year in years:
        for suffix in ["DATA", "VAX", "SYMPTOMS"]:
            path = data_dir / f"{year}VAERS{suffix}.csv"
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required VAERS files:\n" + "\n".join(missing))


def load_clean_year(
    data_dir: str | Path,
    year: int,
    complete_through: str = "2026Q2",
) -> tuple[pd.DataFrame, dict]:
    """Load one year and return an order-aware report-vaccine-symptom table.

    We keep ORDER == 1. This conservative choice avoids attaching information
    added in later report versions to an earlier receipt date. Tables are joined
    on (VAERS_ID, ORDER), not VAERS_ID alone.
    """
    data_dir = Path(data_dir)
    data_path = data_dir / f"{year}VAERSDATA.csv"
    vax_path = data_dir / f"{year}VAERSVAX.csv"
    sym_path = data_dir / f"{year}VAERSSYMPTOMS.csv"

    data = pd.read_csv(
        data_path, encoding="latin1", low_memory=False,
        usecols=lambda c: c in DATA_COLS,
    )
    vax = pd.read_csv(
        vax_path, encoding="latin1", low_memory=False,
        usecols=lambda c: c in VAX_COLS,
    )
    sym = pd.read_csv(
        sym_path, encoding="latin1", low_memory=False,
        usecols=lambda c: c in SYM_COLS,
    )

    audit = {
        "year": year,
        "data_raw_rows": len(data),
        "data_unique_ids": data["VAERS_ID"].nunique(),
        "vax_raw_rows": len(vax),
        "sym_raw_rows": len(sym),
    }

    for frame in (data, vax, sym):
        frame["ORDER"] = pd.to_numeric(frame["ORDER"], errors="coerce")

    data = data[data["ORDER"].eq(1)].copy()
    vax = vax[vax["ORDER"].eq(1)].copy()
    sym = sym[sym["ORDER"].eq(1)].copy()
    audit["data_order1_rows"] = len(data)

    # Receipt date is the surveillance clock: what VAERS knew at that time.
    data["receipt_date"] = pd.to_datetime(data["RECVDATE"], errors="coerce")
    data = data[data["receipt_date"].notna()].copy()
    data["quarter"] = data["receipt_date"].dt.to_period("Q").astype(str)
    data["q_int"] = data["quarter"].map(quarter_to_int)
    data = data[data["q_int"] <= quarter_to_int(complete_through)].copy()

    data["age_yrs"] = pd.to_numeric(data["AGE_YRS"], errors="coerce")
    data["sex"] = data["SEX"].fillna("").astype(str).str.upper()
    data["admin_by"] = data["V_ADMINBY"].fillna("UNK").astype(str).str.strip().str.upper()

    core_serious_cols = ["DIED", "L_THREAT", "HOSPITAL", "X_STAY", "DISABLE", "BIRTH_DEFECT"]
    serious_parts = [_yes_indicator(data[c]) for c in core_serious_cols if c in data.columns]
    data["serious_core"] = np.maximum.reduce(serious_parts).astype("int8") if serious_parts else 0
    data["er_urgent"] = _yes_indicator(data["ER_ED_VISIT"])
    data["office_visit"] = _yes_indicator(data["OFC_VISIT"])

    # Facility where the vaccine was administered; not reporter occupation.
    data["admin_pharmacy"] = data["admin_by"].eq("PHM").astype("int8")
    data["admin_senior"] = data["admin_by"].eq("SEN").astype("int8")
    data["admin_unknown"] = data["admin_by"].isin(["UNK", "", "NAN"]).astype("int8")

    vax["vax_type"] = vax["VAX_TYPE"].fillna("").astype(str).str.strip().str.upper()
    vax = vax[~vax["vax_type"].isin(["", "UNK", "UNKNOWN", "NAN"])].copy()
    vax = vax[["VAERS_ID", "ORDER", "vax_type"]].drop_duplicates()

    symptom_cols = [c for c in SYM_COLS if c.startswith("SYMPTOM")]
    sym_long = (
        sym.melt(
            id_vars=["VAERS_ID", "ORDER"],
            value_vars=symptom_cols,
            value_name="meddra_pt",
        )
        .drop(columns="variable")
        .dropna(subset=["meddra_pt"])
    )
    sym_long["meddra_pt"] = sym_long["meddra_pt"].astype(str).str.strip()
    sym_long = sym_long[
        (~sym_long["meddra_pt"].eq(""))
        & (~sym_long["meddra_pt"].str.casefold().eq("no adverse event"))
    ].drop_duplicates()

    report_cols = [
        "VAERS_ID", "ORDER", "quarter", "q_int", "age_yrs", "sex",
        "serious_core", "er_urgent", "office_visit",
        "admin_pharmacy", "admin_senior", "admin_unknown",
    ]

    long = (
        data[report_cols]
        .merge(vax, on=["VAERS_ID", "ORDER"], how="inner")
        .merge(sym_long, on=["VAERS_ID", "ORDER"], how="inner")
    )

    # One report can list the same type or term more than once. Count it once per pair.
    long = long.drop_duplicates(
        ["VAERS_ID", "ORDER", "quarter", "vax_type", "meddra_pt"]
    ).reset_index(drop=True)
    long["source_year"] = year

    audit["analysis_unique_reports"] = long["VAERS_ID"].nunique()
    audit["analysis_long_rows"] = len(long)
    audit["first_quarter"] = long["quarter"].min() if len(long) else None
    audit["last_quarter"] = long["quarter"].max() if len(long) else None
    return long, audit


def build_long_dataset(
    data_dir: str | Path,
    years: Sequence[int] = tuple(range(2022, 2027)),
    complete_through: str = "2026Q2",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the combined order-aware long dataset for all years."""
    data_dir = Path(data_dir)
    _assert_files(data_dir, years)
    frames, audits = [], []
    for year in years:
        frame, audit = load_clean_year(data_dir, year, complete_through=complete_through)
        frames.append(frame)
        audits.append(audit)
    long = pd.concat(frames, ignore_index=True)
    audit_df = pd.DataFrame(audits)
    return long, audit_df


def build_quarterly_panel(long: pd.DataFrame) -> pd.DataFrame:
    """Build vaccine-type × symptom × quarter 2×2 tables and context summaries."""
    panels = []
    for quarter, gq in long.groupby("quarter", sort=False):
        N = int(gq["VAERS_ID"].nunique())
        pair_reports = gq.drop_duplicates(["VAERS_ID", "vax_type", "meddra_pt"]).copy()
        vax_reports = pair_reports[["VAERS_ID", "vax_type"]].drop_duplicates()
        symptom_reports = pair_reports[["VAERS_ID", "meddra_pt"]].drop_duplicates()

        n_vax = vax_reports.groupby("vax_type")["VAERS_ID"].nunique().rename("n_vax")
        n_sym = symptom_reports.groupby("meddra_pt")["VAERS_ID"].nunique().rename("n_symptom")
        counts = (
            pair_reports.groupby(["vax_type", "meddra_pt"])["VAERS_ID"]
            .nunique().rename("a").reset_index()
            .merge(n_vax.reset_index(), on="vax_type", how="left")
            .merge(n_sym.reset_index(), on="meddra_pt", how="left")
        )
        counts["N"] = N
        counts["b"] = counts["n_vax"] - counts["a"]
        counts["c"] = counts["n_symptom"] - counts["a"]
        counts["d"] = N - counts["a"] - counts["b"] - counts["c"]

        context = (
            pair_reports.groupby(["vax_type", "meddra_pt"])
            .agg(
                serious_share=("serious_core", "mean"),
                er_urgent_share=("er_urgent", "mean"),
                office_visit_share=("office_visit", "mean"),
                mean_age=("age_yrs", "mean"),
                female_share=("sex", _female_share),
                pharmacy_share=("admin_pharmacy", "mean"),
                senior_facility_share=("admin_senior", "mean"),
                unknown_admin_share=("admin_unknown", "mean"),
            )
            .reset_index()
        )
        out = counts.merge(context, on=["vax_type", "meddra_pt"], how="left")
        out["quarter"] = quarter
        out["q_int"] = quarter_to_int(quarter)
        panels.append(out)

    panel = pd.concat(panels, ignore_index=True)
    for c in ["a", "n_vax", "n_symptom", "N", "b", "c", "d"]:
        panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0).astype("int32")
    if (panel["d"] < 0).any():
        raise ValueError("Negative d counts detected; the 2×2 panel construction is invalid.")
    return panel


def add_disproportionality_scores(panel: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """Add PRR, ROR, IC, lower bounds, and a simple shrinkage proxy."""
    x = panel.copy()
    a = x["a"].astype(float) + alpha
    b = x["b"].astype(float) + alpha
    c = x["c"].astype(float) + alpha
    d = x["d"].astype(float) + alpha
    total = a + b + c + d

    x["prr"] = (a / (a + b)) / (c / (c + d))
    x["ror"] = (a / b) / (c / d)
    x["ic"] = np.log2((a * total) / ((a + b) * (a + c)))

    se_log_prr = np.sqrt(np.maximum(1/a - 1/(a+b) + 1/c - 1/(c+d), 1e-12))
    se_log_ror = np.sqrt(np.maximum(1/a + 1/b + 1/c + 1/d, 1e-12))
    x["prr_lcb95"] = np.exp(np.log(x["prr"]) - 1.96 * se_log_prr)
    x["ror_lcb95"] = np.exp(np.log(x["ror"]) - 1.96 * se_log_ror)

    x["expected"] = ((a + b) * (a + c) / total).clip(lower=1e-12)
    # A simple Gamma-Poisson-style shrinkage proxy, not the proprietary MGPS EBGM.
    x["gp_shrinkage"] = (x["a"].astype(float) + alpha) / (x["expected"] + alpha)

    x["strong_signal"] = (
        (x["a"] >= STRONG_A_MIN)
        & (x["prr"] >= STRONG_PRR_MIN)
        & (x["prr_lcb95"] > STRONG_PRR_LCB_MIN)
    ).astype("int8")

    # A transparent baseline for how close a non-signal is to all three thresholds.
    x["threshold_proximity"] = np.minimum.reduce([
        x["a"].astype(float) / STRONG_A_MIN,
        x["prr"] / STRONG_PRR_MIN,
        x["prr_lcb95"] / STRONG_PRR_LCB_MIN,
    ])
    return x


def build_emerging_signal_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    """Create exact-calendar-quarter history and next-quarter labels.

    A pair absent in t+1 is retained and assigned future count/scores of zero.
    Pairs already strong at t are excluded from the primary candidate population.
    """
    x = panel.copy().sort_values(["vax_type", "meddra_pt", "q_int"]).reset_index(drop=True)
    keys = ["vax_type", "meddra_pt", "q_int"]

    # Previous exact quarter.
    prev_cols = ["a", "prr", "gp_shrinkage", "strong_signal"]
    prev = x[["vax_type", "meddra_pt", "q_int"] + prev_cols].copy()
    prev["q_int"] = prev["q_int"] + 1
    prev = prev.rename(columns={
        "a": "a_prev", "prr": "prr_prev", "gp_shrinkage": "gp_prev",
        "strong_signal": "strong_prev",
    })
    x = x.merge(prev, on=keys, how="left")

    prev2 = panel[["vax_type", "meddra_pt", "q_int", "strong_signal"]].copy()
    prev2["q_int"] = prev2["q_int"] + 2
    prev2 = prev2.rename(columns={"strong_signal": "strong_prev2"})
    x = x.merge(prev2, on=keys, how="left")

    # Future exact quarter: shift the future row backward one quarter before merge.
    future = panel[[
        "vax_type", "meddra_pt", "q_int", "a", "prr", "prr_lcb95", "strong_signal"
    ]].copy()
    future["q_int"] = future["q_int"] - 1
    future = future.rename(columns={
        "a": "a_next", "prr": "prr_next", "prr_lcb95": "prr_lcb95_next",
        "strong_signal": "emerging_next_q",
    })
    x = x.merge(future, on=keys, how="left")

    for c in ["a_prev", "prr_prev", "gp_prev", "strong_prev", "strong_prev2"]:
        x[c] = x[c].fillna(0)
    x["observed_prev"] = (x["a_prev"] > 0).astype("int8")
    x["observed_prev2"] = (x["strong_prev2"].notna()).astype("int8")
    # strong_prev2 was filled above; detect observation by a separate exact match.
    obs2 = panel[["vax_type", "meddra_pt", "q_int"]].copy()
    obs2["q_int"] = obs2["q_int"] + 2
    obs2["observed_prev2"] = 1
    x = x.drop(columns=["observed_prev2"]).merge(obs2, on=keys, how="left")
    x["observed_prev2"] = x["observed_prev2"].fillna(0).astype("int8")
    x["strong_prev"] = x["strong_prev"].fillna(0).astype("int8")
    x["strong_prev2"] = x["strong_prev2"].fillna(0).astype("int8")

    for c in ["a_next", "prr_next", "prr_lcb95_next", "emerging_next_q"]:
        x[c] = x[c].fillna(0)
    x["emerging_next_q"] = x["emerging_next_q"].astype("int8")

    x["quarter"] = x["q_int"].map(int_to_quarter)
    max_q = int(panel["q_int"].max())
    x = x[x["q_int"] < max_q].copy()  # final complete quarter has no observable t+1 yet
    x = x[x["strong_signal"].eq(0)].copy()

    x["new_pair"] = (x["observed_prev"] == 0).astype("int8")
    x["persistence_2q"] = (x["observed_prev"] + x["observed_prev2"]).astype("int8")

    # Stable transforms for heavy-tailed counts and ratios.
    x["log_a"] = np.log1p(x["a"])
    x["log_n_vax"] = np.log1p(x["n_vax"])
    x["log_n_symptom"] = np.log1p(x["n_symptom"])
    x["log_N"] = np.log1p(x["N"])
    x["log_expected"] = np.log1p(x["expected"])
    x["log_prr"] = np.log1p(x["prr"])
    x["log_ror"] = np.log1p(x["ror"])
    x["log_prr_lcb95"] = np.log1p(x["prr_lcb95"])
    x["log_gp"] = np.log1p(x["gp_shrinkage"])

    x["log_a_prev"] = np.log1p(x["a_prev"])
    x["log_prr_prev"] = np.log1p(x["prr_prev"])
    x["log_gp_prev"] = np.log1p(x["gp_prev"])
    x["log_count_change"] = np.log1p(x["a"]) - np.log1p(x["a_prev"])
    x["log_prr_change"] = np.log1p(x["prr"]) - np.log1p(x["prr_prev"])

    x["mean_age_missing"] = x["mean_age"].isna().astype("int8")
    # A finite placeholder only. Each fitted model replaces this column using
    # an age median learned exclusively from its own fitting window.
    x["mean_age_filled"] = x["mean_age"].fillna(0.0)
    x["female_share_missing"] = x["female_share"].isna().astype("int8")
    x["female_share_filled"] = x["female_share"].fillna(0.5)

    # Safety checks.
    if not set(FEATURES).issubset(x.columns):
        missing = sorted(set(FEATURES) - set(x.columns))
        raise ValueError(f"Missing engineered features: {missing}")
    if x[FEATURES].replace([np.inf, -np.inf], np.nan).isna().any().any():
        bad = x[FEATURES].replace([np.inf, -np.inf], np.nan).isna().sum()
        raise ValueError(f"Non-finite feature values remain:\n{bad[bad > 0]}")
    return x.reset_index(drop=True)


def build_project_data(
    data_dir: str | Path,
    years: Sequence[int] = tuple(range(2022, 2027)),
    complete_through: str = "2026Q2",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full raw-data pipeline: long data, audit, scored panel, candidates."""
    long, audit = build_long_dataset(data_dir, years, complete_through)
    panel = add_disproportionality_scores(build_quarterly_panel(long))
    candidates = build_emerging_signal_candidates(panel)
    return long, audit, panel, candidates


def make_logistic(seed: int = SEED):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        ),
    )


def make_hgb(seed: int = SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=False,
        random_state=seed,
    )


def calibrate_prefit(estimator, X_cal: pd.DataFrame, y_cal: pd.Series):
    """Sigmoid-calibrate a fitted estimator on a distinct later time period."""
    calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def prepare_features(
    frame: pd.DataFrame,
    features: Sequence[str],
    age_fill_value: float,
) -> pd.DataFrame:
    """Apply the training-window age fill without learning from later data."""
    X = frame[list(features)].copy()
    if "mean_age_filled" in features:
        X["mean_age_filled"] = frame["mean_age"].fillna(age_fill_value)
    return X


def fit_model_bundle(
    fit_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    features: Sequence[str] = FEATURES,
    seed: int = SEED,
) -> ModelBundle:
    if fit_df.empty or cal_df.empty:
        raise ValueError("Fit and calibration data must both be non-empty.")
    if fit_df["emerging_next_q"].nunique() < 2 or cal_df["emerging_next_q"].nunique() < 2:
        raise ValueError("Fit and calibration periods must each contain both outcome classes.")

    age_fill_value = float(fit_df["mean_age"].median())
    if not np.isfinite(age_fill_value):
        raise ValueError("No finite ages in the fitting data.")
    X_fit = prepare_features(fit_df, features, age_fill_value)
    y_fit = fit_df["emerging_next_q"]
    X_cal = prepare_features(cal_df, features, age_fill_value)
    y_cal = cal_df["emerging_next_q"]

    lr = make_logistic(seed)
    lr.fit(X_fit, y_fit)
    lr_cal = calibrate_prefit(lr, X_cal, y_cal)

    hgb = make_hgb(seed)
    hgb.fit(X_fit, y_fit)
    hgb_cal = calibrate_prefit(hgb, X_cal, y_cal)
    return ModelBundle(lr, lr_cal, hgb, hgb_cal, age_fill_value)


def add_model_scores(
    test_df: pd.DataFrame,
    bundle: ModelBundle,
    features: Sequence[str] = FEATURES,
) -> pd.DataFrame:
    out = test_df.copy()
    X = prepare_features(out, features, bundle.age_fill_value)
    if "mean_age_filled" in features:
        out["mean_age_filled"] = X["mean_age_filled"]
    out["score_prr"] = out["prr"].astype(float)
    out["score_gp"] = out["gp_shrinkage"].astype(float)
    out["score_proximity"] = out["threshold_proximity"].astype(float)
    out["score_lr"] = bundle.logistic_calibrated.predict_proba(X)[:, 1]
    out["score_hgb"] = bundle.gradient_boosting_calibrated.predict_proba(X)[:, 1]
    for c in ["score_lr", "score_hgb"]:
        if not out[c].between(0, 1).all():
            raise ValueError(f"{c} contains values outside [0,1].")
    return out


MODEL_SCORE_COLUMNS = {
    "PRR": "score_prr",
    "Gamma-Poisson shrinkage": "score_gp",
    "Threshold proximity": "score_proximity",
    "Logistic Regression": "score_lr",
    "Gradient Boosting": "score_hgb",
}


def evaluate_rankings(scored: pd.DataFrame) -> pd.DataFrame:
    y = scored["emerging_next_q"].to_numpy()
    rows = []
    for name, col in MODEL_SCORE_COLUMNS.items():
        score = scored[col].to_numpy()
        row = {
            "model": name,
            "AUPRC": average_precision_score(y, score),
            "AUROC": roc_auc_score(y, score),
            "Brier": np.nan,
        }
        if col in ["score_lr", "score_hgb"]:
            row["Brier"] = brier_score_loss(y, score)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("AUPRC", ascending=False).reset_index(drop=True)


def precision_recall_lift_at_k(
    scored: pd.DataFrame,
    ks: Sequence[int] = (10, 25, 50, 100, 250, 500),
) -> pd.DataFrame:
    y = scored["emerging_next_q"].to_numpy()
    total_pos = y.sum()
    base_rate = y.mean()
    rows = []
    for name, col in MODEL_SCORE_COLUMNS.items():
        order = np.argsort(-scored[col].to_numpy(), kind="mergesort")
        for k in ks:
            actual_k = min(int(k), len(scored))
            selected = y[order[:actual_k]]
            precision = float(selected.mean()) if actual_k else np.nan
            recall = float(selected.sum() / total_pos) if total_pos else np.nan
            rows.append({
                "model": name,
                "k": actual_k,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "lift_at_k": precision / base_rate if base_rate > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def review_efficiency(scored: pd.DataFrame) -> pd.DataFrame:
    """Cumulative future signals captured as more ranked candidates are reviewed."""
    y = scored["emerging_next_q"].to_numpy()
    total_pos = int(y.sum())
    rows = []
    for name, col in MODEL_SCORE_COLUMNS.items():
        order = np.argsort(-scored[col].to_numpy(), kind="mergesort")
        cumulative = np.cumsum(y[order])
        for i, captured in enumerate(cumulative, start=1):
            rows.append({
                "model": name,
                "pairs_reviewed": i,
                "fraction_candidates_reviewed": i / len(y),
                "signals_captured": int(captured),
                "fraction_signals_captured": captured / total_pos if total_pos else np.nan,
            })
    return pd.DataFrame(rows)


def workload_table(
    efficiency: pd.DataFrame,
    capture_targets: Sequence[float] = (0.25, 0.50, 0.75, 0.80),
    reference_model: str = "Threshold proximity",
) -> pd.DataFrame:
    rows = []
    for target in capture_targets:
        model_counts = {}
        for model, g in efficiency.groupby("model"):
            reached = g[g["fraction_signals_captured"] >= target]
            model_counts[model] = int(reached["pairs_reviewed"].iloc[0]) if len(reached) else np.nan
        reference_count = model_counts.get(reference_model, np.nan)
        for model, count in model_counts.items():
            reduction = np.nan
            if np.isfinite(reference_count) and np.isfinite(count) and reference_count > 0:
                reduction = 1 - count / reference_count
            rows.append({
                "capture_target": target,
                "model": model,
                "pairs_to_review": count,
                "workload_reduction_vs_threshold_proximity": reduction,
            })
    return pd.DataFrame(rows)


def bootstrap_auprc(
    scored: pd.DataFrame,
    n_boot: int = 300,
    seed: int = SEED,
) -> pd.DataFrame:
    """Row bootstrap on the final test quarter; each pair appears once in that quarter."""
    rng = np.random.default_rng(seed)
    y = scored["emerging_next_q"].to_numpy()
    n = len(scored)
    rows = []
    for name, col in MODEL_SCORE_COLUMNS.items():
        score = scored[col].to_numpy()
        estimates = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            if np.unique(y[idx]).size < 2:
                continue
            estimates.append(average_precision_score(y[idx], score[idx]))
        estimates = np.asarray(estimates)
        rows.append({
            "model": name,
            "AUPRC": average_precision_score(y, score),
            "bootstrap_low_95": np.quantile(estimates, 0.025),
            "bootstrap_high_95": np.quantile(estimates, 0.975),
            "n_boot_used": len(estimates),
        })
    return pd.DataFrame(rows).sort_values("AUPRC", ascending=False).reset_index(drop=True)


def rolling_temporal_validation(
    candidates: pd.DataFrame,
    test_quarters: Sequence[str],
    features: Sequence[str] = FEATURES,
    seed: int = SEED,
) -> pd.DataFrame:
    """Expanding-window validation with the immediately preceding quarter for calibration."""
    rows = []
    for test_q in test_quarters:
        test_i = quarter_to_int(test_q)
        cal_i = test_i - 1
        train = candidates[candidates["q_int"] < cal_i]
        cal = candidates[candidates["q_int"] == cal_i]
        test = candidates[candidates["q_int"] == test_i]
        if train.empty or cal.empty or test.empty:
            continue
        if min(train["emerging_next_q"].nunique(), cal["emerging_next_q"].nunique(), test["emerging_next_q"].nunique()) < 2:
            continue
        bundle = fit_model_bundle(train, cal, features=features, seed=seed)
        scored = add_model_scores(test, bundle, features=features)
        metrics = evaluate_rankings(scored).set_index("model")
        row = {
            "test_quarter": test_q,
            "n_candidates": len(test),
            "n_positive": int(test["emerging_next_q"].sum()),
            "base_rate": float(test["emerging_next_q"].mean()),
        }
        for model in MODEL_SCORE_COLUMNS:
            row[f"AUPRC_{model}"] = float(metrics.loc[model, "AUPRC"])
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_study(
    train: pd.DataFrame,
    cal: pd.DataFrame,
    test: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """Remove feature families one at a time and evaluate HGB AUPRC."""
    feature_sets = {
        "Full": FEATURES,
        "No temporal": [f for f in FEATURES if f not in TEMPORAL_FEATURES],
        "No context": [f for f in FEATURES if f not in CONTEXT_FEATURES],
        "No classical": [f for f in FEATURES if f not in CLASSICAL_FEATURES],
    }
    age_fill_value = float(train["mean_age"].median())
    rows = []
    for setting, feats in feature_sets.items():
        X_train = prepare_features(train, feats, age_fill_value)
        X_cal = prepare_features(cal, feats, age_fill_value)
        X_test = prepare_features(test, feats, age_fill_value)
        hgb = make_hgb(seed)
        hgb.fit(X_train, train["emerging_next_q"])
        hgb_cal = calibrate_prefit(hgb, X_cal, cal["emerging_next_q"])
        p = hgb_cal.predict_proba(X_test)[:, 1]
        rows.append({
            "setting": setting,
            "n_features": len(feats),
            "AUPRC": average_precision_score(test["emerging_next_q"], p),
        })
    return pd.DataFrame(rows).sort_values("AUPRC", ascending=False).reset_index(drop=True)


def save_bundle(bundle: ModelBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str | Path) -> ModelBundle:
    return joblib.load(path)
