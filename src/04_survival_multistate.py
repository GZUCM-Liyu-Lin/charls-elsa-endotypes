from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2, kruskal, chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import AalenJohansenFitter, CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

warnings.filterwarnings("ignore")
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import OUT_DIR, ensure_dirs
ensure_dirs()
AXES = ["hgi", "rcii", "epwv", "whtr", "hgs_bmi"]

def fmt_mean_sd(s):
    s = s.dropna()
    return f"{s.mean():.2f} ± {s.std():.2f}" if len(s) else "NA"

def fmt_med_iqr(s):
    s = s.dropna()
    return f"{s.median():.2f} ({s.quantile(0.25):.2f}, {s.quantile(0.75):.2f})" if len(s) else "NA"

def build_table1(df, cohort_name, cluster_col="endotype"):
    k = int(df[cluster_col].max())
    cont_vars = [
        ("age", "Age, years"), ("hgi", "HGI, %"), ("rcii", "RCII, ln(mg/dL*mg/L/10)"),
        ("epwv", "ePWV, m/s"), ("whtr", "WHtR"), ("hgs_bmi", "HGS/BMI, kg/(kg/m2)"),
        ("glu", "Fasting glucose, mg/dL"), ("hba1c", "HbA1c, %"), ("rc", "Remnant cholesterol, mg/dL"),
        ("crp", "hs-CRP, mg/L"), ("sbp", "Systolic BP, mmHg"), ("dbp", "Diastolic BP, mmHg"),
        ("waist", "Waist circumference, cm"), ("bmi", "BMI, kg/m2"), ("hgs", "Handgrip strength, kg"),
        ("tyg", "TyG index"), ("egdr", "eGDR, mg/kg/min"),
    ]
    if "egfr" in df.columns:
        cont_vars.append(("egfr", "eGFR, mL/min/1.73m2"))

    cat_vars = [
        ("female", "Female sex"), ("edu_cat", "Education level"),
        ("marry_status", "Married / partnered"), ("smoke_cur", "Current smoker"),
        ("drink_cur", "Current drinker"), ("htn_status", "Hypertension"),
        ("dyslip_status", "Dyslipidemia"), ("base_diabetes_all", "Baseline diabetes"),
        ("base_heart", "Baseline heart disease"), ("base_stroke", "Baseline stroke"),
        ("base_cvd", "Baseline CVD"), ("base_cmm", "Baseline CMM"),
        ("ckm_proxy", "CKM stage proxy"), ("med_htn", "Antihypertensive medication"),
        ("med_lipid", "Lipid-lowering medication"),
    ]
    if "rural_status" in df.columns:
        cat_vars.insert(2, ("rural_status", "Rural residence"))

    rows = []
    # Sample sizes
    n_row = {"variable": "N (%)", "overall": f"{len(df)} (100.0%)"}
    for c in range(1, k + 1):
        nc = int((df[cluster_col] == c).sum())
        n_row[f"E{c}"] = f"{nc} ({100.0 * nc / len(df):.1f}%)"
    n_row["p_value"] = ""
    rows.append(n_row)

    for col, label in cont_vars:
        if col not in df.columns:
            continue
        r = {"variable": f"{label} [Mean ± SD]", "overall": fmt_mean_sd(df[col])}
        r_med = {"variable": f"{label} [Median (IQR)]", "overall": fmt_med_iqr(df[col])}
        groups = [df.loc[df[cluster_col] == c, col].dropna().to_numpy() for c in range(1, k + 1)]
        for c, g in enumerate(groups, 1):
            r[f"E{c}"] = fmt_mean_sd(pd.Series(g))
            r_med[f"E{c}"] = fmt_med_iqr(pd.Series(g))
        try:
            _, p = kruskal(*[g for g in groups if len(g) > 0])
            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
        except Exception:
            p_str = "NA"
        r["p_value"] = p_str
        r_med["p_value"] = p_str
        rows.extend([r, r_med])

    for col, label in cat_vars:
        if col not in df.columns:
            continue
        vals = sorted([v for v in df[col].dropna().unique()])
        ct = pd.crosstab(df[col], df[cluster_col])
        try:
            _, p, _, _ = chi2_contingency(ct) if ct.shape[0] > 1 and ct.shape[1] > 1 else (0, np.nan, 0, None)
            p_str = f"{p:.2e}" if (np.isfinite(p) and p < 0.001) else (f"{p:.4f}" if np.isfinite(p) else "NA")
        except Exception:
            p_str = "NA"
        show_vals = [1.0] if set(vals).issubset({0.0, 1.0}) and 1.0 in vals else vals
        for idx_v, v in enumerate(show_vals):
            vlab = label if len(show_vals) == 1 else f"{label} = {int(v) if float(v).is_integer() else v}"
            r = {"variable": f"{vlab}, n (%)", "overall": f"{int((df[col] == v).sum())} ({100.0 * (df[col] == v).mean():.1f}%)"}
            for c in range(1, k + 1):
                sub = df[df[cluster_col] == c]
                den = sub[col].notna().sum()
                cnt = int((sub[col] == v).sum())
                r[f"E{c}"] = f"{cnt} ({100.0 * cnt / den:.1f}%)" if den else "0 (0.0%)"
            r["p_value"] = p_str if idx_v == 0 else ""
            rows.append(r)

    return pd.DataFrame(rows)

def prep_design(df, exposure_terms, cov_cols, time_col, status_col, cluster_col=None, extra_cols=()):
    cols = list(dict.fromkeys([*exposure_terms, *cov_cols, time_col, status_col, *([cluster_col] if cluster_col else []), *extra_cols]))
    sub = df[cols].dropna().copy()
    sub = sub[sub[time_col] > 0].copy()
    # One-hot encode edu_cat & ckm_proxy if present
    dummies_done = []
    for cat_c in ["edu_cat"]:
        if cat_c in sub.columns:
            cats = sorted(sub[cat_c].unique())
            for cv in cats[1:]:
                cname = f"{cat_c}_{int(cv)}"
                sub[cname] = (sub[cat_c] == cv).astype(float)
                dummies_done.append(cname)
    return sub, dummies_done

def get_cov_list(cohort, tier, endpoint):
    base_m1 = ["age", "female"]
    if cohort == "CHARLS":
        base_m2 = ["age", "female", "edu_cat", "rural_status", "marry_status", "smoke_cur", "drink_cur"]
        clin = ["htn_status", "dyslip_status", "med_htn", "med_lipid", "egfr"]
    else:
        base_m2 = ["age", "female", "edu_cat", "marry_status", "smoke_cur", "drink_cur"]
        clin = ["htn_status", "dyslip_status", "med_htn", "med_lipid"]

    if tier == "Model1":
        return base_m1
    if tier == "Model2":
        return base_m2
    # Model3
    if endpoint == "cmm":
        return [*base_m2, *clin, "n_cmd_base"]
    if endpoint in ("cvd", "heart", "stroke"):
        return [*base_m2, *clin, "base_diabetes_all"]
    if endpoint == "diabetes":
        return [*base_m2, *clin, "base_cvd"]
    if endpoint == "fcmd":
        return [*base_m2, *clin]
    return [*base_m2, *clin, "base_diabetes_all", "base_heart", "base_stroke"]

def fit_cause_specific_cox(df, cat_col, ref_val, cov_cols, time_col, status_col, event_val=1, cluster_col=None, check_ph=False):
    sub, dum_cols = prep_design(df, [cat_col], cov_cols, time_col, status_col, cluster_col=cluster_col)
    sub["ev"] = (sub[status_col] == event_val).astype(int)
    levels = [l for l in sorted(sub[cat_col].unique()) if l != ref_val]
    exp_terms = []
    for lv in levels:
        tname = f"exp_{lv}"
        sub[tname] = (sub[cat_col] == lv).astype(float)
        exp_terms.append((lv, tname))

    model_covs = [c for c in cov_cols if c != "edu_cat"] + dum_cols
    use_cols = [t for _, t in exp_terms] + model_covs
    # Drop zero-variance covariates
    use_cols = [c for c in use_cols if sub[c].nunique() > 1]
    fit_df = sub[[*use_cols, time_col, "ev", *([cluster_col] if cluster_col else [])]].copy()

    cph = CoxPHFitter(penalizer=1e-4)
    if cluster_col:
        cph.fit(fit_df, duration_col=time_col, event_col="ev", cluster_col=cluster_col)
    else:
        cph.fit(fit_df, duration_col=time_col, event_col="ev", robust=True)

    ph_p = np.nan
    if check_ph:
        try:
            cph_ph = cph
            ph_df = fit_df
            if cluster_col:
                ph_df = fit_df.drop(columns=[cluster_col])
                cph_ph = CoxPHFitter(penalizer=1e-4)
                cph_ph.fit(ph_df, duration_col=time_col, event_col="ev", robust=False)
            ph_res = proportional_hazard_test(cph_ph, ph_df, time_transform="rank")
            ph_p = float(ph_res.summary.loc[[t for _, t in exp_terms if t in ph_res.summary.index], "p"].min())
        except Exception:
            ph_p = np.nan

    res_rows = []
    # Reference row
    n_ref = int((sub[cat_col] == ref_val).sum())
    ev_ref = int(sub.loc[sub[cat_col] == ref_val, "ev"].sum())
    py_ref = float(sub.loc[sub[cat_col] == ref_val, time_col].sum())
    res_rows.append({
        "level": ref_val, "n": n_ref, "events": ev_ref, "py": py_ref,
        "rate_1000py": 1000.0 * ev_ref / py_ref if py_ref else np.nan,
        "hr": 1.0, "ci_low": 1.0, "ci_high": 1.0, "p_value": np.nan, "ph_min_p": ph_p,
    })
    for lv, tname in exp_terms:
        hr = float(np.exp(cph.params_[tname]))
        ci_l = float(np.exp(cph.confidence_intervals_.loc[tname, "95% lower-bound"]))
        ci_h = float(np.exp(cph.confidence_intervals_.loc[tname, "95% upper-bound"]))
        pv = float(cph.summary.loc[tname, "p"])
        n_g = int((sub[cat_col] == lv).sum())
        ev_g = int(sub.loc[sub[cat_col] == lv, "ev"].sum())
        py_g = float(sub.loc[sub[cat_col] == lv, time_col].sum())
        res_rows.append({
            "level": lv, "n": n_g, "events": ev_g, "py": py_g,
            "rate_1000py": 1000.0 * ev_g / py_g if py_g else np.nan,
            "hr": hr, "ci_low": ci_l, "ci_high": ci_h, "p_value": pv, "ph_min_p": ph_p,
        })
    return pd.DataFrame(res_rows)

def fit_fine_gray(df, cat_col, ref_val, cov_cols, time_col, status_col):
    """Fine-Gray subdistribution hazard via vectorized IPCW risk-set expansion."""
    sub, dum_cols = prep_design(df, [cat_col], cov_cols, time_col, status_col)
    sub["pid"] = np.arange(len(sub))
    levels = [l for l in sorted(sub[cat_col].unique()) if l != ref_val]
    exp_terms = []
    for lv in levels:
        tname = f"exp_{lv}"
        sub[tname] = (sub[cat_col] == lv).astype(float)
        exp_terms.append((lv, tname))

    model_covs = [c for c in cov_cols if c != "edu_cat"] + dum_cols
    use_cols = [t for _, t in exp_terms] + model_covs
    use_cols = [c for c in use_cols if sub[c].nunique() > 1]

    km_c = KaplanMeierFitter()
    km_c.fit(sub[time_col], event_observed=(sub[status_col] == 0).astype(int))
    max_t = float(sub[time_col].max())

    non_comp = sub.loc[sub[status_col] != 2, [*use_cols, "pid"]].copy()
    non_comp["start"] = 0.0
    non_comp["stop"] = sub.loc[sub[status_col] != 2, time_col].to_numpy()
    non_comp["ev"] = (sub.loc[sub[status_col] != 2, status_col] == 1).astype(int).to_numpy()
    non_comp["w"] = 1.0

    comp_mask = sub[status_col] == 2
    comp_parts = []
    if comp_mask.any():
        comp_sub = sub.loc[comp_mask, [*use_cols, "pid"]].copy()
        comp_t = sub.loc[comp_mask, time_col].to_numpy()
        c0 = comp_sub.copy()
        c0["start"] = 0.0
        c0["stop"] = comp_t
        c0["ev"] = 0
        c0["w"] = 1.0
        comp_parts.append(c0)

        grid = np.arange(0.5, max_t + 0.5, 0.5)
        g_grid = np.maximum(km_c.predict(grid).to_numpy(), 0.05)
        g_comp = np.maximum(km_c.predict(np.maximum(comp_t - 1e-6, 0.0)).to_numpy(), 0.05)

        # Vectorized Cartesian mask of (comp_idx, grid_idx) where grid > comp_t
        later_mask = grid[None, :] > comp_t[:, None]
        r_idx, g_idx = np.where(later_mask)
        if len(r_idx) > 0:
            starts = np.where(g_idx == 0, comp_t[r_idx], np.maximum(comp_t[r_idx], grid[g_idx - 1]))
            stops = np.minimum(grid[g_idx], max_t + 0.01)
            g_starts = np.maximum(km_c.predict(starts).to_numpy(), 0.05)
            ws = np.clip(g_starts / g_comp[r_idx], 0.05, 5.0)
            exp_df = comp_sub.iloc[r_idx].copy()
            exp_df["start"] = starts
            exp_df["stop"] = stops
            exp_df["ev"] = 0
            exp_df["w"] = ws
            comp_parts.append(exp_df)

    long_df = pd.concat([non_comp, *comp_parts], ignore_index=True)
    long_df = long_df[long_df["stop"] > long_df["start"] + 1e-6].copy()

    cph = CoxPHFitter(penalizer=1e-4)
    cph.fit(
        long_df[[*use_cols, "start", "stop", "ev", "w"]],
        duration_col="stop", entry_col="start", event_col="ev",
        weights_col="w", robust=True,
    )
    out = {ref_val: (1.0, 1.0, 1.0, np.nan)}
    for lv, tname in exp_terms:
        shr = float(np.exp(cph.params_[tname]))
        ci_l = float(np.exp(cph.confidence_intervals_.loc[tname, "95% lower-bound"]))
        ci_h = float(np.exp(cph.confidence_intervals_.loc[tname, "95% upper-bound"]))
        pv = float(cph.summary.loc[tname, "p"])
        out[lv] = (shr, ci_l, ci_h, pv)
    return out

def compute_cif_curves(df, cohort, endpoint, time_col, status_col, time_grid):
    sub = df[["endotype", "endotype_label", time_col, status_col]].dropna().copy()
    sub = sub[sub[time_col] > 0]
    rows = []
    # Gray-type unadjusted Fine-Gray p-value (event=1)
    fg_unadj = fit_fine_gray(df.dropna(subset=[time_col, status_col]), "endotype", 1, [], time_col, status_col)
    min_p_fg = min([v[3] for k, v in fg_unadj.items() if k != 1 and np.isfinite(v[3])], default=np.nan)

    for eid in sorted(sub["endotype"].unique()):
        g = sub[sub["endotype"] == eid]
        elab = g["endotype_label"].iloc[0]
        for ev_code, ev_name in [(1, "target_event"), (2, "competing_death")]:
            if (g[status_col] == ev_code).sum() == 0:
                continue
            aj = AalenJohansenFitter(calculate_variance=True)
            aj.fit(g[time_col], g[status_col], event_of_interest=ev_code)
            max_obs = float(g[time_col].max())
            for t in time_grid:
                if t > max_obs:
                    continue
                # Nearest index <= t
                idx_pos = aj.cumulative_density_.index.searchsorted(t, side="right") - 1
                if idx_pos < 0:
                    est, lo, hi = 0.0, 0.0, 0.0
                else:
                    est = float(aj.cumulative_density_.iloc[idx_pos, 0])
                    lo = float(aj.confidence_interval_.iloc[idx_pos, 0])
                    hi = float(aj.confidence_interval_.iloc[idx_pos, 1])
                rows.append({
                    "cohort": cohort, "endpoint": endpoint, "endotype": int(eid), "endotype_label": elab,
                    "event_type": ev_name, "time_yr": float(t), "cif": est, "ci_low": lo, "ci_high": hi,
                    "n_at_risk": int((g[time_col] >= t).sum()), "gray_type_min_p": min_p_fg,
                })
    return pd.DataFrame(rows)

def rcs_basis(x, knots):
    """Harrell 4-knot restricted cubic spline basis (k=4 -> 3 columns: x, s1, s2)."""
    t1, t2, t3, t4 = knots
    def dpos(u):
        return np.maximum(u, 0.0) ** 3
    norm = (t4 - t1) ** 2
    s1 = (dpos(x - t1) - dpos(x - t3) * (t4 - t1) / (t4 - t3) + dpos(x - t4) * (t3 - t1) / (t4 - t3)) / norm
    s2 = (dpos(x - t2) - dpos(x - t3) * (t4 - t2) / (t4 - t3) + dpos(x - t4) * (t3 - t2) / (t4 - t3)) / norm
    return np.column_stack([x, s1, s2])

def fit_rcs_axis(df, axis_col, cov_cols, time_col, status_col, cluster_col=None):
    sub, dum_cols = prep_design(df, [axis_col, "female"], cov_cols, time_col, status_col, cluster_col=cluster_col)
    sub["ev"] = (sub[status_col] == 1).astype(int)
    # Use sex-specific Z-score so curve is harmonized across sex
    z_col = f"{axis_col}_z_local"
    for fv in [0.0, 1.0]:
        m = sub["female"] == fv
        sub.loc[m, z_col] = (sub.loc[m, axis_col] - sub.loc[m, axis_col].mean()) / sub.loc[m, axis_col].std(ddof=0)
    x_arr = sub[z_col].to_numpy()
    knots = np.percentile(x_arr, [5, 35, 65, 95])
    B = rcs_basis(x_arr, knots)
    sub["rcs0"], sub["rcs1"], sub["rcs2"] = B[:, 0], B[:, 1], B[:, 2]

    model_covs = [c for c in cov_cols if c != "edu_cat"] + dum_cols
    use_cols = ["rcs0", "rcs1", "rcs2"] + [c for c in model_covs if sub[c].nunique() > 1]
    fit_df = sub[[*use_cols, time_col, "ev", *([cluster_col] if cluster_col else [])]].copy()

    cph = CoxPHFitter(penalizer=1e-4)
    if cluster_col:
        cph.fit(fit_df, duration_col=time_col, event_col="ev", cluster_col=cluster_col)
    else:
        cph.fit(fit_df, duration_col=time_col, event_col="ev", robust=True)

    b_est = cph.params_[["rcs0", "rcs1", "rcs2"]].to_numpy()
    V = cph.variance_matrix_.loc[["rcs0", "rcs1", "rcs2"], ["rcs0", "rcs1", "rcs2"]].to_numpy()
    chi_all = float(b_est @ np.linalg.solve(V, b_est))
    p_overall = float(1.0 - chi2.cdf(chi_all, 3))
    chi_nl = float(b_est[1:] @ np.linalg.solve(V[1:, 1:], b_est[1:]))
    p_nl = float(1.0 - chi2.cdf(chi_nl, 2))

    x_ref = float(np.median(x_arr))
    b_ref = rcs_basis(np.array([x_ref]), knots)[0]
    x_grid = np.linspace(float(np.percentile(x_arr, 1)), float(np.percentile(x_arr, 99)), 200)
    B_grid = rcs_basis(x_grid, knots) - b_ref
    lhrs = B_grid @ b_est
    ses = np.sqrt(np.einsum("ij,jk,ik->i", B_grid, V, B_grid))
    return pd.DataFrame({
        "axis": axis_col, "z_value": x_grid, "z_ref": x_ref,
        "hr": np.exp(lhrs), "ci_low": np.exp(lhrs - 1.96 * ses), "ci_high": np.exp(lhrs + 1.96 * ses),
        "p_overall": p_overall, "p_nonlinear": p_nl, "n": len(sub), "events": int(sub["ev"].sum()),
    })

def fit_continuous_exposures(df, cohort, endpoint, time_col, status_col, cluster_col=None):
    rows = []
    # Within-sex Z-score inside risk set
    work = df.copy()
    for col in [*AXES, "tyg", "egdr"]:
        if col not in work.columns:
            continue
        z_name = f"{col}_z"
        for fv in [0.0, 1.0]:
            m = work["female"] == fv
            work.loc[m, z_name] = (work.loc[m, col] - work.loc[m, col].mean()) / work.loc[m, col].std(ddof=0)

    for tier in ["Model2", "Model3"]:
        cov_cols = get_cov_list(cohort, tier, endpoint)
        for col in [*AXES, "tyg", "egdr"]:
            z_name = f"{col}_z"
            if z_name not in work.columns:
                continue
            sub, dum_cols = prep_design(work, [z_name], cov_cols, time_col, status_col, cluster_col=cluster_col)
            sub["ev"] = (sub[status_col] == 1).astype(int)
            model_covs = [c for c in cov_cols if c != "edu_cat"] + dum_cols
            use_cols = [z_name] + [c for c in model_covs if sub[c].nunique() > 1]
            fit_df = sub[[*use_cols, time_col, "ev", *([cluster_col] if cluster_col else [])]].copy()
            cph = CoxPHFitter(penalizer=1e-4)
            if cluster_col:
                cph.fit(fit_df, duration_col=time_col, event_col="ev", cluster_col=cluster_col)
            else:
                cph.fit(fit_df, duration_col=time_col, event_col="ev", robust=True)
            rows.append({
                "cohort": cohort, "endpoint": endpoint, "tier": tier, "mode": "Individual_PerSD",
                "exposure": col, "n": len(sub), "events": int(sub["ev"].sum()),
                "hr": float(np.exp(cph.params_[z_name])),
                "ci_low": float(np.exp(cph.confidence_intervals_.loc[z_name, "95% lower-bound"])),
                "ci_high": float(np.exp(cph.confidence_intervals_.loc[z_name, "95% upper-bound"])),
                "p_value": float(cph.summary.loc[z_name, "p"]),
            })

        # Simultaneous 5 axes
        z_axes = [f"{a}_z" for a in AXES]
        sub, dum_cols = prep_design(work, z_axes, cov_cols, time_col, status_col, cluster_col=cluster_col)
        sub["ev"] = (sub[status_col] == 1).astype(int)
        model_covs = [c for c in cov_cols if c != "edu_cat"] + dum_cols
        use_cols = z_axes + [c for c in model_covs if sub[c].nunique() > 1]
        fit_df = sub[[*use_cols, time_col, "ev", *([cluster_col] if cluster_col else [])]].copy()
        cph = CoxPHFitter(penalizer=1e-4)
        if cluster_col:
            cph.fit(fit_df, duration_col=time_col, event_col="ev", cluster_col=cluster_col)
        else:
            cph.fit(fit_df, duration_col=time_col, event_col="ev", robust=True)
        for a, zn in zip(AXES, z_axes):
            rows.append({
                "cohort": cohort, "endpoint": endpoint, "tier": tier, "mode": "Simultaneous_5Axes_PerSD",
                "exposure": a, "n": len(sub), "events": int(sub["ev"].sum()),
                "hr": float(np.exp(cph.params_[zn])),
                "ci_low": float(np.exp(cph.confidence_intervals_.loc[zn, "95% lower-bound"])),
                "ci_high": float(np.exp(cph.confidence_intervals_.loc[zn, "95% upper-bound"])),
                "p_value": float(cph.summary.loc[zn, "p"]),
            })
    return pd.DataFrame(rows)

def run_multistate(df, cohort, cluster_col=None):
    """Multi-state transitions among CMD-free (n_cmd_base == 0): T1-T4 + T5 (CMM->Death)."""
    cmd_free = df[df["n_cmd_base"] == 0].copy()
    k = int(df["endotype"].max())
    rows = []

    def add_res(trans_name, tier, cs_df):
        for _, r in cs_df.iterrows():
            rows.append({
                "cohort": cohort, "transition": trans_name, "tier": tier,
                "endotype": int(r["level"]), "n": int(r["n"]), "events": int(r["events"]),
                "py": float(r["py"]), "hr": float(r["hr"]),
                "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]), "p_value": float(r["p_value"]),
            })

    # T1: CMD-free -> First CMD; T2: CMD-free -> Death without CMD
    for tier in ["Model2", "Model3"]:
        covs = get_cov_list(cohort, tier, "fcmd")
        add_res("T1_Free_to_FirstCMD", tier, fit_cause_specific_cox(cmd_free, "endotype", 1, covs, "time_fcmd", "status_fcmd", 1, cluster_col))
        add_res("T2_Free_to_Death", tier, fit_cause_specific_cox(cmd_free, "endotype", 1, covs, "time_fcmd", "status_fcmd", 2, cluster_col))

    # T3 & T4: Among those who developed First CMD (status_fcmd == 1) BEFORE simultaneous CMM
    fcmd_pos = cmd_free[cmd_free["status_fcmd"] == 1].copy()
    # Exclude simultaneous FCMD+CMM if first_wave_fcmd == first_wave_cmm (or time_cmm <= time_fcmd + 0.05)
    # Actually keep note of simultaneous vs sequential: for T3/T4 risk set after FCMD, require time_cmm > time_fcmd + 0.05 (if CMM occurred or censored/died after FCMD)
    post_fcmd = fcmd_pos[(fcmd_pos["time_cmm"] > fcmd_pos["time_fcmd"] + 0.05)].copy()
    post_fcmd["dur_after_fcmd"] = post_fcmd["time_cmm"] - post_fcmd["time_fcmd"]
    print(f"  [{cohort}] Multi-state FCMD={len(fcmd_pos)}, non-simultaneous post-FCMD={len(post_fcmd)}, T3_CMM_ev={int((post_fcmd['status_cmm']==1).sum())}, T4_Death_ev={int((post_fcmd['status_cmm']==2).sum())}")

    for tier in ["Model2", "Model3"]:
        covs = get_cov_list(cohort, tier, "fcmd")
        # Clock-reset (time since FCMD onset)
        if (post_fcmd["status_cmm"] == 1).sum() >= 20:
            add_res("T3_FirstCMD_to_CMM_clockreset", tier, fit_cause_specific_cox(post_fcmd, "endotype", 1, covs, "dur_after_fcmd", "status_cmm", 1, cluster_col))
        if (post_fcmd["status_cmm"] == 2).sum() >= 15:
            add_res("T4_FirstCMD_to_Death_clockreset", tier, fit_cause_specific_cox(post_fcmd, "endotype", 1, covs, "dur_after_fcmd", "status_cmm", 2, cluster_col))

        # Clock-forward (entry = time_fcmd, stop = time_cmm)
        if (post_fcmd["status_cmm"] == 1).sum() >= 20:
            sub, dum_cols = prep_design(post_fcmd, ["endotype"], covs, "time_cmm", "status_cmm", cluster_col=cluster_col, extra_cols=["time_fcmd"])
            sub["ev"] = (sub["status_cmm"] == 1).astype(int)
            exp_terms = [(lv, f"exp_{lv}") for lv in range(2, k + 1)]
            for lv, tn in exp_terms:
                sub[tn] = (sub["endotype"] == lv).astype(float)
            mcovs = [c for c in covs if c != "edu_cat"] + dum_cols
            ucols = [t for _, t in exp_terms] + [c for c in mcovs if sub[c].nunique() > 1]
            cph = CoxPHFitter(penalizer=1e-4)
            fdf = sub[[*ucols, "time_fcmd", "time_cmm", "ev", *([cluster_col] if cluster_col else [])]]
            if cluster_col:
                cph.fit(fdf, duration_col="time_cmm", entry_col="time_fcmd", event_col="ev", cluster_col=cluster_col)
            else:
                cph.fit(fdf, duration_col="time_cmm", entry_col="time_fcmd", event_col="ev", robust=True)
            rows.append({"cohort": cohort, "transition": "T3_FirstCMD_to_CMM_clockforward", "tier": tier, "endotype": 1, "n": int((sub["endotype"]==1).sum()), "events": int(sub.loc[sub["endotype"]==1,"ev"].sum()), "py": float((sub.loc[sub["endotype"]==1,"time_cmm"]-sub.loc[sub["endotype"]==1,"time_fcmd"]).sum()), "hr": 1.0, "ci_low": 1.0, "ci_high": 1.0, "p_value": np.nan})
            for lv, tn in exp_terms:
                rows.append({"cohort": cohort, "transition": "T3_FirstCMD_to_CMM_clockforward", "tier": tier, "endotype": lv, "n": int((sub["endotype"]==lv).sum()), "events": int(sub.loc[sub["endotype"]==lv,"ev"].sum()), "py": float((sub.loc[sub["endotype"]==lv,"time_cmm"]-sub.loc[sub["endotype"]==lv,"time_fcmd"]).sum()), "hr": float(np.exp(cph.params_[tn])), "ci_low": float(np.exp(cph.confidence_intervals_.loc[tn,"95% lower-bound"])), "ci_high": float(np.exp(cph.confidence_intervals_.loc[tn,"95% upper-bound"])), "p_value": float(cph.summary.loc[tn,"p"])})

    # T5: Baseline CMM -> Death (if enough events)
    cmm_prev = df[df["base_cmm"] == 1].copy()
    if (cmm_prev["status_death"] == 1).sum() >= 25:
        for tier in ["Model2", "Model3"]:
            covs = get_cov_list(cohort, tier, "death")
            add_res("T5_PrevalentCMM_to_Death", tier, fit_cause_specific_cox(cmm_prev, "endotype", 1, covs, "time_death", "status_death", 1, cluster_col))

    return pd.DataFrame(rows)

def main():
    charls = pd.read_csv(OUT_DIR / "derived" / "charls_clustered.csv", index_col="ID", low_memory=False)
    elsa = pd.read_csv(OUT_DIR / "derived" / "elsa_clustered.csv", index_col="idauniq", low_memory=False)
    ch_el = charls[charls["eligible_w3"] == 1].copy()
    el_el = elsa[elsa["eligible_w4"] == 1].copy()
    k = int(ch_el["endotype"].max())
    name_map = ch_el.groupby("endotype")["endotype_label"].first().to_dict()

    # 1. Baseline Table 1 & Table S1
    t1_ch = build_table1(ch_el, "CHARLS")
    t1_ch.to_csv(OUT_DIR / "tables" / "table1_baseline_charls.csv", index=False, encoding="utf-8-sig")
    t1_el = build_table1(el_el, "ELSA")
    t1_el.to_csv(OUT_DIR / "tables" / "tableS1_baseline_elsa.csv", index=False, encoding="utf-8-sig")

    # 2. Primary Landmark Models (CHARLS & ELSA) + CIFs
    endpoints = [("cmm", "time_cmm", "status_cmm"), ("cvd", "time_cvd", "status_cvd"), ("heart", "time_heart", "status_heart"), ("stroke", "time_stroke", "status_stroke"), ("diabetes", "time_diabetes", "status_diabetes"), ("death", "time_death", "status_death")]

    for cohort_name, df_c, cl_col, tgrid in [
        ("CHARLS", ch_el, "communityID", np.linspace(0, 7.0, 29)),
        ("ELSA", el_el, None, np.linspace(0, 14.0, 29)),
    ]:
        prim_rows = []
        cif_list = []
        for ep, tcol, scol in endpoints:
            print(f"  Running {cohort_name} primary endpoint: {ep}...", flush=True)
            sub_ep = df_c.dropna(subset=[tcol, scol]).copy()
            for tier in ["Model1", "Model2", "Model3"]:
                covs = get_cov_list(cohort_name, tier, ep)
                cs = fit_cause_specific_cox(sub_ep, "endotype", 1, covs, tcol, scol, event_val=1, cluster_col=cl_col, check_ph=(tier == "Model3"))
                if ep != "death" and (sub_ep[scol] == 2).sum() >= 10:
                    fg = fit_fine_gray(sub_ep, "endotype", 1, covs, tcol, scol)
                else:
                    fg = {eid: (np.nan, np.nan, np.nan, np.nan) for eid in range(1, k + 1)}
                for _, r in cs.iterrows():
                    eid = int(r["level"])
                    shr, sl, sh, sp = fg.get(eid, (np.nan, np.nan, np.nan, np.nan))
                    prim_rows.append({
                        "cohort": cohort_name, "endpoint": ep, "tier": tier,
                        "endotype": eid, "label": name_map[eid],
                        "n": int(r["n"]), "events": int(r["events"]), "py": float(r["py"]),
                        "rate_1000py": float(r["rate_1000py"]),
                        "cs_hr": float(r["hr"]), "cs_ci_low": float(r["ci_low"]), "cs_ci_high": float(r["ci_high"]), "cs_p": float(r["p_value"]) if pd.notna(r["p_value"]) else np.nan,
                        "fg_shr": shr, "fg_ci_low": sl, "fg_ci_high": sh, "fg_p": sp,
                        "ph_min_p": float(r["ph_min_p"]) if pd.notna(r["ph_min_p"]) else np.nan,
                    })
            if ep in ("cmm", "cvd", "heart", "stroke", "diabetes"):
                cif_list.append(compute_cif_curves(sub_ep, cohort_name, ep, tcol, scol, tgrid))
        prim_df = pd.DataFrame(prim_rows)
        fn = "table2_primary_charls.csv" if cohort_name == "CHARLS" else "table2_primary_elsa.csv"
        prim_df.to_csv(OUT_DIR / "tables" / fn, index=False, encoding="utf-8-sig")
        pd.concat(cif_list, ignore_index=True).to_csv(OUT_DIR / "figure_data" / f"fig3_cif_{cohort_name.lower()}.csv", index=False, encoding="utf-8-sig")
        if cohort_name == "CHARLS":
            ch_prim = prim_df
        else:
            el_prim = prim_df

    pd.concat([ch_prim, el_prim], ignore_index=True).to_csv(OUT_DIR / "figure_data" / "fig3_forest_endotypes.csv", index=False, encoding="utf-8-sig")

    # 3. Single-Axis Benchmarks & 4-Knot RCS (CHARLS primary)
    s_axes_list = []
    for ep, tcol, scol in [("cmm", "time_cmm", "status_cmm"), ("cvd", "time_cvd", "status_cvd")]:
        s_axes_list.append(fit_continuous_exposures(ch_el.dropna(subset=[tcol, scol]), "CHARLS", ep, tcol, scol, "communityID"))
        s_axes_list.append(fit_continuous_exposures(el_el.dropna(subset=[tcol, scol]), "ELSA", ep, tcol, scol, None))
    pd.concat(s_axes_list, ignore_index=True).to_csv(OUT_DIR / "tables" / "tableS2_single_axes.csv", index=False, encoding="utf-8-sig")

    for ep, tcol, scol in [("cmm", "time_cmm", "status_cmm"), ("cvd", "time_cvd", "status_cvd")]:
        rcs_dfs = []
        sub_ep = ch_el.dropna(subset=[tcol, scol]).copy()
        covs_m3 = get_cov_list("CHARLS", "Model3", ep)
        for ax in AXES:
            rcs_dfs.append(fit_rcs_axis(sub_ep, ax, covs_m3, tcol, scol, "communityID"))
        pd.concat(rcs_dfs, ignore_index=True).to_csv(OUT_DIR / "figure_data" / f"fig5_rcs_{ep}.csv", index=False, encoding="utf-8-sig")

    # 4. Multi-State Models
    ms_ch = run_multistate(ch_el, "CHARLS", "communityID")
    ms_ch.to_csv(OUT_DIR / "tables" / "table3_multistate_charls.csv", index=False, encoding="utf-8-sig")
    ms_el = run_multistate(el_el, "ELSA", None)
    ms_el.to_csv(OUT_DIR / "tables" / "table3_multistate_elsa.csv", index=False, encoding="utf-8-sig")
    pd.concat([ms_ch, ms_el], ignore_index=True).to_csv(OUT_DIR / "figure_data" / "fig4_multistate_hrs.csv", index=False, encoding="utf-8-sig")

    # 5. 4-Year Transition Models (CHARLS W1->W3 & ELSA W2->W4)
    for cohort_name, df_c, cl_col, fn in [
        ("CHARLS", ch_el[ch_el["trans_group"].notna()].copy(), "communityID", "table4_transitions_charls.csv"),
        ("ELSA", el_el[el_el["trans_group"].notna()].copy(), None, "table4_transitions_elsa.csv"),
    ]:
        t_rows = []
        for ep, tcol, scol in [("cmm", "time_cmm", "status_cmm"), ("cvd", "time_cvd", "status_cvd")]:
            sub_ep = df_c.dropna(subset=[tcol, scol]).copy()
            for tier in ["Model1", "Model2", "Model3"]:
                covs = get_cov_list(cohort_name, tier, ep)
                cs = fit_cause_specific_cox(sub_ep, "trans_group", "1_Stable_Low", covs, tcol, scol, 1, cl_col)
                if (sub_ep[scol] == 2).sum() >= 10:
                    fg = fit_fine_gray(sub_ep, "trans_group", "1_Stable_Low", covs, tcol, scol)
                else:
                    fg = {}
                for _, r in cs.iterrows():
                    lv = r["level"]
                    n_ev = int(r["events"])
                    shr, sl, sh, sp = fg.get(lv, (np.nan, np.nan, np.nan, np.nan))
                    is_ref = str(lv).startswith("1_")
                    if (not is_ref) and n_ev == 0:
                        cs_hr, cs_l, cs_h, cs_p = np.nan, np.nan, np.nan, np.nan
                        shr, sl, sh, sp = np.nan, np.nan, np.nan, np.nan
                    else:
                        cs_hr, cs_l, cs_h = float(r["hr"]), float(r["ci_low"]), float(r["ci_high"])
                        cs_p = float(r["p_value"]) if pd.notna(r["p_value"]) else np.nan
                    t_rows.append({
                        "cohort": cohort_name, "endpoint": ep, "tier": tier, "trans_group": lv,
                        "n": int(r["n"]), "events": n_ev, "py": float(r["py"]),
                        "rate_1000py": float(r["rate_1000py"]),
                        "cs_hr": cs_hr, "cs_ci_low": cs_l, "cs_ci_high": cs_h, "cs_p": cs_p,
                        "fg_shr": shr, "fg_ci_low": sl, "fg_ci_high": sh, "fg_p": sp,
                    })
        tdf = pd.DataFrame(t_rows)
        tdf.to_csv(OUT_DIR / "tables" / fn, index=False, encoding="utf-8-sig")
        if cohort_name == "CHARLS":
            tdf_ch = tdf
        else:
            tdf_el = tdf
    pd.concat([tdf_ch, tdf_el], ignore_index=True).to_csv(OUT_DIR / "figure_data" / "fig4_transition_hrs.csv", index=False, encoding="utf-8-sig")

    # 6. Subgroups & Interactions (CHARLS CMM & CVD, Model 3)
    sg_rows = []
    ch_sg = ch_el.copy()
    ch_sg["age_ge60"] = (ch_sg["age"] >= 60).astype(int)
    sg_defs = [
        ("Age", "age_ge60", {0: "<60 years", 1: ">=60 years"}),
        ("Sex", "female", {0.0: "Male", 1.0: "Female"}),
        ("Residence", "rural_status", {0.0: "Urban", 1.0: "Rural"}),
        ("Baseline_CMD", "n_cmd_base", {0: "0 CMD", 1: "1 CMD"}),
        ("CKM_Proxy", "ckm_proxy", {1: "Stage 0-1", 2: "Stage 2", 3: "Stage 3"}),
    ]
    for ep, tcol, scol in [("cmm", "time_cmm", "status_cmm"), ("cvd", "time_cvd", "status_cvd")]:
        sub_ep = ch_sg.dropna(subset=[tcol, scol]).copy()
        covs_m3 = get_cov_list("CHARLS", "Model3", ep)
        for sg_name, sg_col, lv_map in sg_defs:
            if ep == "cvd" and sg_col == "n_cmd_base":
                continue
            # Multiplicative interaction P via Wald test in full sample
            int_p = np.nan
            try:
                int_sub, dum_cols = prep_design(sub_ep, ["endotype", sg_col], covs_m3, tcol, scol, "communityID")
                int_sub["ev"] = (int_sub[scol] == 1).astype(int)
                int_sub["high_risk_any"] = (int_sub["endotype"] > 1).astype(float)
                int_sub["sg_bin"] = (int_sub[sg_col] != min(lv_map.keys())).astype(float)
                int_sub["inter"] = int_sub["high_risk_any"] * int_sub["sg_bin"]
                mcovs = [c for c in covs_m3 if c not in ("edu_cat", sg_col)] + dum_cols
                ucols = ["high_risk_any", "sg_bin", "inter"] + [c for c in mcovs if int_sub[c].nunique() > 1]
                cph_i = CoxPHFitter(penalizer=1e-4)
                cph_i.fit(int_sub[[*ucols, tcol, "ev", "communityID"]], duration_col=tcol, event_col="ev", cluster_col="communityID")
                int_p = float(cph_i.summary.loc["inter", "p"])
            except Exception:
                pass

            for lv_val, lv_lab in lv_map.items():
                s_sub = sub_ep[sub_ep[sg_col] == lv_val].copy()
                if len(s_sub) < 100 or (s_sub[scol] == 1).sum() < 15:
                    continue
                sub_covs = [c for c in covs_m3 if c != sg_col]
                cs = fit_cause_specific_cox(s_sub, "endotype", 1, sub_covs, tcol, scol, 1, "communityID")
                for _, r in cs.iterrows():
                    eid = int(r["level"])
                    sg_rows.append({
                        "endpoint": ep, "subgroup_var": sg_name, "subgroup_level": lv_lab,
                        "endotype": eid, "label": name_map[eid],
                        "n": int(r["n"]), "events": int(r["events"]),
                        "hr": float(r["hr"]), "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]),
                        "p_value": float(r["p_value"]) if pd.notna(r["p_value"]) else np.nan,
                        "p_interaction_any_high": int_p,
                    })
    sg_df = pd.DataFrame(sg_rows)
    sg_df.to_csv(OUT_DIR / "tables" / "tableS3_subgroups.csv", index=False, encoding="utf-8-sig")
    sg_df.to_csv(OUT_DIR / "figure_data" / "fig5_subgroups.csv", index=False, encoding="utf-8-sig")

    # 7. Wave 6 Exclusive Secondary Endpoints
    run_w6_extended(ch_el, k, name_map)
    print("Step 4 survival, competing risk, multi-state, transitions, subgroups, and W6 endpoints complete!")

def _entry():
    import sys
    if "--patch-step4" in sys.argv:
        patch_step4()
    else:
        main()

def run_w6_extended(ch_el, k, name_map):
    w6_rows = []
    w6_eps = [
        ("w6_dementia_atrophy", ch_el[ch_el["base_memory"] == 0].copy(), "New Dementia or Brain Atrophy (W6)"),
        ("w6_dementia", ch_el[ch_el["base_memory"] == 0].copy(), "New Dementia (W6)"),
        ("w6_atrophy", ch_el[ch_el["base_memory"] == 0].copy(), "New Brain Atrophy (W6)"),
        ("w6_cm_hosp", ch_el.copy(), "Cardiometabolic Inpatient Admission (W6)"),
        ("w6_card_hosp", ch_el.copy(), "Cardiology/Cardiac Surgery Admission (W6)"),
        ("w6_heart_attack", ch_el[ch_el["base_heart"] == 1].copy(), "Heart Attack Among Baseline Heart Disease (W6)"),
    ]
    for ep_col, base_df, ep_desc in w6_eps:
        for tier in ["Model2", "Model3"]:
            covs = get_cov_list("CHARLS", tier, "death")
            sub = base_df.dropna(subset=[ep_col, *covs, "communityID", "endotype"]).copy()
            sub["endotype"] = sub["endotype"].astype(int)
            if len(sub) < 100 or sub[ep_col].sum() < 15:
                continue
            covs_use = [c for c in covs if sub[c].nunique() > 1]
            cov_f = " + ".join([c if c != "edu_cat" else "C(edu_cat)" for c in covs_use])
            try:
                mod = smf.glm(f"{ep_col} ~ C(endotype, Treatment(reference=1)) + {cov_f}", data=sub, family=sm.families.Poisson())
                res = mod.fit(cov_type="cluster", cov_kwds={"groups": sub["communityID"]})
                s1 = sub[sub["endotype"] == 1]
                w6_rows.append({"endpoint": ep_col, "description": ep_desc, "tier": tier, "endotype": 1, "label": name_map[1], "n": len(s1), "events": int(s1[ep_col].sum()), "pct": 100.0 * s1[ep_col].mean(), "rr": 1.0, "ci_low": 1.0, "ci_high": 1.0, "p_value": np.nan})
                for eid in range(2, k + 1):
                    tname = f"C(endotype, Treatment(reference=1))[T.{eid}]"
                    if tname in res.params:
                        lo, hi = res.conf_int().loc[tname]
                        se = sub[sub["endotype"] == eid]
                        w6_rows.append({"endpoint": ep_col, "description": ep_desc, "tier": tier, "endotype": eid, "label": name_map[eid], "n": len(se), "events": int(se[ep_col].sum()), "pct": 100.0 * se[ep_col].mean(), "rr": float(np.exp(res.params[tname])), "ci_low": float(np.exp(lo)), "ci_high": float(np.exp(hi)), "p_value": float(res.pvalues[tname])})
            except Exception as exc:
                print(f"W6 endpoint {ep_col} {tier} failed: {exc}")
    w6_df = pd.DataFrame(w6_rows)
    w6_df.to_csv(OUT_DIR / "tables" / "tableS4_w6_extended.csv", index=False, encoding="utf-8-sig")
    w6_df.to_csv(OUT_DIR / "figure_data" / "fig5_w6_extended.csv", index=False, encoding="utf-8-sig")
    return w6_df

def patch_step4():
    """Fast targeted patch for Fixes A (W6 int cast), B (CHARLS PH test), and C (0-event masking)."""
    charls = pd.read_csv(OUT_DIR / "derived" / "charls_clustered.csv", index_col="ID", low_memory=False)
    ch_el = charls[charls["eligible_w3"] == 1].copy()
    k = int(ch_el["endotype"].max())
    name_map = ch_el.groupby("endotype")["endotype_label"].first().to_dict()

    # Fix A: W6 extended endpoints
    w6_df = run_w6_extended(ch_el, k, name_map)
    print(f"Fix A complete: tableS4_w6_extended.csv rows = {len(w6_df)}")

    # Fix B: CHARLS Model 3 ph_min_p
    p_ch = pd.read_csv(OUT_DIR / "tables" / "table2_primary_charls.csv")
    endpoints = [("cmm", "time_cmm", "status_cmm"), ("cvd", "time_cvd", "status_cvd"), ("heart", "time_heart", "status_heart"), ("stroke", "time_stroke", "status_stroke"), ("diabetes", "time_diabetes", "status_diabetes"), ("death", "time_death", "status_death")]
    for ep, tcol, scol in endpoints:
        sub_ep = ch_el.dropna(subset=[tcol, scol]).copy()
        covs = get_cov_list("CHARLS", "Model3", ep)
        cs_ph = fit_cause_specific_cox(sub_ep, "endotype", 1, covs, tcol, scol, event_val=1, cluster_col="communityID", check_ph=True)
        ph_val = float(cs_ph["ph_min_p"].iloc[0])
        p_ch.loc[(p_ch["endpoint"] == ep) & (p_ch["tier"] == "Model3"), "ph_min_p"] = ph_val
        print(f"  Fix B {ep} Model 3 ph_min_p = {ph_val:.4f}")
    p_ch.to_csv(OUT_DIR / "tables" / "table2_primary_charls.csv", index=False, encoding="utf-8-sig")
    p_el = pd.read_csv(OUT_DIR / "tables" / "table2_primary_elsa.csv")
    pd.concat([p_ch, p_el], ignore_index=True).to_csv(OUT_DIR / "figure_data" / "fig3_forest_endotypes.csv", index=False, encoding="utf-8-sig")

    # Fix C: Mask 0-event sparse separation in transition tables
    tr_ch = pd.read_csv(OUT_DIR / "tables" / "table4_transitions_charls.csv")
    tr_el = pd.read_csv(OUT_DIR / "tables" / "table4_transitions_elsa.csv")
    for tdf in (tr_ch, tr_el):
        mask = (~tdf["trans_group"].astype(str).str.startswith("1_")) & (tdf["events"] == 0)
        tdf.loc[mask, ["cs_hr", "cs_ci_low", "cs_ci_high", "cs_p", "fg_shr", "fg_ci_low", "fg_ci_high", "fg_p"]] = np.nan
    tr_ch.to_csv(OUT_DIR / "tables" / "table4_transitions_charls.csv", index=False, encoding="utf-8-sig")
    tr_el.to_csv(OUT_DIR / "tables" / "table4_transitions_elsa.csv", index=False, encoding="utf-8-sig")
    pd.concat([tr_ch, tr_el], ignore_index=True).to_csv(OUT_DIR / "figure_data" / "fig4_transition_hrs.csv", index=False, encoding="utf-8-sig")
    print("Fix C complete: 0-event transition rows masked to NaN.")

if __name__ == "__main__":
    _entry()
