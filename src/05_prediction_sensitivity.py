from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedKFold
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import OUT_DIR, ensure_dirs
ensure_dirs()
AXES = ["hgi", "rcii", "epwv", "whtr", "hgs_bmi"]
SEED = 42

def get_covs(cohort, tier, endpoint):
    if cohort == "CHARLS":
        m2 = ["age", "female", "edu_cat", "rural_status", "marry_status", "smoke_cur", "drink_cur"]
        clin = ["htn_status", "dyslip_status", "med_htn", "med_lipid", "egfr"]
    else:
        m2 = ["age", "female", "edu_cat", "marry_status", "smoke_cur", "drink_cur"]
        clin = ["htn_status", "dyslip_status", "med_htn", "med_lipid"]
    if tier == "Model2":
        return m2
    if endpoint == "cmm":
        return [*m2, *clin, "n_cmd_base"]
    return [*m2, *clin, "base_diabetes_all"]

def encode_covs(df, cov_cols):
    out = df.copy()
    use = []
    for c in cov_cols:
        if c == "edu_cat":
            for v in [2, 3]:
                cn = f"edu_cat_{v}"
                out[cn] = (out["edu_cat"] == v).astype(float)
                use.append(cn)
        else:
            use.append(c)
    return out, use

def ipcw_weights_at_t(time_arr, status_arr, t_eval):
    """Compute IPCW weights for binary status at fixed horizon t_eval."""
    km = KaplanMeierFitter()
    km.fit(time_arr, event_observed=(status_arr == 0).astype(int))
    ev_by_t = (status_arr == 1) & (time_arr <= t_eval)
    surv_beyond_t = time_arr > t_eval
    comp_by_t = (status_arr == 2) & (time_arr <= t_eval)

    g_t = max(float(km.predict(t_eval)), 0.05)
    w = np.zeros(len(time_arr))
    y = np.zeros(len(time_arr))
    valid = ev_by_t | surv_beyond_t | comp_by_t
    # For observed events <= t_eval, weight = 1 / G(T_i-)
    for i in np.where(ev_by_t | comp_by_t)[0]:
        gi = max(float(km.predict(max(time_arr[i] - 1e-6, 0.0))), 0.05)
        w[i] = 1.0 / gi
        y[i] = 1.0 if ev_by_t[i] else 0.0
    # For event-free past t_eval, weight = 1 / G(t_eval)
    w[surv_beyond_t] = 1.0 / g_t
    y[surv_beyond_t] = 0.0
    return valid, y, w

def calc_metrics_at_t(time_arr, status_arr, pred_risk, pred_lp, t_eval):
    c_idx = float(concordance_index(time_arr, -pred_lp, (status_arr == 1).astype(int)))
    valid, y_bin, w = ipcw_weights_at_t(time_arr, status_arr, t_eval)
    yv, wv, pv = y_bin[valid], w[valid], np.clip(pred_risk[valid], 1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(yv, pv, sample_weight=wv))
    brier = float(np.average((yv - pv) ** 2, weights=wv))
    return c_idx, auc, brier, valid, yv, wv, pv

def nri_idi_ipcw(yv, wv, p_base, p_new):
    ev = yv == 1
    nev = yv == 0
    up_ev = np.average((p_new[ev] > p_base[ev]).astype(float), weights=wv[ev])
    dn_ev = np.average((p_new[ev] < p_base[ev]).astype(float), weights=wv[ev])
    up_nev = np.average((p_new[nev] > p_base[nev]).astype(float), weights=wv[nev])
    dn_nev = np.average((p_new[nev] < p_base[nev]).astype(float), weights=wv[nev])
    nri_ev = float(up_ev - dn_ev)
    nri_nev = float(dn_nev - up_nev)
    nri = nri_ev + nri_nev
    idi = float((np.average(p_new[ev], weights=wv[ev]) - np.average(p_base[ev], weights=wv[ev])) -
                (np.average(p_new[nev], weights=wv[nev]) - np.average(p_base[nev], weights=wv[nev])))
    return nri, nri_ev, nri_nev, idi

def dca_curve(yv, wv, probs, thresholds):
    n_tot = np.sum(wv)
    prev = np.sum(wv * yv) / n_tot
    rows = []
    for pt in thresholds:
        pred_pos = probs >= pt
        tp = np.sum(wv * (pred_pos & (yv == 1))) / n_tot
        fp = np.sum(wv * (pred_pos & (yv == 0))) / n_tot
        nb_model = tp - fp * (pt / (1.0 - pt))
        nb_all = prev - (1.0 - prev) * (pt / (1.0 - pt))
        rows.append((pt, nb_model, nb_all, 0.0))
    return rows

def fit_fold_kmeans_and_assign(train_df, val_df, k, time_col, status_col):
    """Zero-leakage: compute sex-z and KMeans purely on train_df, sort by train risk, assign val_df."""
    tr_z = pd.DataFrame(index=train_df.index)
    va_z = pd.DataFrame(index=val_df.index)
    for ax in AXES:
        for fv in [0.0, 1.0]:
            mt = train_df["female"] == fv
            mv = val_df["female"] == fv
            mu = float(train_df.loc[mt, ax].mean())
            sd = float(train_df.loc[mt, ax].std(ddof=0))
            tr_z.loc[mt, ax] = ((train_df.loc[mt, ax] - mu) / sd).clip(-4.0, 4.0)
            va_z.loc[mv, ax] = ((val_df.loc[mv, ax] - mu) / sd).clip(-4.0, 4.0)
    km = KMeans(n_clusters=k, n_init=30, max_iter=500, random_state=SEED)
    tr_raw = km.fit_predict(tr_z[AXES].to_numpy())
    # Order clusters by train-fold event rate ascending
    ev_tr = (train_df[status_col] == 1).astype(int)
    rank = ev_tr.groupby(tr_raw).mean().sort_values()
    omap = {oc: ni + 1 for ni, oc in enumerate(rank.index)}
    tr_lab = pd.Series(tr_raw, index=train_df.index).map(omap)
    va_raw = km.predict(va_z[AXES].to_numpy())
    va_lab = pd.Series(va_raw, index=val_df.index).map(omap)
    return tr_lab, va_lab, tr_z, va_z

def run_cv_suite(df, cohort, endpoint, tier, time_col, status_col, t_eval, k, cluster_col=None):
    cov_cols = get_covs(cohort, tier, endpoint)
    req = list(dict.fromkeys([*cov_cols, *AXES, "tyg", "egdr", time_col, status_col, *([cluster_col] if cluster_col else [])]))
    sub = df[req].dropna().copy()
    sub = sub[sub[time_col] > 0].copy()
    sub["ev"] = (sub[status_col] == 1).astype(int)
    sub_enc, cov_use = encode_covs(sub, cov_cols)

    # 5 folds: community-grouped in CHARLS, stratified in ELSA
    if cluster_col:
        splitter = GroupKFold(n_splits=5)
        folds = list(splitter.split(sub_enc, sub_enc["ev"], groups=sub_enc[cluster_col]))
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        folds = list(splitter.split(sub_enc, sub_enc["ev"]))

    model_names = ["M0_Base", "M1_Base_TyG", "M2_Base_eGDR", "M3_Base_ePWV", "M4_Base_Endotypes", "M5_Base_All5Axes"]
    oof_lp = {m: np.zeros(len(sub_enc)) for m in model_names}
    oof_risk = {m: np.zeros(len(sub_enc)) for m in model_names}

    for tr_idx, va_idx in folds:
        tr = sub_enc.iloc[tr_idx].copy()
        va = sub_enc.iloc[va_idx].copy()
        tr_k, va_k, tr_z, va_z = fit_fold_kmeans_and_assign(tr, va, k, time_col, status_col)
        for ax in AXES:
            tr[f"{ax}_z"] = tr_z[ax]
            va[f"{ax}_z"] = va_z[ax]
        # Sex-standardize TyG & eGDR inside fold
        for bm in ["tyg", "egdr"]:
            tr[f"{bm}_z"] = np.nan
            va[f"{bm}_z"] = np.nan
            for fv in [0.0, 1.0]:
                mt, mv = tr["female"] == fv, va["female"] == fv
                mu, sd = float(tr.loc[mt, bm].mean()), float(tr.loc[mt, bm].std(ddof=0))
                tr.loc[mt, f"{bm}_z"] = (tr.loc[mt, bm] - mu) / sd
                va.loc[mv, f"{bm}_z"] = (va.loc[mv, bm] - mu) / sd
        # Endotype dummies (ref = 1)
        endo_cols = []
        for eid in range(2, k + 1):
            ec = f"endo_{eid}"
            tr[ec] = (tr_k == eid).astype(float)
            va[ec] = (va_k == eid).astype(float)
            endo_cols.append(ec)

        base_c = [c for c in cov_use if tr[c].nunique() > 1]
        specs = {
            "M0_Base": base_c,
            "M1_Base_TyG": [*base_c, "tyg_z"],
            "M2_Base_eGDR": [*base_c, "egdr_z"],
            "M3_Base_ePWV": [*base_c, "epwv_z"],
            "M4_Base_Endotypes": [*base_c, *endo_cols],
            "M5_Base_All5Axes": [*base_c, *[f"{a}_z" for a in AXES]],
        }
        for mname, xcols in specs.items():
            cph = CoxPHFitter(penalizer=1e-4)
            cph.fit(tr[[*xcols, time_col, "ev"]], duration_col=time_col, event_col="ev")
            lp = cph.predict_log_partial_hazard(va[xcols]).to_numpy().ravel()
            sf = cph.predict_survival_function(va[xcols], times=[t_eval])
            risk = 1.0 - sf.iloc[0].to_numpy()
            oof_lp[mname][va_idx] = lp
            oof_risk[mname][va_idx] = risk

    time_a = sub_enc[time_col].to_numpy()
    stat_a = sub_enc[status_col].to_numpy()
    pred_rows = []
    roc_rows = []
    cal_rows = []
    dca_rows = []

    metrics_cache = {}
    for mname in model_names:
        c_idx, auc, brier, valid, yv, wv, pv = calc_metrics_at_t(time_a, stat_a, oof_risk[mname], oof_lp[mname], t_eval)
        metrics_cache[mname] = (c_idx, auc, brier, valid, yv, wv, pv)

    # Bootstrap CI for delta C-index, delta AUC, NRI, IDI (200x)
    rng = np.random.default_rng(SEED)
    n_obs = len(sub_enc)
    if cluster_col:
        groups = sub_enc[cluster_col].to_numpy()
        uniq_g = np.unique(groups)
        g_idx_map = {g: np.where(groups == g)[0] for g in uniq_g}

    _, _, _, valid0, yv0, wv0, pv0 = metrics_cache["M0_Base"]
    for mname in model_names:
        c_idx, auc, brier, valid, yv, wv, pv = metrics_cache[mname]
        if mname == "M0_Base":
            dc, dc_l, dc_h = 0.0, 0.0, 0.0
            dauc, dauc_l, dauc_h = 0.0, 0.0, 0.0
            nri, nri_ev, nri_nev, idi = 0.0, 0.0, 0.0, 0.0
            nri_l, nri_h, idi_l, idi_h = 0.0, 0.0, 0.0, 0.0
        else:
            nri, nri_ev, nri_nev, idi = nri_idi_ipcw(yv0, wv0, pv0, pv)
            dc_b, dauc_b, nri_b, idi_b = [], [], [], []
            for b in range(200):
                if cluster_col:
                    sampled_g = rng.choice(uniq_g, size=len(uniq_g), replace=True)
                    bidx = np.concatenate([g_idx_map[g] for g in sampled_g])
                else:
                    bidx = rng.integers(0, n_obs, size=n_obs)
                try:
                    c_b0 = concordance_index(time_a[bidx], -oof_lp["M0_Base"][bidx], (stat_a[bidx] == 1).astype(int))
                    c_bm = concordance_index(time_a[bidx], -oof_lp[mname][bidx], (stat_a[bidx] == 1).astype(int))
                    dc_b.append(c_bm - c_b0)
                    vb = valid[bidx]
                    yb, wb = y_bin_b(stat_a[bidx], time_a[bidx], t_eval) if False else (None, None)
                    # Reuse full valid mask & IPCW weights on bootstrap slice (fast paired resample)
                    sel = valid[bidx]
                    if sel.sum() > 20 and len(np.unique((stat_a[bidx][sel] == 1) & (time_a[bidx][sel] <= t_eval))) == 2:
                        y_sub = ((stat_a[bidx][sel] == 1) & (time_a[bidx][sel] <= t_eval)).astype(float)
                        p0_sub = np.clip(oof_risk["M0_Base"][bidx][sel], 1e-6, 1 - 1e-6)
                        pm_sub = np.clip(oof_risk[mname][bidx][sel], 1e-6, 1 - 1e-6)
                        w_sub = np.ones_like(y_sub)
                        dauc_b.append(roc_auc_score(y_sub, pm_sub) - roc_auc_score(y_sub, p0_sub))
                        nr, _, _, id_v = nri_idi_ipcw(y_sub, w_sub, p0_sub, pm_sub)
                        nri_b.append(nr); idi_b.append(id_v)
                except Exception:
                    continue
            dc = c_idx - metrics_cache["M0_Base"][0]
            dc_l, dc_h = np.percentile(dc_b, [2.5, 97.5]) if len(dc_b) else (np.nan, np.nan)
            dauc = auc - metrics_cache["M0_Base"][1]
            dauc_l, dauc_h = np.percentile(dauc_b, [2.5, 97.5]) if len(dauc_b) else (np.nan, np.nan)
            nri_l, nri_h = np.percentile(nri_b, [2.5, 97.5]) if len(nri_b) else (np.nan, np.nan)
            idi_l, idi_h = np.percentile(idi_b, [2.5, 97.5]) if len(idi_b) else (np.nan, np.nan)

        pred_rows.append({
            "cohort": cohort, "endpoint": endpoint, "tier": tier, "t_eval": t_eval,
            "model": mname, "n": n_obs, "events": int(sub_enc["ev"].sum()),
            "c_index": c_idx, "delta_c_vs_m0": dc, "delta_c_ci_low": dc_l, "delta_c_ci_high": dc_h,
            "ipcw_auc": auc, "delta_auc_vs_m0": dauc, "delta_auc_ci_low": dauc_l, "delta_auc_ci_high": dauc_h,
            "brier": brier, "nri": nri, "nri_ci_low": nri_l, "nri_ci_high": nri_h,
            "nri_event": nri_ev, "nri_nonevent": nri_nev,
            "idi": idi, "idi_ci_low": idi_l, "idi_ci_high": idi_h,
        })

        # ROC coordinates (downsampled to 101 points for clean plotting CSV)
        fpr, tpr, _ = roc_curve(yv, pv, sample_weight=wv)
        idx_ds = np.unique(np.linspace(0, len(fpr) - 1, min(101, len(fpr))).astype(int))
        for fp, tp in zip(fpr[idx_ds], tpr[idx_ds]):
            roc_rows.append({"cohort": cohort, "endpoint": endpoint, "tier": tier, "model": mname, "fpr": float(fp), "tpr": float(tp), "auc": auc})

        # Calibration deciles for M0, M4, M5
        if mname in ("M0_Base", "M4_Base_Endotypes", "M5_Base_All5Axes"):
            qs = pd.qcut(oof_risk[mname], 10, labels=False, duplicates="drop")
            for dec in sorted(np.unique(qs)):
                m_dec = qs == dec
                t_d, s_d = time_a[m_dec], stat_a[m_dec]
                # Observed cumulative risk at t_eval via Aalen-Johansen / KM
                obs_r = float(np.average(y_bin_val := np.zeros(m_dec.sum())))
                v_d, y_d, w_d = ipcw_weights_at_t(t_d, s_d, t_eval)
                obs_r = float(np.average(y_d[v_d], weights=w_d[v_d])) if v_d.any() else np.nan
                cal_rows.append({
                    "cohort": cohort, "endpoint": endpoint, "tier": tier, "model": mname,
                    "decile": int(dec) + 1, "n": int(m_dec.sum()),
                    "mean_pred_risk": float(oof_risk[mname][m_dec].mean()),
                    "obs_ipcw_risk": obs_r,
                })

        # DCA curve for key models
        if mname in ("M0_Base", "M1_Base_TyG", "M2_Base_eGDR", "M4_Base_Endotypes", "M5_Base_All5Axes"):
            for pt, nbm, nba, nbn in dca_curve(yv, wv, pv, np.round(np.arange(0.01, 0.31, 0.01), 2)):
                dca_rows.append({
                    "cohort": cohort, "endpoint": endpoint, "tier": tier, "model": mname,
                    "threshold": pt, "net_benefit": nbm, "treat_all": nba, "treat_none": nbn,
                })

    return pd.DataFrame(pred_rows), pd.DataFrame(roc_rows), pd.DataFrame(cal_rows), pd.DataFrame(dca_rows)

def cox_hr_summary(df, cat_col, ref_val, cov_cols, time_col, status_col, cluster_col=None):
    sub = df.dropna(subset=[cat_col, *cov_cols, time_col, status_col, *([cluster_col] if cluster_col else [])]).copy()
    sub = sub[sub[time_col] > 0].copy()
    sub["ev"] = (sub[status_col] == 1).astype(int)
    sub_enc, cov_use = encode_covs(sub, cov_cols)
    levels = [l for l in sorted(sub_enc[cat_col].unique()) if l != ref_val]
    exp_terms = []
    for lv in levels:
        tn = f"exp_{lv}"
        sub_enc[tn] = (sub_enc[cat_col] == lv).astype(float)
        exp_terms.append((lv, tn))
    ucols = [t for _, t in exp_terms] + [c for c in cov_use if sub_enc[c].nunique() > 1]
    cph = CoxPHFitter(penalizer=1e-4)
    fdf = sub_enc[[*ucols, time_col, "ev", *([cluster_col] if cluster_col else [])]]
    if cluster_col:
        cph.fit(fdf, duration_col=time_col, event_col="ev", cluster_col=cluster_col)
    else:
        cph.fit(fdf, duration_col=time_col, event_col="ev", robust=True)
    out = {}
    for lv, tn in exp_terms:
        b = float(cph.params_[tn])
        se = float(cph.standard_errors_[tn])
        ci_l = float(np.exp(cph.confidence_intervals_.loc[tn, "95% lower-bound"]))
        ci_h = float(np.exp(cph.confidence_intervals_.loc[tn, "95% upper-bound"]))
        out[lv] = (len(sub_enc), int(sub_enc["ev"].sum()), np.exp(b), ci_l, ci_h, float(cph.summary.loc[tn, "p"]), b, se)
    return out

def evalue(hr, lo, hi):
    def calc(r):
        if r < 1.0:
            r = 1.0 / r
        return r + np.sqrt(max(r * (r - 1.0), 0.0))
    ev_pt = calc(hr)
    if lo <= 1.0 <= hi:
        ev_ci = 1.0
    elif hr >= 1.0:
        ev_ci = calc(lo)
    else:
        ev_ci = calc(hi)
    return float(ev_pt), float(ev_ci)

def run_sensitivities(ch, el, k):
    rows = []
    ch_el = ch[ch["eligible_w3"] == 1].copy()
    el_el = el[el["eligible_w4"] == 1].copy()

    def add_sens(scenario, cohort, ep, res_dict):
        for lv, tup in res_dict.items():
            n, ev, hr, cl, ch_b, pv, _, _ = tup
            evp, evc = evalue(hr, cl, ch_b)
            rows.append({
                "scenario": scenario, "cohort": cohort, "endpoint": ep,
                "endotype": lv, "n": n, "events": ev,
                "hr": hr, "ci_low": cl, "ci_high": ch_b, "p_value": pv,
                "evalue_point": evp, "evalue_ci": evc,
            })

    # S0: Primary Model 3 & E-values (S7)
    for ep, tc, sc in [("cmm", "time_cmm", "status_cmm"), ("cvd", "time_cvd", "status_cvd")]:
        add_sens("S0_Primary_Model3", "CHARLS", ep, cox_hr_summary(ch_el, "endotype", 1, get_covs("CHARLS", "Model3", ep), tc, sc, "communityID"))
        add_sens("S0_Primary_Model3", "ELSA", ep, cox_hr_summary(el_el, "endotype", 1, get_covs("ELSA", "Model3", ep), tc, sc, None))

        # S1: 2-year lag (exclude events <= 2.0y)
        ch_lag = ch_el[~((ch_el[sc] == 1) & (ch_el[tc] <= 2.0))].copy()
        el_lag = el_el[~((el_el[sc] == 1) & (el_el[tc] <= 2.0))].copy()
        add_sens("S1_Lag_2yr", "CHARLS", ep, cox_hr_summary(ch_lag, "endotype", 1, get_covs("CHARLS", "Model3", ep), tc, sc, "communityID"))
        add_sens("S1_Lag_2yr", "ELSA", ep, cox_hr_summary(el_lag, "endotype", 1, get_covs("ELSA", "Model3", ep), tc, sc, None))

        # S2: Exclude acute inflammation CRP > 10 mg/L
        add_sens("S2_Excl_CRP_gt10", "CHARLS", ep, cox_hr_summary(ch_el[ch_el["crp"] <= 10.0], "endotype", 1, get_covs("CHARLS", "Model3", ep), tc, sc, "communityID"))
        add_sens("S2_Excl_CRP_gt10", "ELSA", ep, cox_hr_summary(el_el[el_el["crp"] <= 10.0], "endotype", 1, get_covs("ELSA", "Model3", ep), tc, sc, None))

        # S3: CHARLS Include non-fasting
        ch_nf = ch[ch["endotype_nofast"].notna()].copy()
        add_sens("S3_Incl_NonFasting", "CHARLS", ep, cox_hr_summary(ch_nf, "endotype_nofast", 1, get_covs("CHARLS", "Model3", ep), tc, sc, "communityID"))

        # S4: MICE (m=10) for missing covariates (including ELSA s4 where drink_cur was missing!)
        for cname, raw_df, e_col, cl_c in [
            ("CHARLS", ch[ch["endotype"].notna()].copy(), "endotype", "communityID"),
            ("ELSA", el[el["endotype_nodrink"].notna()].copy(), "endotype_nodrink", None),
        ]:
            covs = get_covs(cname, "Model3", ep)
            base_sub = raw_df.dropna(subset=[e_col, tc, sc]).copy()
            base_sub = base_sub[base_sub[tc] > 0].copy()
            imp_cols = list(dict.fromkeys([*covs, *AXES, tc, sc]))
            b_list = {lv: [] for lv in range(2, k + 1)}
            v_list = {lv: [] for lv in range(2, k + 1)}
            n_ev = int((base_sub[sc] == 1).sum())
            for m in range(10):
                imp = IterativeImputer(max_iter=10, sample_posterior=True, random_state=SEED + m)
                mat = imp.fit_transform(base_sub[imp_cols])
                imp_df = base_sub.copy()
                for idx_c, col_nm in enumerate(imp_cols):
                    if col_nm in covs:
                        vals = mat[:, idx_c]
                        if col_nm == "edu_cat":
                            vals = np.clip(np.round(vals), 1, 3)
                        elif col_nm in ("rural_status", "marry_status", "smoke_cur", "drink_cur", "htn_status", "dyslip_status", "med_htn", "med_lipid", "base_diabetes_all"):
                            vals = np.clip(np.round(vals), 0, 1)
                        elif col_nm == "n_cmd_base":
                            vals = np.clip(np.round(vals), 0, 1)
                        imp_df[col_nm] = vals
                res_m = cox_hr_summary(imp_df, e_col, 1, covs, tc, sc, cl_c)
                for lv in range(2, k + 1):
                    b_list[lv].append(res_m[lv][6])
                    v_list[lv].append(res_m[lv][7] ** 2)
            for lv in range(2, k + 1):
                b_bar = float(np.mean(b_list[lv]))
                W = float(np.mean(v_list[lv]))
                B = float(np.var(b_list[lv], ddof=1))
                T_var = W + (1.0 + 1.0 / 10.0) * B
                se_pool = float(np.sqrt(T_var))
                hr_p = float(np.exp(b_bar))
                cl_p = float(np.exp(b_bar - 1.96 * se_pool))
                ch_p = float(np.exp(b_bar + 1.96 * se_pool))
                from scipy.stats import norm
                pv_p = float(2.0 * (1.0 - norm.cdf(abs(b_bar) / se_pool)))
                evp, evc = evalue(hr_p, cl_p, ch_p)
                rows.append({
                    "scenario": "S4_MICE_m10", "cohort": cname, "endpoint": ep,
                    "endotype": lv, "n": len(base_sub), "events": n_ev,
                    "hr": hr_p, "ci_low": cl_p, "ci_high": ch_p, "p_value": pv_p,
                    "evalue_point": evp, "evalue_ci": evc,
                })

        # S5: Alternative K (K-1, K+1) in CHARLS
        add_sens(f"S5_AltK_{max(2,k-1)}", "CHARLS", ep, cox_hr_summary(ch_el, "endotype_km1", 1, get_covs("CHARLS", "Model3", ep), tc, sc, "communityID"))
        add_sens(f"S5_AltK_{min(6,k+1)}", "CHARLS", ep, cox_hr_summary(ch_el, "endotype_kp1", 1, get_covs("CHARLS", "Model3", ep), tc, sc, "communityID"))

        # S6: ELSA variations
        add_sens("S6a_ELSA_CHARLS_ZParams", "ELSA", ep, cox_hr_summary(el_el, "endotype_charls_params", 1, get_covs("ELSA", "Model3", ep), tc, sc, None))
        add_sens("S6b_ELSA_DeNovo_KMeans", "ELSA", ep, cox_hr_summary(el_el, "endotype_denovo", 1, get_covs("ELSA", "Model3", ep), tc, sc, None))
        cov_nd = [c for c in get_covs("ELSA", "Model3", ep) if c != "drink_cur"]
        add_sens("S6c_ELSA_NoDrinkDrop", "ELSA", ep, cox_hr_summary(el[el["endotype_nodrink"].notna()], "endotype_nodrink", 1, cov_nd, tc, sc, None))
        add_sens("S6d_ELSA_Excl_W10Disputed", "ELSA", ep, cox_hr_summary(el_el[el_el["w10_disputed_any"] == 0], "endotype", 1, get_covs("ELSA", "Model3", ep), tc, sc, None))

    sens_df = pd.DataFrame(rows)
    sens_df.to_csv(OUT_DIR / "tables" / "tableS6_sensitivity.csv", index=False, encoding="utf-8-sig")
    sens_df.to_csv(OUT_DIR / "figure_data" / "fig6_sensitivity.csv", index=False, encoding="utf-8-sig")
    return sens_df

def main():
    ch = pd.read_csv(OUT_DIR / "derived" / "charls_clustered.csv", index_col="ID", low_memory=False)
    el = pd.read_csv(OUT_DIR / "derived" / "elsa_clustered.csv", index_col="idauniq", low_memory=False)
    ch_el = ch[ch["eligible_w3"] == 1].copy()
    el_el = el[el["eligible_w4"] == 1].copy()
    k = int(ch_el["endotype"].max())

    pred_all, roc_all, cal_all, dca_all = [], [], [], []
    for cohort, df_c, cl_c, t_ev in [("CHARLS", ch_el, "communityID", 6.0), ("ELSA", el_el, None, 10.0)]:
        for ep, tc, sc in [("cmm", "time_cmm", "status_cmm"), ("cvd", "time_cvd", "status_cvd")]:
            for tier in ["Model2", "Model3"]:
                p, r, ca, dc = run_cv_suite(df_c, cohort, ep, tier, tc, sc, t_ev, k, cl_c)
                pred_all.append(p); roc_all.append(r); cal_all.append(ca); dca_all.append(dc)
                print(f"CV done: {cohort} {ep} {tier}")

    pd.concat(pred_all, ignore_index=True).to_csv(OUT_DIR / "tables" / "tableS5_prediction.csv", index=False, encoding="utf-8-sig")
    pd.concat(roc_all, ignore_index=True).to_csv(OUT_DIR / "figure_data" / "fig6_roc.csv", index=False, encoding="utf-8-sig")
    pd.concat(cal_all, ignore_index=True).to_csv(OUT_DIR / "figure_data" / "fig6_calibration.csv", index=False, encoding="utf-8-sig")
    pd.concat(dca_all, ignore_index=True).to_csv(OUT_DIR / "figure_data" / "fig6_dca.csv", index=False, encoding="utf-8-sig")

    # M6 paired subcohort CV: Base vs Base+Landmark Endotype vs Base+4-Year Transition
    m6_rows = []
    for cohort, df_c, cl_c, t_ev in [("CHARLS", ch_el[ch_el["trans_group"].notna()].copy(), "communityID", 6.0),
                                     ("ELSA", el_el[el_el["trans_group"].notna()].copy(), None, 10.0)]:
        for ep, tc, sc in [("cmm", "time_cmm", "status_cmm"), ("cvd", "time_cvd", "status_cvd")]:
            cov_cols = get_covs(cohort, "Model3", ep)
            req = list(dict.fromkeys([*cov_cols, "endotype", "trans_group", tc, sc, *([cl_c] if cl_c else [])]))
            sub = df_c[req].dropna().copy()
            sub = sub[sub[tc] > 0].copy()
            if len(sub) < 200 or (sub[sc] == 1).sum() < 25:
                continue
            sub["ev"] = (sub[sc] == 1).astype(int)
            sub_enc, cov_use = encode_covs(sub, cov_cols)
            endo_c = []
            for eid in range(2, k + 1):
                cn = f"endo_{eid}"
                sub_enc[cn] = (sub_enc["endotype"] == eid).astype(float)
                endo_c.append(cn)
            tr_c = []
            for tg in ["2_Favorable_Transition", "3_Adverse_Transition", "4_Stable_High"]:
                cn = f"tg_{tg[0]}"
                sub_enc[cn] = (sub_enc["trans_group"] == tg).astype(float)
                tr_c.append(cn)
            if cl_c:
                folds = list(GroupKFold(n_splits=5).split(sub_enc, sub_enc["ev"], groups=sub_enc[cl_c]))
            else:
                folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(sub_enc, sub_enc["ev"]))
            for mname, add_c in [("M6_Base", []), ("M6_Base_LandmarkEndotype", endo_c), ("M6_Base_4YrTransition", tr_c)]:
                lp_oof = np.zeros(len(sub_enc))
                r_oof = np.zeros(len(sub_enc))
                for tr_i, va_i in folds:
                    tr, va = sub_enc.iloc[tr_i], sub_enc.iloc[va_i]
                    xcols = [c for c in [*cov_use, *add_c] if tr[c].nunique() > 1]
                    cph = CoxPHFitter(penalizer=1e-4)
                    cph.fit(tr[[*xcols, tc, "ev"]], duration_col=tc, event_col="ev")
                    lp_oof[va_i] = cph.predict_log_partial_hazard(va[xcols]).to_numpy().ravel()
                    r_oof[va_i] = 1.0 - cph.predict_survival_function(va[xcols], times=[t_ev]).iloc[0].to_numpy()
                c_idx, auc, brier, _, _, _, _ = calc_metrics_at_t(sub_enc[tc].to_numpy(), sub_enc[sc].to_numpy(), r_oof, lp_oof, t_ev)
                m6_rows.append({"cohort": cohort, "endpoint": ep, "model": mname, "n": len(sub_enc), "events": int(sub_enc["ev"].sum()), "c_index": c_idx, "ipcw_auc": auc, "brier": brier})
    if m6_rows:
        pd.DataFrame(m6_rows).to_csv(OUT_DIR / "tables" / "tableS5b_paired_transition_prediction.csv", index=False, encoding="utf-8-sig")

    run_sensitivities(ch, el, k)
    print("Step 5 CV prediction and sensitivity suite complete!")

if __name__ == "__main__":
    main()
