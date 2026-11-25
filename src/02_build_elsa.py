from pathlib import Path
import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import ELSA_DIR, OUT_DIR, ensure_dirs
ensure_dirs()

def find_elsa(fname):
    direct = ELSA_DIR / fname
    return direct if direct.exists() else next(ELSA_DIR.rglob(fname))

def clean_neg(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s >= 0)

def compute_elsa_axes(nurse, hgi_params=None, charls_hgi_params=None):
    age = clean_neg(nurse["confage"]).where(lambda s: s.between(45, 105))
    fglu_mgdl = clean_neg(nurse["fglu"]).where(lambda s: s.between(2.0, 30.0)) * 18.0182
    hba1c = clean_neg(nurse["hba1c"]).where(lambda s: s.between(3.5, 18.5))
    chol_mgdl = clean_neg(nurse["chol"]).where(lambda s: s.between(1.0, 20.0)) * 38.67
    hdl_mgdl = clean_neg(nurse["hdl"]).where(lambda s: s.between(0.2, 6.0)) * 38.67
    ldl_mgdl = clean_neg(nurse["ldl"]).where(lambda s: s.between(0.2, 15.0)) * 38.67
    tg_mgdl = clean_neg(nurse["trig"]).where(lambda s: s.between(0.2, 30.0)) * 88.57
    rc = (chol_mgdl - hdl_mgdl - ldl_mgdl).where(lambda s: s > 0)
    crp = clean_neg(nurse["hscrp"]).where(lambda s: s > 0)
    sbp = clean_neg(nurse["sysval"]).where(lambda s: s.between(60, 260))
    dbp = clean_neg(nurse["diaval"]).where(lambda s: s.between(40, 180))
    valid_bp = sbp.notna() & dbp.notna() & (sbp > dbp)
    sbp, dbp = sbp.where(valid_bp), dbp.where(valid_bp)
    waist = clean_neg(nurse["wstval"]).where(lambda s: s.between(45, 180))
    ht_cm = clean_neg(nurse["htval"]).where(lambda s: s.between(110, 220))
    bmi = clean_neg(nurse["bmival"]).where(lambda s: s.between(12, 65))
    gcols = ["mmgsd1", "mmgsd2", "mmgsd3", "mmgsn1", "mmgsn2", "mmgsn3"]
    # Mask sentinel 99 and missing codes outside [1, 90]
    gmat = nurse[gcols].apply(lambda c: pd.to_numeric(c, errors="coerce").where(lambda s: s.between(1.0, 90.0)))
    hgs = gmat.max(axis=1)

    if hgi_params is None:
        ok_hgi = fglu_mgdl.notna() & hba1c.notna()
        x = fglu_mgdl[ok_hgi].to_numpy()
        y = hba1c[ok_hgi].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
        hgi_params = (float(intercept), float(slope), r2, int(ok_hgi.sum()))

    a, b, _, _ = hgi_params
    hgi = (hba1c - (a + b * fglu_mgdl)).where(fglu_mgdl.notna() & hba1c.notna())
    if charls_hgi_params is not None:
        ac, bc = charls_hgi_params
        hgi_charls_eq = (hba1c - (ac + bc * fglu_mgdl)).where(fglu_mgdl.notna() & hba1c.notna())
    else:
        hgi_charls_eq = pd.Series(np.nan, index=nurse.index)

    rcii = np.log(rc * crp / 10.0).where(rc.notna() & crp.notna())
    mbp = dbp + 0.4 * (sbp - dbp)
    epwv = (
        9.587
        - 0.402 * age
        + 4.560e-3 * (age ** 2)
        - 2.621e-5 * (age ** 2) * mbp
        + 3.176e-3 * age * mbp
        - 1.832e-2 * mbp
    ).where(age.notna() & mbp.notna())
    whtr = (waist / ht_cm).where(waist.notna() & ht_cm.notna())
    hgs_bmi = (hgs / bmi).where(hgs.notna() & bmi.notna())
    tyg = np.log(tg_mgdl * fglu_mgdl / 2.0).where(tg_mgdl.notna() & fglu_mgdl.notna())

    out = pd.DataFrame({
        "age_nurse": age, "hgi": hgi, "hgi_charls_eq": hgi_charls_eq,
        "rcii": rcii, "epwv": epwv, "whtr": whtr, "hgs_bmi": hgs_bmi,
        "tyg": tyg, "rc": rc, "crp": crp, "glu": fglu_mgdl, "hba1c": hba1c,
        "sbp": sbp, "dbp": dbp, "waist": waist, "bmi": bmi, "hgs": hgs,
        "chol_mgdl": chol_mgdl, "hdl_mgdl": hdl_mgdl, "tg_mgdl": tg_mgdl,
    }, index=nurse.index)
    return out, hgi_params

def build():
    # Read CHARLS HGI equation params if already saved
    eq_fp = OUT_DIR / "tables" / "table_qc_equations.csv"
    charls_eq = None
    if eq_fp.exists():
        eq_df = pd.read_csv(eq_fp)
        row = eq_df[eq_df["cohort"] == "CHARLS_W3"].iloc[0]
        charls_eq = (float(row["intercept"]), float(row["slope_per_mgdl"]))

    n2 = pd.read_stata(
        find_elsa("wave_2_nurse_data_v2.dta"),
        columns=["idauniq","confage","sex","fglu","hba1c","chol","hdl","ldl","trig","hscrp","sysval","diaval","wstval","htval","bmival","mmgsd1","mmgsd2","mmgsd3","mmgsn1","mmgsn2","mmgsn3"],
        convert_categoricals=False,
    ).drop_duplicates("idauniq").set_index("idauniq")

    n4 = pd.read_stata(
        find_elsa("wave_4_nurse_data.dta"),
        columns=["idauniq","confage","dhsex","fglu","hba1c","chol","hdl","ldl","trig","hscrp","sysval","diaval","wstval","htval","bmival","mmgsd1","mmgsd2","mmgsd3","mmgsn1","mmgsn2","mmgsn3"],
        convert_categoricals=False,
    ).drop_duplicates("idauniq").set_index("idauniq")

    # Harmonized ELSA W1-W9
    hcols = ["idauniq", "ragender", "raeducl", "r4mstat", "r4smoken", "r4drink", "r4hibpe", "r4cancre", "r4rxhibp", "r4rxdiab", "r4rxdiabi", "r4rxdiabo", "radyear"]
    for w in [4, 5, 6, 7, 8, 9]:
        hcols.extend([f"r{w}iwstat", f"r{w}iwindy", f"r{w}iwindm", f"r{w}hearte", f"r{w}stroke", f"r{w}diabe"])
    hcols.append("r4agey")
    eg = pd.read_stata(find_elsa("h_elsa_g3.dta"), columns=hcols, convert_categoricals=False).drop_duplicates("idauniq").set_index("idauniq")

    # W4 core lipid medication
    w4c = pd.read_stata(find_elsa("wave_4_elsa_data_v3.dta"), columns=["idauniq", "hechmd", "hechme"], convert_categoricals=False).drop_duplicates("idauniq").set_index("idauniq")

    # W10 core outcomes
    w10_cols = [
        "idauniq", "iintdaty", "iintdatm",
        "heeveran", "henvran", "heevermi", "henvrmi", "heeverhf", "henvrhf",
        "heeverhm", "henvrhm", "heeverah", "henvrah", "heeverst", "henvrst",
        "heacd", "henvrdi", "heothcv2", "heothcv3", "heothcv4", "heothcv5", "heothcv6", "heothcv7",
    ]
    w10 = pd.read_stata(find_elsa("wave_10_elsa_data_eul_v4.dta"), columns=w10_cols, convert_categoricals=False).drop_duplicates("idauniq").set_index("idauniq")

    # Compute axes at W4 (landmark) and W2 (prior wave)
    axes_w4, hgi_params_w4 = compute_elsa_axes(n4, hgi_params=None, charls_hgi_params=charls_eq)
    axes_w2, _ = compute_elsa_axes(n2, hgi_params=hgi_params_w4, charls_hgi_params=charls_eq)

    df = axes_w4.copy()
    for c in ["hgi", "rcii", "epwv", "whtr", "hgs_bmi"]:
        df[f"{c}_w2"] = axes_w2[c].reindex(df.index)

    # Demographics & covariates from h_elsa_g3 (fallback age/sex to nurse)
    df["age"] = clean_neg(eg["r4agey"]).reindex(df.index).fillna(df["age_nurse"])
    rag = clean_neg(eg["ragender"]).reindex(df.index)
    dh = clean_neg(n4["dhsex"])
    sex_raw = rag.fillna(dh)
    df["female"] = np.where(sex_raw.eq(2), 1.0, np.where(sex_raw.eq(1), 0.0, np.nan))
    df["edu_cat"] = clean_neg(eg["raeducl"]).reindex(df.index).where(lambda s: s.isin([1, 2, 3]))
    mstat = clean_neg(eg["r4mstat"]).reindex(df.index)
    df["marry_status"] = np.where(mstat.isin([1, 2, 3]), 1.0, np.where(mstat.isin([4, 5, 7, 8]), 0.0, np.nan))
    df["smoke_cur"] = clean_neg(eg["r4smoken"]).reindex(df.index).where(lambda s: s.isin([0, 1]))
    df["drink_cur"] = clean_neg(eg["r4drink"]).reindex(df.index).where(lambda s: s.isin([0, 1]))
    # Note: r4drink has -3 (no self-completion) for ~15%, keep missing flag & allow sensitivity
    df["med_htn"] = clean_neg(eg["r4rxhibp"]).reindex(df.index).eq(1).astype(int)
    df["med_diab"] = (
        clean_neg(eg["r4rxdiab"]).reindex(df.index).eq(1)
        | clean_neg(eg["r4rxdiabi"]).reindex(df.index).eq(1)
        | clean_neg(eg["r4rxdiabo"]).reindex(df.index).eq(1)
    ).astype(int)
    df["med_lipid"] = (w4c["hechmd"].reindex(df.index).eq(1) | w4c["hechme"].reindex(df.index).eq(1)).astype(int)
    hibpe_raw = clean_neg(eg["r4hibpe"]).reindex(df.index)
    df["htn_status"] = np.where(
        hibpe_raw.eq(1) | df["med_htn"].eq(1) | (df["sbp"] >= 140) | (df["dbp"] >= 90),
        1,
        np.where(hibpe_raw.eq(0) & df["sbp"].lt(140) & df["dbp"].lt(90), 0, np.nan),
    )
    df["egdr"] = (
        21.158 - 0.09 * df["waist"] - 3.407 * df["htn_status"] - 0.551 * df["hba1c"]
    ).where(df["waist"].notna() & df["htn_status"].notna() & df["hba1c"].notna())
    df["dyslip_status"] = np.where(
        df["med_lipid"].eq(1) | (df["chol_mgdl"] >= 240) | (df["tg_mgdl"] >= 200) | (df["hdl_mgdl"] < 40),
        1.0,
        0.0,
    )

    df["base_cancer"] = clean_neg(eg["r4cancre"]).reindex(df.index).eq(1).astype(int)
    df["base_heart"] = clean_neg(eg["r4hearte"]).reindex(df.index).eq(1).astype(int)
    df["base_stroke"] = clean_neg(eg["r4stroke"]).reindex(df.index).eq(1).astype(int)
    df["base_diabetes_all"] = (clean_neg(eg["r4diabe"]).reindex(df.index).eq(1) | (df["med_diab"] == 1)).astype(int)
    df["base_cvd"] = ((df["base_heart"] == 1) | (df["base_stroke"] == 1)).astype(int)
    df["n_cmd_base"] = df["base_diabetes_all"] + df["base_heart"] + df["base_stroke"]
    df["base_cmm"] = (df["n_cmd_base"] >= 2).astype(int)
    df["base_any_cmd"] = (df["n_cmd_base"] >= 1).astype(int)

    # CKM proxy
    high_met = (df["htn_status"] == 1) | (df["base_diabetes_all"] == 1) | (df["dyslip_status"] == 1)
    very_high = (df["age"] >= 65) & (df["htn_status"] == 1) & (df["base_diabetes_all"] == 1)
    df["ckm_proxy"] = np.where(very_high, 3, np.where(high_met, 2, 1))

    # Dates for W4..W9 and W10
    for w in [4, 5, 6, 7, 8, 9]:
        yr = clean_neg(eg[f"r{w}iwindy"]).reindex(df.index)
        mo = clean_neg(eg[f"r{w}iwindm"]).reindex(df.index)
        df[f"date_w{w}"] = (yr + (mo.where(mo.between(1, 12), 7.5) - 0.5) / 12.0).where(yr.between(2007, 2024))
        df[f"iwstat_w{w}"] = clean_neg(eg[f"r{w}iwstat"]).reindex(df.index)
        for d_raw, d_nm in [("hearte", "heart"), ("stroke", "stroke"), ("diabe", "diabetes")]:
            df[f"{d_nm}_w{w}"] = clean_neg(eg[f"r{w}{d_raw}"]).reindex(df.index).where(lambda s: s.isin([0, 1]))

    # W10 dates & outcomes
    yr10 = clean_neg(w10["iintdaty"]).reindex(df.index)
    mo10 = clean_neg(w10["iintdatm"]).reindex(df.index)
    df["date_w10"] = (yr10 + (mo10.where(mo10.between(1, 12), 7.5) - 0.5) / 12.0).where(yr10.between(2020, 2024))
    in_w10 = df.index.isin(w10.index)

    hrt10_yes = (
        w10[["heeveran", "heevermi", "heeverhf", "heeverhm", "heeverah", "heothcv2", "heothcv3", "heothcv4", "heothcv5", "heothcv6"]]
        .eq(1).any(axis=1)
    ).reindex(df.index, fill_value=False)
    hrt10_no = w10["heeveran"].eq(2).reindex(df.index, fill_value=False)
    stk10_yes = (w10["heeverst"].eq(1) | w10["heothcv7"].eq(1)).reindex(df.index, fill_value=False)
    stk10_no = w10["heeverst"].eq(2).reindex(df.index, fill_value=False)
    dia10_yes = w10["heacd"].eq(1).reindex(df.index, fill_value=False)
    dia10_no = w10["heacd"].eq(2).reindex(df.index, fill_value=False)

    # Also track disputed diagnoses at W10 (henvr* == 1 means disputes preloaded diagnosis -> set negative in strict sens)
    disp_hrt = w10[["henvran", "henvrmi", "henvrhf", "henvrhm", "henvrah"]].eq(1).any(axis=1).reindex(df.index, fill_value=False)
    disp_stk = w10["henvrst"].eq(1).reindex(df.index, fill_value=False)
    disp_dia = w10["henvrdi"].eq(1).reindex(df.index, fill_value=False)
    df["w10_disputed_any"] = (disp_hrt | disp_stk | disp_dia).astype(int)

    # Carry forward W9 positive status for W10 if W10 did not re-ask preloaded affirmative directly
    df["heart_w10"] = np.where(in_w10, np.where(hrt10_yes | (df["heart_w9"].eq(1) & ~disp_hrt), 1.0, np.where(hrt10_no | disp_hrt | df["heart_w9"].eq(0), 0.0, np.nan)), np.nan)
    df["stroke_w10"] = np.where(in_w10, np.where(stk10_yes | (df["stroke_w9"].eq(1) & ~disp_stk), 1.0, np.where(stk10_no | disp_stk | df["stroke_w9"].eq(0), 0.0, np.nan)), np.nan)
    df["diabetes_w10"] = np.where(in_w10, np.where(dia10_yes | (df["diabetes_w9"].eq(1) & ~disp_dia), 1.0, np.where(dia10_no | disp_dia | df["diabetes_w9"].eq(0), 0.0, np.nan)), np.nan)

    # Mortality in ELSA (available in h_elsa_g3 through W6: radyear or iwstat==5)
    rad_yr = clean_neg(eg["radyear"]).reindex(df.index)
    died_w5 = df["iwstat_w5"].eq(5)
    died_w6 = df["iwstat_w6"].eq(5)
    df["died_any"] = (died_w5 | died_w6 | rad_yr.between(2008, 2024)).astype(int)
    d_w4 = df["date_w4"].fillna(2008.75)
    wave_ddt = np.where(died_w5, 0.5 * (d_w4 + 2010.75), np.where(died_w6, 0.5 * (df["date_w5"].fillna(2010.75) + 2012.75), np.nan))
    df["death_date"] = np.where(rad_yr.between(2008, 2024), rad_yr + 0.5, wave_ddt)
    df.loc[df["died_any"].eq(1) & (df["death_date"] <= d_w4), "death_date"] = d_w4 + 0.5

    df["date_w4_clean"] = d_w4
    last_int = d_w4.copy()
    fu_any = pd.Series(False, index=df.index)
    for w, fb in [(5, 2010.75), (6, 2012.75), (7, 2014.75), (8, 2016.75), (9, 2018.75), (10, 2022.25)]:
        obs = df[f"heart_w{w}"].notna() | df[f"stroke_w{w}"].notna() | df[f"diabetes_w{w}"].notna()
        fu_any = fu_any | obs
        dt = df[f"date_w{w}"].fillna(fb)
        last_int = np.where(obs, dt, last_int)
    df["has_followup"] = (fu_any | df["died_any"].eq(1)).astype(int)
    df["last_alive_date"] = last_int

    def compute_endpoint(df_in, base_col, wave_pos_fn):
        t0 = df_in["date_w4_clean"].to_numpy()
        fbs = {5: 2010.75, 6: 2012.75, 7: 2014.75, 8: 2016.75, 9: 2018.75, 10: 2022.25}
        dates = {w: df_in[f"date_w{w}"].fillna(fb).to_numpy() for w, fb in fbs.items()}
        pos = wave_pos_fn(df_in)
        obs = {w: df_in[f"heart_w{w}"].notna().to_numpy() for w in fbs}

        first_w = np.zeros(len(df_in), dtype=int)
        prev_neg = np.full(len(df_in), t0)
        ev_date = np.full(len(df_in), np.nan)
        for w in [5, 6, 7, 8, 9, 10]:
            hit = (first_w == 0) & pos[w]
            first_w = np.where(hit, w, first_w)
            ev_date = np.where(hit, 0.5 * (prev_neg + dates[w]), ev_date)
            prev_neg = np.where(obs[w] & ~pos[w], dates[w], prev_neg)

        has_ev = first_w > 0
        died = df_in["died_any"].to_numpy() == 1
        d_dt = df_in["death_date"].to_numpy()
        cens_dt = df_in["last_alive_date"].to_numpy()
        is_death = died & (~has_ev | (d_dt < ev_date))
        status = np.where(has_ev & ~is_death, 1, np.where(is_death, 2, 0))
        end_dt = np.where(status == 1, ev_date, np.where(status == 2, d_dt, cens_dt))
        time_yrs = np.maximum(end_dt - t0, 0.08)
        prev = df_in[base_col].to_numpy() == 1
        return (
            np.where(prev, np.nan, status),
            np.where(prev, np.nan, time_yrs),
            np.where(prev, np.nan, first_w),
        )

    for dis, bcol in [("diabetes", "base_diabetes_all"), ("heart", "base_heart"), ("stroke", "base_stroke")]:
        st, tm, fw = compute_endpoint(df, bcol, lambda d, dis=dis: {w: d[f"{dis}_w{w}"].eq(1).to_numpy() for w in [5,6,7,8,9,10]})
        df[f"status_{dis}"] = st
        df[f"time_{dis}"] = tm
        df[f"first_wave_{dis}"] = fw

    st, tm, fw = compute_endpoint(df, "base_cvd", lambda d: {w: (d[f"heart_w{w}"].eq(1) | d[f"stroke_w{w}"].eq(1)).to_numpy() for w in [5,6,7,8,9,10]})
    df["status_cvd"] = st
    df["time_cvd"] = tm
    df["first_wave_cvd"] = fw

    # Cumulative CMD across W5..W10
    diab = df["base_diabetes_all"].eq(1)
    hrt = df["base_heart"].eq(1)
    stk = df["base_stroke"].eq(1)
    for w in [5, 6, 7, 8, 9, 10]:
        diab = diab | df[f"diabetes_w{w}"].eq(1)
        hrt = hrt | df[f"heart_w{w}"].eq(1)
        stk = stk | df[f"stroke_w{w}"].eq(1)
        df[f"n_cmd_w{w}"] = diab.astype(int) + hrt.astype(int) + stk.astype(int)

    st, tm, fw = compute_endpoint(df, "base_cmm", lambda d: {w: d[f"n_cmd_w{w}"].ge(2).to_numpy() for w in [5,6,7,8,9,10]})
    df["status_cmm"] = st
    df["time_cmm"] = tm
    df["first_wave_cmm"] = fw

    st, tm, fw = compute_endpoint(df, "base_any_cmd", lambda d: {w: d[f"n_cmd_w{w}"].ge(1).to_numpy() for w in [5,6,7,8,9,10]})
    df["status_fcmd"] = st
    df["time_fcmd"] = tm
    df["first_wave_fcmd"] = fw

    df["status_death"] = df["died_any"].astype(int)
    df["time_death"] = np.maximum(np.where(df["died_any"] == 1, df["death_date"], df["last_alive_date"]) - df["date_w4_clean"], 0.08)

    # STROBE Flow
    axes_cols = ["hgi", "rcii", "epwv", "whtr", "hgs_bmi"]
    s0 = df["age"].between(45, 105)
    s1 = s0 & df["glu"].notna() & df["hba1c"].notna() & df["rc"].notna() & df["crp"].notna()
    s2 = s1 & df[axes_cols].notna().all(axis=1)
    s3 = s2 & df["base_cancer"].eq(0)
    s4 = s3 & df["has_followup"].eq(1) & df["time_death"].gt(0)
    cov_m2 = ["age", "female", "edu_cat", "marry_status", "smoke_cur", "drink_cur"]
    s5 = s4 & df[cov_m2].notna().all(axis=1)
    df["eligible_w4"] = s5.astype(int)
    df["eligible_w4_nodrink"] = (s4 & df[["age", "female", "edu_cat", "marry_status", "smoke_cur"]].notna().all(axis=1)).astype(int)
    df["paired_w2_w4"] = (df["eligible_w4"].eq(1) & df[[f"{c}_w2" for c in axes_cols]].notna().all(axis=1)).astype(int)

    flow_rows = [
        ("1. ELSA Wave 4 (2008/09) nurse participants aged >=45 years", int(s0.sum()), 0),
        ("2. Fasting blood complete for HGI & RCII (RC>0, CRP>0)", int(s1.sum()), int(s0.sum() - s1.sum())),
        ("3. Complete 5 physiological axes (add ePWV, WHtR, HGS/BMI)", int(s2.sum()), int(s1.sum() - s2.sum())),
        ("4. Free of baseline cancer", int(s3.sum()), int(s2.sum() - s3.sum())),
        ("5. >=1 follow-up wave (W5-W10) or vital status ascertained", int(s4.sum()), int(s3.sum() - s4.sum())),
        ("6. Complete core Model 2 covariates (External Validation Cohort)", int(s5.sum()), int(s4.sum() - s5.sum())),
        ("   6a. Baseline CMM-free validation cohort (<2 CMD)", int((s5 & df["base_cmm"].eq(0)).sum()), int((s5 & df["base_cmm"].eq(1)).sum())),
        ("   6b. Baseline CVD-free validation cohort", int((s5 & df["base_cvd"].eq(0)).sum()), int((s5 & df["base_cvd"].eq(1)).sum())),
        ("   6c. Baseline CMD-free Multi-State cohort (0 CMD)", int((s5 & df["n_cmd_base"].eq(0)).sum()), int((s5 & df["n_cmd_base"].gt(0)).sum())),
        ("   6d. Paired W2 (2004/05) & W4 (2008/09) Transition sub-cohort", int(df["paired_w2_w4"].sum()), int(s5.sum() - df["paired_w2_w4"].sum())),
    ]
    flow_df = pd.DataFrame(flow_rows, columns=["step", "n_remaining", "n_excluded"])
    flow_df.to_csv(OUT_DIR / "tables" / "table_flow_elsa.csv", index=False, encoding="utf-8-sig")
    flow_df.to_csv(OUT_DIR / "figure_data" / "fig1_flow_elsa.csv", index=False, encoding="utf-8-sig")

    if eq_fp.exists():
        eq_df = pd.read_csv(eq_fp)
        eq_df = eq_df[eq_df["cohort"] != "ELSA_W4"]
        eq_df = pd.concat([eq_df, pd.DataFrame([{
            "cohort": "ELSA_W4", "intercept": hgi_params_w4[0], "slope_per_mgdl": hgi_params_w4[1],
            "r2": hgi_params_w4[2], "n": hgi_params_w4[3],
        }])], ignore_index=True)
        eq_df.to_csv(eq_fp, index=False, encoding="utf-8-sig")

    out_df = df.loc[s4].copy()
    out_df.to_csv(OUT_DIR / "derived" / "elsa_analytic.csv", index=True, encoding="utf-8-sig")
    print("ELSA build complete!")
    print(flow_df.to_string(index=False))
    print("W4 HGI equation:", hgi_params_w4)

if __name__ == "__main__":
    build()
