from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CHARLS_DIR, OUT_DIR, ensure_dirs
ensure_dirs()

DISEASES = {"diabetes": 3, "cancer": 4, "heart": 7, "stroke": 8, "kidney": 9}

def norm_id(series):
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(12)
    )

def read_stata_idx(source, columns):
    df = pd.read_stata(source, columns=columns, convert_categoricals=False)
    df["ID"] = norm_id(df["ID"])
    df = df.drop_duplicates("ID").set_index("ID")
    return df

def read_zip_stata(archive, member, columns):
    return read_stata_idx(BytesIO(archive.read(member)), columns)

def disease_status(data, direct, comparison, dispute_is_negative=False):
    return pd.Series(
        np.select(
            [
                data[direct].eq(1),
                data[direct].eq(2),
                data[comparison].isin([1, 2, 3]),
                data[comparison].eq(99) & dispute_is_negative,
            ],
            [1.0, 0.0, 1.0, 0.0],
            default=np.nan,
        ),
        index=data.index,
    )

def raw_w3_baseline(archive_path):
    direct = ["da007", "da007_w2_1", "da007_w2_2", "zda007"]
    dcols = [f"{p}_{k}_" for k in DISEASES.values() for p in direct]
    meds = ["da014s1", "da014s2", "da014s3"]
    with ZipFile(archive_path) as z:
        raw = read_zip_stata(z, "Health_Status_and_Functioning.dta", ["ID", *dcols, *meds])
    out = pd.DataFrame(index=raw.index)
    for d, k in DISEASES.items():
        new = raw[f"da007_{k}_"]
        since = raw[f"da007_w2_2_{k}_"]
        pre = raw[f"zda007_{k}_"].eq(1)
        ver = raw[f"da007_w2_1_{k}_"]
        prior_ok = (pre & ver.eq(1)) | (~pre & ver.eq(2))
        out[d] = np.select(
            [new.eq(1) | since.eq(1) | prior_ok, new.eq(2) | since.eq(2)],
            [1.0, 0.0],
            default=np.nan,
        )
    out["diabetes_med_raw"] = raw[meds].isin([1, 2, 3]).any(axis=1).astype(float)
    return out

def followup_wave(archive_path, direct, comparison, date_prefix=None):
    extra = [f"{date_prefix}_{k}_" for k in DISEASES.values()] if date_prefix else []
    cols = [f"{p}_{k}_" for k in DISEASES.values() for p in (direct, comparison)]
    with ZipFile(archive_path) as z:
        raw = read_zip_stata(z, "Health_Status_and_Functioning.dta", ["ID", *cols, *extra])
    out = pd.DataFrame(index=raw.index)
    for d, k in DISEASES.items():
        out[d] = disease_status(
            raw, f"{direct}_{k}_", f"{comparison}_{k}_", dispute_is_negative=(direct == "da007")
        )
        if date_prefix:
            yr = pd.to_numeric(raw[f"{date_prefix}_{k}_"], errors="coerce")
            out[f"{d}_year"] = yr.where(out[d].eq(1) & yr.between(1950, 2024))
    return out

def read_wave6(archive_path):
    direct = ["da011", "da012", "da016_1"]
    dcols = [f"{p}_{k}_" for k in DISEASES.values() for p in direct]
    extra = ["da011_12_", "da012_12_", "da011_13_", "da012_13_", "da028_7_", "da031_8_", "da139"]
    im_codes = [3, 4, 5, 6]
    surg_codes = [8, 9]
    care_cols = ["ec001", "ec004"]
    for p in (1, 2, 3):
        care_cols.extend([f"ec017_{p}__s{c}" for c in im_codes])
        care_cols.extend([f"ec018_{p}__s{c}" for c in surg_codes])
    with ZipFile(archive_path) as z:
        sample = read_zip_stata(
            z, "CHARLS-W6-2023/Sample_Infor.dta", ["ID", "res_wave", "iyear", "imonth", "died"]
        )
        sample["iyear"] = pd.to_numeric(sample["iyear"], errors="coerce")
        sample["imonth"] = pd.to_numeric(sample["imonth"], errors="coerce")
        h_list, c_list = [], []
        for yr in (2021, 2022, 2023):
            hd = pd.read_stata(
                BytesIO(z.read(f"CHARLS-W6-{yr}/Health_Status_and_Functioning.dta")),
                columns=["ID", *dcols, *extra],
                convert_categoricals=False,
            )
            hd["ID"] = norm_id(hd["ID"])
            hd["file_year"] = yr
            h_list.append(hd)
            with pd.io.stata.StataReader(BytesIO(z.read(f"CHARLS-W6-{yr}/Health_Care_Insurance.dta")), convert_categoricals=False) as rdr:
                avail = set(rdr.variable_labels().keys())
                use_c = [c for c in ["ID", *care_cols] if c in avail]
                cd = rdr.read(columns=use_c)
            cd["ID"] = norm_id(cd["ID"])
            cd["file_year"] = yr
            c_list.append(cd)
    health = pd.concat(h_list, ignore_index=True).set_index("ID")
    health = health.join(sample[["res_wave", "iyear", "died"]], how="inner")
    health = health.loc[health["died"].eq(0) & health["file_year"].eq(health["res_wave"])]
    health = health.loc[~health.index.duplicated()]

    care = pd.concat(c_list, ignore_index=True).set_index("ID")
    care = care.join(sample[["res_wave", "died"]], how="inner")
    care = care.loc[care["died"].eq(0) & care["file_year"].eq(care["res_wave"])]
    care = care.loc[~care.index.duplicated()]

    out = sample.copy()
    for d, k in DISEASES.items():
        out[d] = disease_status(health, f"da011_{k}_", f"da012_{k}_").reindex(out.index)
        yr = pd.to_numeric(health[f"da016_1_{k}_"], errors="coerce").reindex(out.index)
        out[f"{d}_year"] = yr.where(out[d].eq(1) & yr.between(1950, 2024))
    out["dementia_new_w6"] = health["da011_12_"].eq(1).astype(float).where(health["da011_12_"].isin([1, 2])).reindex(out.index)
    out["atrophy_new_w6"] = health["da011_13_"].eq(1).astype(float).where(health["da011_13_"].isin([1, 2])).reindex(out.index)
    out["heart_attack_w6"] = health["da028_7_"].eq(1).astype(float).where(health["da028_7_"].isin([1, 2])).reindex(out.index)
    out["stroke_relapse_w6"] = health["da031_8_"].eq(1).astype(float).where(health["da031_8_"].isin([1, 2])).reindex(out.index)

    hosp_valid = care["ec001"].isin([1, 2])
    cm_dept = pd.Series(False, index=care.index)
    card_dept = pd.Series(False, index=care.index)
    for p in (1, 2, 3):
        for c in im_codes:
            col = f"ec017_{p}__s{c}"
            if col in care.columns:
                cm_dept |= care[col].eq(c)
                if c == 3:
                    card_dept |= care[col].eq(c)
        for c in surg_codes:
            col = f"ec018_{p}__s{c}"
            if col in care.columns:
                cm_dept |= care[col].eq(c)
                if c == 8:
                    card_dept |= care[col].eq(c)
    out["w6_cm_hosp"] = pd.Series(
        np.where(hosp_valid, np.where(care["ec001"].eq(1) & cm_dept, 1.0, 0.0), np.nan),
        index=care.index,
    ).reindex(out.index)
    out["w6_card_hosp"] = pd.Series(
        np.where(hosp_valid, np.where(care["ec001"].eq(1) & card_dept, 1.0, 0.0), np.nan),
        index=care.index,
    ).reindex(out.index)
    return out

def ckd_epi_2021(crea, age, female):
    cr = crea.where(crea.between(0.2, 15.0))
    kappa = np.where(female == 1, 0.7, 0.9)
    alpha = np.where(female == 1, -0.241, -0.302)
    ratio = cr / kappa
    egfr = (
        142.0
        * (np.minimum(ratio, 1.0) ** alpha)
        * (np.maximum(ratio, 1.0) ** -1.200)
        * (0.9938 ** age)
        * np.where(female == 1, 1.012, 1.0)
    )
    return pd.Series(egfr, index=crea.index).where(cr.notna() & age.notna())

def compute_axes(df, fasting_only=True, hgi_params=None):
    age = df["age"].where(df["age"].between(45, 110))
    fast = df["bl_fasting"].eq(1) if fasting_only else df["bl_fasting"].isin([0, 1, 2]) | df["bl_fasting"].isna()
    glu = df["bl_glu"].where(fast & df["bl_glu"].between(40, 500))
    hba1c = df["bl_hbalc"].where(df["bl_hbalc"].between(3.5, 18.5))
    cho = df["bl_cho"].where(df["bl_cho"].between(35, 650))
    hdl = df["bl_hdl"].where(df["bl_hdl"].between(5, 200))
    ldl = df["bl_ldl"].where(df["bl_ldl"].between(5, 500))
    tg = df["bl_tg"].where(fast & df["bl_tg"].between(20, 1000))
    rc = (cho - hdl - ldl).where((cho - hdl - ldl) > 0)
    crp = df["bl_crp"].where(df["bl_crp"] > 0)
    sbp = df["systo"].where(df["systo"].between(60, 260))
    dbp = df["diasto"].where(df["diasto"].between(40, 180))
    valid_bp = sbp.notna() & dbp.notna() & (sbp > dbp)
    sbp = sbp.where(valid_bp)
    dbp = dbp.where(valid_bp)
    waist = df["mwaist"].where(df["mwaist"].between(45, 180))
    height_m = df["mheight"].where(df["mheight"].between(1.10, 2.20))
    bmi = df["bmi"].where(df["bmi"].between(12, 65))
    lg = df["lgrip"].where(df["lgrip"].gt(0) & df["lgrip"].le(90))
    rg = df["rgrip"].where(df["rgrip"].gt(0) & df["rgrip"].le(90))
    hgs = pd.concat([lg, rg], axis=1).max(axis=1)

    if hgi_params is None:
        ok_hgi = glu.notna() & hba1c.notna()
        x = glu[ok_hgi].to_numpy()
        y = hba1c[ok_hgi].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
        hgi_params = (float(intercept), float(slope), r2, int(ok_hgi.sum()))
    a, b, _, _ = hgi_params
    hgi = (hba1c - (a + b * glu)).where(glu.notna() & hba1c.notna())
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
    whtr = (waist / (100.0 * height_m)).where(waist.notna() & height_m.notna())
    hgs_bmi = (hgs / bmi).where(hgs.notna() & bmi.notna())
    tyg = np.log(tg * glu / 2.0).where(tg.notna() & glu.notna())

    out = pd.DataFrame({
        "hgi": hgi, "rcii": rcii, "epwv": epwv, "whtr": whtr, "hgs_bmi": hgs_bmi,
        "tyg": tyg, "rc": rc, "crp": crp, "glu": glu, "hba1c": hba1c,
        "sbp": sbp, "dbp": dbp, "waist": waist, "bmi": bmi, "hgs": hgs,
    }, index=df.index)
    return out, hgi_params

def build():
    dta_path = next(CHARLS_DIR.rglob("charls.dta"))
    cols = [
        "ID", "wave", "communityID", "province", "iwy", "iwm", "age", "gender",
        "edu", "rural", "rural2", "marry", "smoken", "drinkl",
        "diabe", "hearte", "stroke", "kidneye", "cancre", "memrye", "hibpe", "dyslipe",
        "systo", "diasto", "lgrip", "rgrip", "mheight", "mwaist", "bmi",
        "bl_fasting", "bl_glu", "bl_hbalc", "bl_cho", "bl_hdl", "bl_ldl", "bl_tg", "bl_crp", "bl_crea",
    ]
    panel = pd.read_stata(dta_path, columns=cols, convert_categoricals=False)
    panel["ID"] = norm_id(panel["ID"])

    # Verify gender coding
    print("CHARLS gender value counts:", panel["gender"].value_counts(dropna=False).to_dict())
    # In CHARLS.dta: 1=male, 0=female
    panel["female"] = np.where(panel["gender"].eq(0), 1.0, np.where(panel["gender"].eq(1), 0.0, np.nan))

    w1 = panel[panel["wave"].eq(1)].drop_duplicates("ID").set_index("ID")
    w2 = panel[panel["wave"].eq(2)].drop_duplicates("ID").set_index("ID")
    w3 = panel[panel["wave"].eq(3)].drop_duplicates("ID").set_index("ID")

    # Harmonized CHARLS meds & death
    h_fp = next(CHARLS_DIR.rglob("H_CHARLS_D_Data.dta"))
    h_df = read_stata_idx(
        h_fp,
        ["ID", "r1rxhibp_c", "r3rxhibp_c", "r3rxdiab_c", "r3rxdiabi", "r3rxdyslip_c", "radyear", "radmonth", "raeducl"],
    )

    # Raw W3, W4, W5, W6 disease & dates
    w3_zip = next(CHARLS_DIR.rglob("CHARLS2015r.zip"))
    w4_zip = next(CHARLS_DIR.rglob("CHARLS2018r.zip"))
    w5_zip = next(CHARLS_DIR.rglob("CHARLS2020r.zip"))
    w6_zip = next(CHARLS_DIR.rglob("CHARLS_W6.zip"))

    w3_raw = raw_w3_baseline(w3_zip)
    w4_dis = followup_wave(w4_zip, "da007", "da010_w2_2", "da009_1")
    w5_dis = followup_wave(w5_zip, "da003", "da002", None)
    w6_dis = read_wave6(w6_zip)

    with ZipFile(w4_zip) as z:
        w4_si = read_zip_stata(z, "Sample_Infor.dta", ["ID", "iyear", "imonth", "died"])
    with ZipFile(w5_zip) as z:
        w5_si = read_zip_stata(z, "Sample_Infor.dta", ["ID", "iyear", "imonth", "died"])
        w5_ex = read_zip_stata(z, "Exit_Module.dta", ["ID", "exb001_1", "exb001_2", "exda003", "exda007"])

    # Build W3 core frame
    df = w3.copy()
    axes_w3, hgi_params_w3 = compute_axes(df, fasting_only=True)
    axes_w3_nofast, _ = compute_axes(df, fasting_only=False, hgi_params=hgi_params_w3)

    # W1 axes using locked W3 HGI equation
    axes_w1, hgi_params_w1_own = compute_axes(w1, fasting_only=True, hgi_params=hgi_params_w3)
    axes_w1_own, _ = compute_axes(w1, fasting_only=True, hgi_params=None)

    for c in axes_w3.columns:
        df[c] = axes_w3[c]
    for c in ["hgi", "rcii", "epwv", "whtr", "hgs_bmi"]:
        df[f"{c}_nofast"] = axes_w3_nofast[c]
        df[f"{c}_w1"] = axes_w1[c].reindex(df.index)
    df["hgi_w1_own"] = axes_w1_own["hgi"].reindex(df.index)
    df["fasting_w3"] = df["bl_fasting"].eq(1).astype(int)

    # Dates
    iwy3 = pd.to_numeric(df["iwy"], errors="coerce").fillna(2015)
    iwm3 = pd.to_numeric(df["iwm"], errors="coerce")
    df["date_w3"] = iwy3 + (iwm3.where(iwm3.between(1, 12), 7.5) - 0.5) / 12.0

    for wnum, si in [(4, w4_si), (5, w5_si), (6, w6_dis)]:
        yr = pd.to_numeric(si["iyear"], errors="coerce")
        mo = pd.to_numeric(si["imonth"], errors="coerce")
        fb = {4: 2018.5, 5: 2020.5, 6: 2021.8}[wnum]
        dt = yr + (mo.where(mo.between(1, 12), 7.5) - 0.5) / 12.0
        dt = dt.where(yr.between(2017, 2024), np.nan)
        df[f"date_w{wnum}"] = dt.reindex(df.index)
        df[f"died_w{wnum}"] = si["died"].reindex(df.index).eq(1).astype(int)

    # Exact death date
    rad_yr = pd.to_numeric(h_df["radyear"], errors="coerce").reindex(df.index)
    rad_mo = pd.to_numeric(h_df["radmonth"], errors="coerce").reindex(df.index)
    ex_yr = pd.to_numeric(w5_ex["exb001_1"], errors="coerce").reindex(df.index)
    ex_mo = pd.to_numeric(w5_ex["exb001_2"], errors="coerce").reindex(df.index)
    exact_dyr = rad_yr.where(rad_yr.between(2015, 2024), ex_yr.where(ex_yr.between(2015, 2024)))
    exact_dmo = rad_mo.where(rad_yr.between(2015, 2024), ex_mo.where(ex_yr.between(2015, 2024)))
    exact_ddate = exact_dyr + (exact_dmo.where(exact_dmo.between(1, 12), 6.5) - 0.5) / 12.0

    # Wave-level death date fallback (midpoint between last alive wave and death-reported wave)
    any_died = df["died_w4"].eq(1) | df["died_w5"].eq(1) | df["died_w6"].eq(1) | exact_ddate.notna()
    first_dead_wave_dt = np.select(
        [df["died_w4"].eq(1), df["died_w5"].eq(1), df["died_w6"].eq(1)],
        [df["date_w4"].fillna(2018.5), df["date_w5"].fillna(2020.5), df["date_w6"].fillna(2022.0)],
        default=np.nan,
    )
    last_alive_dt = df["date_w3"].copy()
    last_alive_dt = np.where(df["died_w4"].eq(0) & df["date_w4"].notna(), df["date_w4"], last_alive_dt)
    last_alive_dt = np.where(df["died_w5"].eq(0) & df["date_w5"].notna(), df["date_w5"], last_alive_dt)
    mid_ddate = 0.5 * (last_alive_dt + first_dead_wave_dt)
    df["death_date"] = np.where(exact_ddate.notna(), exact_ddate, mid_ddate)
    df["died_any"] = any_died.astype(int)
    # Safeguard death_date > date_w3
    df.loc[df["died_any"].eq(1) & (df["death_date"] <= df["date_w3"]), "death_date"] = df["date_w3"] + 0.5

    # Baseline disease prevalence (combining W1, W2, W3 panel + W3 raw)
    for d, pcol in [("diabetes", "diabe"), ("heart", "hearte"), ("stroke", "stroke"), ("kidney", "kidneye"), ("cancer", "cancre")]:
        past = w1[pcol].eq(1).reindex(df.index, fill_value=False) | w2[pcol].eq(1).reindex(df.index, fill_value=False)
        w3_p = df[pcol].eq(1)
        w3_r = w3_raw[d].reindex(df.index).eq(1)
        df[f"base_{d}"] = (past | w3_p | w3_r).astype(int)
    df["base_memory"] = (
        w1["memrye"].eq(1).reindex(df.index, fill_value=False)
        | w2["memrye"].eq(1).reindex(df.index, fill_value=False)
        | df["memrye"].eq(1)
    ).astype(int)

    # Medications & covariates at W3
    df["med_htn"] = h_df["r3rxhibp_c"].reindex(df.index).eq(1).astype(int)
    df["med_diab"] = (
        h_df["r3rxdiab_c"].reindex(df.index).eq(1)
        | h_df["r3rxdiabi"].reindex(df.index).eq(1)
        | w3_raw["diabetes_med_raw"].reindex(df.index).eq(1)
    ).astype(int)
    df["med_lipid"] = h_df["r3rxdyslip_c"].reindex(df.index).eq(1).astype(int)
    df["htn_status"] = np.where(
        df["hibpe"].eq(1) | df["med_htn"].eq(1) | (df["sbp"] >= 140) | (df["dbp"] >= 90),
        1,
        np.where(df["hibpe"].eq(0) & df["sbp"].lt(140) & df["dbp"].lt(90), 0, np.nan),
    )
    df["egdr"] = (
        21.158 - 0.09 * df["waist"] - 3.407 * df["htn_status"] - 0.551 * df["hba1c"]
    ).where(df["waist"].notna() & df["htn_status"].notna() & df["hba1c"].notna())
    df["egfr"] = ckd_epi_2021(df["bl_crea"], df["age"], df["female"])

    # Harmonized education (1: <lower sec, 2: upper sec/voc, 3: tertiary) fallback to CHARLS edu
    raedu = h_df["raeducl"].reindex(df.index)
    df["edu_cat"] = np.where(
        raedu.isin([1, 2, 3]), raedu,
        np.where(df["edu"].isin([1, 2]), 1.0, np.where(df["edu"].eq(3), 2.0, np.where(df["edu"].eq(4), 3.0, np.nan)))
    )
    df["rural_status"] = df["rural"].where(df["rural"].isin([0, 1]))
    df["marry_status"] = df["marry"].where(df["marry"].isin([0, 1]))
    df["smoke_cur"] = df["smoken"].where(df["smoken"].isin([0, 1]))
    df["drink_cur"] = df["drinkl"].where(df["drinkl"].isin([0, 1]))
    df["dyslip_status"] = np.where(
        df["dyslipe"].eq(1) | df["med_lipid"].eq(1) | (df["bl_cho"] >= 240) | (df["bl_tg"] >= 200) | (df["bl_hdl"] < 40),
        1.0,
        np.where(df["dyslipe"].eq(0), 0.0, np.nan),
    )
    df["base_diabetes_all"] = ((df["base_diabetes"] == 1) | (df["med_diab"] == 1)).astype(int)
    df["base_cvd"] = ((df["base_heart"] == 1) | (df["base_stroke"] == 1)).astype(int)
    df["n_cmd_base"] = df["base_diabetes_all"] + df["base_heart"] + df["base_stroke"]
    df["base_cmm"] = (df["n_cmd_base"] >= 2).astype(int)

    # CKM stage proxy (0-1 vs 2 vs 3) among CVD-free / CMM-free
    high_met = (df["htn_status"] == 1) | (df["base_diabetes_all"] == 1) | (df["dyslip_status"] == 1) | (df["egfr"] < 60)
    very_high_risk = (df["egfr"] < 30) | ((df["age"] >= 65) & (df["htn_status"] == 1) & (df["base_diabetes_all"] == 1))
    df["ckm_proxy"] = np.where(very_high_risk, 3, np.where(high_met, 2, 1))

    # Attach W4, W5, W6 disease statuses
    for d in ["diabetes", "heart", "stroke", "kidney"]:
        df[f"{d}_w4"] = w4_dis[d].reindex(df.index)
        df[f"{d}_w5"] = w5_dis[d].reindex(df.index)
        df[f"{d}_w6"] = w6_dis[d].reindex(df.index)
    # Also incorporate Exit_Module acute heart attack / stroke prior to death if missed
    df.loc[w5_ex["exda003"].reindex(df.index).eq(1), "heart_w5"] = 1.0
    df.loc[w5_ex["exda007"].reindex(df.index).eq(1), "stroke_w5"] = 1.0

    # Follow-up status check
    has_w4 = df["date_w4"].notna() | df["heart_w4"].notna() | df["died_w4"].eq(1)
    has_w5 = df["date_w5"].notna() | df["heart_w5"].notna() | df["died_w5"].eq(1)
    has_w6 = df["date_w6"].notna() | df["heart_w6"].notna() | df["died_w6"].eq(1)
    df["has_followup"] = (has_w4 | has_w5 | has_w6 | df["died_any"].eq(1)).astype(int)

    # Last alive interview date for censoring
    last_int = df["date_w3"].copy()
    for wnum, fb in [(4, 2018.5), (5, 2020.5), (6, 2022.0)]:
        present = df[f"date_w{wnum}"].notna() | df[f"{'heart'}_w{wnum}"].notna()
        dt_use = df[f"date_w{wnum}"].fillna(fb)
        last_int = np.where(present & (df[f"died_w{wnum}"] == 0), dt_use, last_int)
    df["last_alive_date"] = last_int

    # Compute exact incident event date & status for each target endpoint
    def compute_endpoint(df_in, name, base_flag_col, wave_pos_fn):
        # Returns (status: 0=cens, 1=event, 2=competing_death; time_yrs; wave_first)
        d_w3 = df_in["date_w3"].to_numpy()
        d_w4 = df_in["date_w4"].fillna(2018.5).to_numpy()
        d_w5 = df_in["date_w5"].fillna(2020.5).to_numpy()
        d_w6 = df_in["date_w6"].fillna(2022.0).to_numpy()

        p4, p5, p6 = wave_pos_fn(df_in)
        o4 = df_in["heart_w4"].notna().to_numpy()
        o5 = df_in["heart_w5"].notna().to_numpy()
        o6 = df_in["heart_w6"].notna().to_numpy()

        first_w = np.where(p4, 4, np.where(p5, 5, np.where(p6, 6, 0)))
        # Last negative date before first positive wave
        prev_neg_4 = d_w3
        prev_neg_5 = np.where(o4 & ~p4, d_w4, d_w3)
        prev_neg_6 = np.where(o5 & ~p5, d_w5, prev_neg_5)

        ev_date = np.select(
            [first_w == 4, first_w == 5, first_w == 6],
            [0.5 * (prev_neg_4 + d_w4), 0.5 * (prev_neg_5 + d_w5), 0.5 * (prev_neg_6 + d_w6)],
            default=np.nan,
        )
        has_ev = first_w > 0
        died = df_in["died_any"].to_numpy() == 1
        d_dt = df_in["death_date"].to_numpy()
        cens_dt = df_in["last_alive_date"].to_numpy()

        # Competing death if died and (no event OR death before event)
        is_death = died & (~has_ev | (d_dt < ev_date))
        status = np.where(has_ev & ~is_death, 1, np.where(is_death, 2, 0))
        end_dt = np.where(status == 1, ev_date, np.where(status == 2, d_dt, cens_dt))
        time_yrs = np.maximum(end_dt - d_w3, 0.08)
        # Mask baseline prevalent
        prev = df_in[base_flag_col].to_numpy() == 1
        status = np.where(prev, np.nan, status)
        time_yrs = np.where(prev, np.nan, time_yrs)
        first_w = np.where(prev, np.nan, first_w)
        return status, time_yrs, first_w

    # Single disease endpoints
    for dis, base_col in [("diabetes", "base_diabetes_all"), ("heart", "base_heart"), ("stroke", "base_stroke")]:
        st, tm, fw = compute_endpoint(
            df, dis, base_col,
            lambda d, dis=dis: (d[f"{dis}_w4"].eq(1).to_numpy(), d[f"{dis}_w5"].eq(1).to_numpy(), d[f"{dis}_w6"].eq(1).to_numpy())
        )
        df[f"status_{dis}"] = st
        df[f"time_{dis}"] = tm
        df[f"first_wave_{dis}"] = fw

    # CVD (heart OR stroke)
    st, tm, fw = compute_endpoint(
        df, "cvd", "base_cvd",
        lambda d: (
            (d["heart_w4"].eq(1) | d["stroke_w4"].eq(1)).to_numpy(),
            (d["heart_w5"].eq(1) | d["stroke_w5"].eq(1)).to_numpy(),
            (d["heart_w6"].eq(1) | d["stroke_w6"].eq(1)).to_numpy(),
        )
    )
    df["status_cvd"] = st
    df["time_cvd"] = tm
    df["first_wave_cvd"] = fw

    # Cumulative disease status at W4, W5, W6 for CMM (>=2 of diabetes, heart, stroke)
    def cum_cmd(d, w):
        diab = d["base_diabetes_all"].eq(1)
        hrt = d["base_heart"].eq(1)
        stk = d["base_stroke"].eq(1)
        for ww in [4, 5, 6]:
            if ww <= w:
                diab = diab | d[f"diabetes_w{ww}"].eq(1)
                hrt = hrt | d[f"heart_w{ww}"].eq(1)
                stk = stk | d[f"stroke_w{ww}"].eq(1)
        return diab.astype(int) + hrt.astype(int) + stk.astype(int)

    df["n_cmd_w4"] = cum_cmd(df, 4)
    df["n_cmd_w5"] = cum_cmd(df, 5)
    df["n_cmd_w6"] = cum_cmd(df, 6)

    st, tm, fw = compute_endpoint(
        df, "cmm", "base_cmm",
        lambda d: (d["n_cmd_w4"].ge(2).to_numpy(), d["n_cmd_w5"].ge(2).to_numpy(), d["n_cmd_w6"].ge(2).to_numpy())
    )
    df["status_cmm"] = st
    df["time_cmm"] = tm
    df["first_wave_cmm"] = fw

    # First CMD (any of diabetes, heart, stroke) among baseline CMD-free (n_cmd_base == 0)
    df["base_any_cmd"] = (df["n_cmd_base"] >= 1).astype(int)
    st, tm, fw = compute_endpoint(
        df, "fcmd", "base_any_cmd",
        lambda d: (d["n_cmd_w4"].ge(1).to_numpy(), d["n_cmd_w5"].ge(1).to_numpy(), d["n_cmd_w6"].ge(1).to_numpy())
    )
    df["status_fcmd"] = st
    df["time_fcmd"] = tm
    df["first_wave_fcmd"] = fw

    # All-cause mortality
    df["status_death"] = df["died_any"].astype(int)
    df["time_death"] = np.maximum(
        np.where(df["died_any"] == 1, df["death_date"], df["last_alive_date"]) - df["date_w3"],
        0.08,
    )

    # Wave 6 exclusive endpoints
    df["w6_dementia"] = w6_dis["dementia_new_w6"].reindex(df.index)
    df["w6_atrophy"] = w6_dis["atrophy_new_w6"].reindex(df.index)
    df["w6_dementia_atrophy"] = np.where(
        df["w6_dementia"].notna() | df["w6_atrophy"].notna(),
        np.where(df["w6_dementia"].eq(1) | df["w6_atrophy"].eq(1), 1.0, 0.0),
        np.nan,
    )
    df["w6_cm_hosp"] = w6_dis["w6_cm_hosp"].reindex(df.index)
    df["w6_card_hosp"] = w6_dis["w6_card_hosp"].reindex(df.index)
    df["w6_heart_attack"] = w6_dis["heart_attack_w6"].reindex(df.index)

    # STROBE Flow Tracking
    axes_cols = ["hgi", "rcii", "epwv", "whtr", "hgs_bmi"]
    s0 = df["age"].between(45, 110)
    s1 = s0 & df["fasting_w3"].eq(1) & df["glu"].notna() & df["hba1c"].notna() & df["rc"].notna() & df["crp"].notna()
    s2 = s1 & df[axes_cols].notna().all(axis=1)
    s3 = s2 & df["base_cancer"].eq(0)
    s4 = s3 & df["has_followup"].eq(1) & df["time_death"].gt(0)
    cov_m2 = ["age", "female", "edu_cat", "rural_status", "marry_status", "smoke_cur", "drink_cur"]
    s5 = s4 & df[cov_m2].notna().all(axis=1)
    df["eligible_w3"] = s5.astype(int)

    # Paired W1 & W3 5-axis flag
    w1_cols = [f"{c}_w1" for c in axes_cols]
    df["paired_w1_w3"] = (df["eligible_w3"].eq(1) & df[w1_cols].notna().all(axis=1)).astype(int)

    flow_rows = [
        ("1. CHARLS Wave 3 (2015) respondents aged >=45 years", int(s0.sum()), 0),
        ("2. Fasting blood complete for HGI & RCII (RC>0, CRP>0)", int(s1.sum()), int(s0.sum() - s1.sum())),
        ("3. Complete 5 physiological axes (add ePWV, WHtR, HGS/BMI)", int(s2.sum()), int(s1.sum() - s2.sum())),
        ("4. Free of baseline cancer", int(s3.sum()), int(s2.sum() - s3.sum())),
        ("5. >=1 follow-up wave (W4-W6) or vital status ascertained", int(s4.sum()), int(s3.sum() - s4.sum())),
        ("6. Complete core Model 2 covariates (Discovery Landmark Cohort)", int(s5.sum()), int(s4.sum() - s5.sum())),
        ("   6a. Baseline CMM-free analytic cohort (<2 CMD)", int((s5 & df["base_cmm"].eq(0)).sum()), int((s5 & df["base_cmm"].eq(1)).sum())),
        ("   6b. Baseline CVD-free analytic cohort", int((s5 & df["base_cvd"].eq(0)).sum()), int((s5 & df["base_cvd"].eq(1)).sum())),
        ("   6c. Baseline CMD-free Multi-State cohort (0 CMD)", int((s5 & df["n_cmd_base"].eq(0)).sum()), int((s5 & df["n_cmd_base"].gt(0)).sum())),
        ("   6d. Paired W1 (2011) & W3 (2015) 5-axis Transition sub-cohort", int(df["paired_w1_w3"].sum()), int(s5.sum() - df["paired_w1_w3"].sum())),
    ]
    flow_df = pd.DataFrame(flow_rows, columns=["step", "n_remaining", "n_excluded"])
    flow_df.to_csv(OUT_DIR / "tables" / "table_flow_charls.csv", index=False, encoding="utf-8-sig")
    flow_df.to_csv(OUT_DIR / "figure_data" / "fig1_flow_charls.csv", index=False, encoding="utf-8-sig")

    # Save HGI equation QC
    qc_hgi = pd.DataFrame([
        {"cohort": "CHARLS_W3", "intercept": hgi_params_w3[0], "slope_per_mgdl": hgi_params_w3[1], "r2": hgi_params_w3[2], "n": hgi_params_w3[3]},
        {"cohort": "CHARLS_W1_own", "intercept": hgi_params_w1_own[0], "slope_per_mgdl": hgi_params_w1_own[1], "r2": hgi_params_w1_own[2], "n": hgi_params_w1_own[3]},
    ])
    qc_hgi.to_csv(OUT_DIR / "tables" / "table_qc_equations.csv", index=False, encoding="utf-8-sig")

    # Export full eligible + nonfasting sensitivity dataset (age>=45 & no cancer & followup & all axes under nofast)
    keep_mask = s0 & df[[f"{c}_nofast" for c in axes_cols]].notna().all(axis=1) & df["base_cancer"].eq(0) & df["has_followup"].eq(1)
    out_df = df.loc[keep_mask].copy()
    out_df.to_csv(OUT_DIR / "derived" / "charls_analytic.csv", index=True, encoding="utf-8-sig")
    print("CHARLS build complete!")
    print(flow_df.to_string(index=False))
    print("W3 HGI equation:", hgi_params_w3)

if __name__ == "__main__":
    build()
