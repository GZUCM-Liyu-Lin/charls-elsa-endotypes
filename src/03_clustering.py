from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import OUT_DIR, ensure_dirs
ensure_dirs()
AXES = ["hgi", "rcii", "epwv", "whtr", "hgs_bmi"]
SEED = 42

def fit_sex_z_params(df, axes=AXES):
    params = {}
    for sex_val, sex_name in [(0.0, "male"), (1.0, "female")]:
        sub = df[df["female"] == sex_val]
        params[sex_val] = {}
        for ax in axes:
            m = float(sub[ax].mean())
            s = float(sub[ax].std(ddof=0))
            params[sex_val][ax] = (m, s)
    return params

def apply_sex_z(df, params, suffix="", clip=4.0):
    Z = pd.DataFrame(index=df.index)
    for ax in AXES:
        col = f"{ax}{suffix}"
        mu = df["female"].map({0.0: params[0.0][ax][0], 1.0: params[1.0][ax][0]})
        sd = df["female"].map({0.0: params[0.0][ax][1], 1.0: params[1.0][ax][1]})
        z = (df[col] - mu) / sd
        if clip is not None:
            z = z.clip(-clip, clip)
        Z[ax] = z
    return Z

def hungarian_match(ref_centers, new_centers, new_labels):
    # Match new_centers to ref_centers minimizing Euclidean distance
    k = ref_centers.shape[0]
    cost = np.linalg.norm(ref_centers[:, None, :] - new_centers[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {col_ind[i]: row_ind[i] for i in range(k)}
    matched_labels = np.array([mapping[l] for l in new_labels])
    matched_centers = np.zeros_like(new_centers)
    for new_c, ref_c in mapping.items():
        matched_centers[ref_c] = new_centers[new_c]
    return matched_labels, matched_centers

def bootstrap_stability(Z_arr, base_centers, base_labels, k, n_boot=200):
    rng = np.random.default_rng(SEED)
    n = len(Z_arr)
    aris = []
    jaccards = {c: [] for c in range(k)}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Zb = Z_arr[idx]
        km = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=SEED + b)
        lb = km.fit_predict(Zb)
        mlab, _ = hungarian_match(base_centers, km.cluster_centers_, lb)
        # Compare on unique sampled indices against base_labels[idx]
        ref_b = base_labels[idx]
        aris.append(adjusted_rand_score(ref_b, mlab))
        for c in range(k):
            in_ref = ref_b == c
            in_boot = mlab == c
            union = np.sum(in_ref | in_boot)
            inter = np.sum(in_ref & in_boot)
            jaccards[c].append(inter / union if union > 0 else 0.0)
    mean_jaccard = float(np.mean([np.median(jaccards[c]) for c in range(k)]))
    return float(np.median(aris)), float(np.percentile(aris, 25)), float(np.percentile(aris, 75)), mean_jaccard

def assign_nearest(Z_df, centers):
    arr = Z_df[AXES].to_numpy()
    dists = np.linalg.norm(arr[:, None, :] - centers[None, :, :], axis=2)
    return pd.Series(np.argmin(dists, axis=1), index=Z_df.index)

def name_endotypes(centers_df):
    """Assign objective, data-driven labels ordered by cluster index 1..K."""
    names = {}
    used = set()
    for cid, row in centers_df.iterrows():
        if cid == 1:
            names[cid] = "E1: Resilient / Low-Risk"
            used.add(names[cid])
            continue
        # Rank axes by absolute deviation in adverse direction (low HGS_BMI is adverse)
        adv = {
            "Hyperglycation (HGI)": row["hgi"],
            "Inflammatory-Lipid (RCII)": row["rcii"],
            "Vascular Stiffness (ePWV)": row["epwv"],
            "Central Adiposity (WHtR)": row["whtr"],
            "Sarcopenic Weakness (-HGS/BMI)": -row["hgs_bmi"],
        }
        top2 = sorted(adv.items(), key=lambda x: x[1], reverse=True)
        # Check if multi-axis high risk (>=3 axes > +0.30)
        n_high = sum(1 for _, v in top2 if v >= 0.30)
        if n_high >= 3 and top2[0][1] >= 0.45:
            cand = f"E{cid}: Multi-Axis Cardiometabolic High-Risk"
        elif top2[0][1] >= 0.45 and top2[1][1] >= 0.30:
            a1 = top2[0][0].split(" (")[0]
            a2 = top2[1][0].split(" (")[0]
            cand = f"E{cid}: {a1}–{a2}"
        else:
            a1 = top2[0][0].split(" (")[0]
            cand = f"E{cid}: {a1}-Dominant"
        if cand in used:
            cand = f"{cand} (Tier {cid})"
        used.add(cand)
        names[cid] = cand
    return names

def main():
    charls = pd.read_csv(OUT_DIR / "derived" / "charls_analytic.csv", index_col="ID", low_memory=False)
    elsa = pd.read_csv(OUT_DIR / "derived" / "elsa_analytic.csv", index_col="idauniq", low_memory=False)

    ch_el = charls[charls["eligible_w3"] == 1].copy()
    el_el = elsa[elsa["eligible_w4"] == 1].copy()

    # 1. Spearman Correlations of the 5 Axes
    corr_rows = []
    for cname, df_c in [("CHARLS_W3", ch_el), ("ELSA_W4", el_el)]:
        for i, a1 in enumerate(AXES):
            for j, a2 in enumerate(AXES):
                r, p = spearmanr(df_c[a1], df_c[a2], nan_policy="omit")
                corr_rows.append({"cohort": cname, "axis_1": a1, "axis_2": a2, "spearman_r": float(r), "p_value": float(p)})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_DIR / "tables" / "table_axis_correlations.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(OUT_DIR / "figure_data" / "fig1_axis_corr.csv", index=False, encoding="utf-8-sig")

    # 2. Sex-specific Z-standardization in CHARLS W3
    ch_z_params = fit_sex_z_params(ch_el)
    z_param_rows = []
    for sv, sn in [(0.0, "Male"), (1.0, "Female")]:
        for ax in AXES:
            z_param_rows.append({"cohort": "CHARLS_W3", "sex": sn, "axis": ax, "mean": ch_z_params[sv][ax][0], "sd": ch_z_params[sv][ax][1]})
    Z_ch = apply_sex_z(ch_el, ch_z_params, clip=4.0)
    Z_arr = Z_ch[AXES].to_numpy()

    # 3. K=2..6 evaluation + 200x bootstrap stability
    k_rows = []
    models = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=100, max_iter=1000, random_state=SEED)
        labels = km.fit_predict(Z_arr)
        sil = float(silhouette_score(Z_arr, labels))
        ch_idx = float(calinski_harabasz_score(Z_arr, labels))
        db_idx = float(davies_bouldin_score(Z_arr, labels))
        props = pd.Series(labels).value_counts(normalize=True)
        fem_by_c = ch_el["female"].groupby(labels).mean()
        ari_med, ari_q25, ari_q75, jacc_med = bootstrap_stability(Z_arr, km.cluster_centers_, labels, k, n_boot=200)
        k_rows.append({
            "k": k, "silhouette": sil, "calinski_harabasz": ch_idx, "davies_bouldin": db_idx,
            "inertia": float(km.inertia_), "min_cluster_pct": float(props.min() * 100),
            "max_cluster_pct": float(props.max() * 100),
            "female_pct_min": float(fem_by_c.min() * 100), "female_pct_max": float(fem_by_c.max() * 100),
            "boot_ari_median": ari_med, "boot_ari_q25": ari_q25, "boot_ari_q75": ari_q75,
            "boot_jaccard_median": jacc_med,
        })
        models[k] = (km, labels)
        print(f"K={k}: sil={sil:.3f}, CH={ch_idx:.1f}, DB={db_idx:.3f}, min%={props.min()*100:.1f}%, fem%=[{fem_by_c.min()*100:.1f}-{fem_by_c.max()*100:.1f}], boot_ARI={ari_med:.3f} [{ari_q25:.3f}-{ari_q75:.3f}]")

    k_df = pd.DataFrame(k_rows)
    k_df.to_csv(OUT_DIR / "tables" / "table_kmeans_selection.csv", index=False, encoding="utf-8-sig")
    k_df.to_csv(OUT_DIR / "figure_data" / "fig2_kmeans_k_metrics.csv", index=False, encoding="utf-8-sig")

    # Select optimal K in [4, 5] with ARI >= 0.70 & min_pct >= 5%, preferring highest ARI (or K=3 fallback)
    cands = k_df[k_df["k"].isin([4, 5]) & (k_df["min_cluster_pct"] >= 5.0) & (k_df["boot_ari_median"] >= 0.70)]
    if len(cands) > 0:
        opt_k = int(cands.sort_values("boot_ari_median", ascending=False).iloc[0]["k"])
    else:
        opt_k = int(k_df[k_df["k"].isin([3, 4, 5])].sort_values(["boot_ari_median", "silhouette"], ascending=False).iloc[0]["k"])
    print(f"Selected optimal K = {opt_k}")

    # Order clusters by CMM-free incident CMM rate (ascending) so E1 = Lowest Risk Reference
    km_opt, raw_lab = models[opt_k]
    ch_el["raw_cluster"] = raw_lab
    cmm_sub = ch_el[ch_el["base_cmm"] == 0].copy()
    cmm_sub["ev"] = (cmm_sub["status_cmm"] == 1).astype(int)
    # Composite risk score combining crude CMM rate and CVD rate for monotonic ordering
    cvd_sub = ch_el[ch_el["base_cvd"] == 0].copy()
    cvd_sub["ev"] = (cvd_sub["status_cvd"] == 1).astype(int)
    risk_rank = (cmm_sub.groupby("raw_cluster")["ev"].mean() + cvd_sub.groupby("raw_cluster")["ev"].mean()).sort_values()
    order_map = {old_c: new_i + 1 for new_i, old_c in enumerate(risk_rank.index)}

    ch_el["endotype"] = ch_el["raw_cluster"].map(order_map)
    ordered_centers = np.zeros_like(km_opt.cluster_centers_)
    for old_c, new_c in order_map.items():
        ordered_centers[new_c - 1] = km_opt.cluster_centers_[old_c]

    centers_z_df = pd.DataFrame(ordered_centers, index=range(1, opt_k + 1), columns=AXES)
    name_map = name_endotypes(centers_z_df)
    ch_el["endotype_label"] = ch_el["endotype"].map(name_map)

    # Save K-1, K+1, and non-sex-standardized K for sensitivity analyses
    for alt_label, alt_k in [("km1", max(2, opt_k - 1)), ("kp1", min(6, opt_k + 1))]:
        _, alt_lab = models[alt_k]
        ch_el[f"raw_{alt_label}"] = alt_lab
        alt_rank = ch_el[ch_el["base_cmm"] == 0].assign(ev=lambda d: (d["status_cmm"] == 1).astype(int)).groupby(f"raw_{alt_label}")["ev"].mean().sort_values()
        alt_map = {oc: ni + 1 for ni, oc in enumerate(alt_rank.index)}
        ch_el[f"endotype_{alt_label}"] = ch_el[f"raw_{alt_label}"].map(alt_map)

    # Non-sex-standardized K-means (sensitivity to verify sex-split trap)
    Z_nosex = ((ch_el[AXES] - ch_el[AXES].mean()) / ch_el[AXES].std(ddof=0)).clip(-4.0, 4.0)
    km_ns = KMeans(n_clusters=opt_k, n_init=50, random_state=SEED)
    ns_lab = km_ns.fit_predict(Z_nosex[AXES].to_numpy())
    ch_el["endotype_nosex"] = ns_lab + 1
    ns_fem = ch_el["female"].groupby(ch_el["endotype_nosex"]).mean() * 100
    print("Non-sex-standardized cluster female %:", ns_fem.round(1).to_dict())

    # Export CHARLS Centroids (both Z-score and Raw clinical values)
    cent_rows = []
    for eid in range(1, opt_k + 1):
        sub = ch_el[ch_el["endotype"] == eid]
        row = {
            "cohort": "CHARLS_W3", "endotype": eid, "label": name_map[eid],
            "n": len(sub), "pct": 100.0 * len(sub) / len(ch_el),
            "female_pct": 100.0 * sub["female"].mean(),
            "age_mean": sub["age"].mean(),
        }
        for ax in AXES:
            row[f"z_{ax}"] = float(centers_z_df.loc[eid, ax])
            row[f"raw_mean_{ax}"] = float(sub[ax].mean())
            row[f"raw_sd_{ax}"] = float(sub[ax].std())
            row[f"raw_med_{ax}"] = float(sub[ax].median())
        cent_rows.append(row)
    cent_df = pd.DataFrame(cent_rows)
    cent_df.to_csv(OUT_DIR / "tables" / "table_centroids_charls.csv", index=False, encoding="utf-8-sig")
    cent_df.to_csv(OUT_DIR / "figure_data" / "fig2_centroids_charls.csv", index=False, encoding="utf-8-sig")

    # PCA 2D projection coordinates for visualization
    pca = PCA(n_components=2, random_state=SEED)
    pcs = pca.fit_transform(Z_arr)
    pca_df = pd.DataFrame({
        "ID": ch_el.index,
        "endotype": ch_el["endotype"].to_numpy(),
        "endotype_label": ch_el["endotype_label"].to_numpy(),
        "pc1": pcs[:, 0],
        "pc2": pcs[:, 1],
        "pc1_var_exp": float(pca.explained_variance_ratio_[0]),
        "pc2_var_exp": float(pca.explained_variance_ratio_[1]),
    })
    pca_df.to_csv(OUT_DIR / "figure_data" / "fig2_pca_coords.csv", index=False, encoding="utf-8-sig")

    # 4. CHARLS 4-Year Phenotype Transitions (W1 -> W3)
    paired_ch = ch_el[ch_el["paired_w1_w3"] == 1].copy()
    Z_ch_w1 = apply_sex_z(paired_ch, ch_z_params, suffix="_w1", clip=4.0)
    paired_ch["endotype_w1"] = assign_nearest(Z_ch_w1, ordered_centers) + 1
    ch_el["endotype_w1"] = paired_ch["endotype_w1"].reindex(ch_el.index)

    # Full K x K transition matrix & Sankey export
    trans_mat = pd.crosstab(paired_ch["endotype_w1"], paired_ch["endotype"], margins=False)
    trans_long = []
    for w1_e in range(1, opt_k + 1):
        row_tot = int(trans_mat.loc[w1_e].sum()) if w1_e in trans_mat.index else 0
        for w3_e in range(1, opt_k + 1):
            cnt = int(trans_mat.loc[w1_e, w3_e]) if (w1_e in trans_mat.index and w3_e in trans_mat.columns) else 0
            trans_long.append({
                "cohort": "CHARLS_W1_to_W3",
                "from_endotype": w1_e, "from_label": name_map[w1_e],
                "to_endotype": w3_e, "to_label": name_map[w3_e],
                "n": cnt, "pct_of_from": (100.0 * cnt / row_tot) if row_tot else 0.0,
                "pct_of_total": 100.0 * cnt / len(paired_ch),
            })
    trans_df = pd.DataFrame(trans_long)
    trans_df.to_csv(OUT_DIR / "tables" / "table_transitions_charls.csv", index=False, encoding="utf-8-sig")
    trans_df.to_csv(OUT_DIR / "figure_data" / "fig4_sankey_charls.csv", index=False, encoding="utf-8-sig")

    # Define 4-level transition group (Low-Risk = E1 vs Elevated-Risk = E2..EK)
    def make_trans_group(e_prev, e_curr):
        return np.select(
            [(e_prev == 1) & (e_curr == 1), (e_prev > 1) & (e_curr == 1), (e_prev == 1) & (e_curr > 1), (e_prev > 1) & (e_curr > 1)],
            ["1_Stable_Low", "2_Favorable_Transition", "3_Adverse_Transition", "4_Stable_High"],
            default=None,
        )

    ch_el["trans_group"] = make_trans_group(ch_el["endotype_w1"], ch_el["endotype"])
    ch_el.loc[ch_el["endotype_w1"].isna(), "trans_group"] = np.nan

    # 5. ELSA External Replication (W4 & W2 -> W4)
    el_z_params = fit_sex_z_params(el_el)
    for sv, sn in [(0.0, "Male"), (1.0, "Female")]:
        for ax in AXES:
            z_param_rows.append({"cohort": "ELSA_W4", "sex": sn, "axis": ax, "mean": el_z_params[sv][ax][0], "sd": el_z_params[sv][ax][1]})
    pd.DataFrame(z_param_rows).to_csv(OUT_DIR / "tables" / "table_qc_zparams.csv", index=False, encoding="utf-8-sig")

    Z_el = apply_sex_z(el_el, el_z_params, clip=4.0)
    el_el["endotype"] = assign_nearest(Z_el, ordered_centers) + 1
    el_el["endotype_label"] = el_el["endotype"].map(name_map)

    # Sensitivity: direct projection using CHARLS W3 sex-specific Z params
    Z_el_chp = apply_sex_z(el_el, ch_z_params, clip=4.0)
    el_el["endotype_charls_params"] = assign_nearest(Z_el_chp, ordered_centers) + 1

    # De novo K-means in ELSA W4 & Hungarian match to CHARLS centroids
    Z_el_arr = Z_el[AXES].to_numpy()
    km_el = KMeans(n_clusters=opt_k, n_init=100, max_iter=1000, random_state=SEED)
    el_dn_raw = km_el.fit_predict(Z_el_arr)
    el_dn_lab, el_dn_centers = hungarian_match(ordered_centers, km_el.cluster_centers_, el_dn_raw)
    el_el["endotype_denovo"] = el_dn_lab + 1

    rep_ari = float(adjusted_rand_score(el_el["endotype"], el_el["endotype_denovo"]))
    rep_rows = []
    for eid in range(1, opt_k + 1):
        c_ch = ordered_centers[eid - 1]
        c_el_dn = el_dn_centers[eid - 1]
        cos_sim = float(np.dot(c_ch, c_el_dn) / (np.linalg.norm(c_ch) * np.linalg.norm(c_el_dn)))
        pear_r, _ = pearsonr(c_ch, c_el_dn)
        sub_p = el_el[el_el["endotype"] == eid]
        sub_d = el_el[el_el["endotype_denovo"] == eid]
        rep_rows.append({
            "endotype": eid, "label": name_map[eid],
            "elsa_proj_n": len(sub_p), "elsa_proj_pct": 100.0 * len(sub_p) / len(el_el),
            "elsa_denovo_n": len(sub_d), "elsa_denovo_pct": 100.0 * len(sub_d) / len(el_el),
            "centroid_cosine_sim": cos_sim, "centroid_pearson_r": float(pear_r),
            "overall_proj_vs_denovo_ari": rep_ari,
        })
    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(OUT_DIR / "tables" / "table_elsa_replication_qc.csv", index=False, encoding="utf-8-sig")

    for mode, col_e, cmat in [("ELSA_W4_Projected", "endotype", None), ("ELSA_W4_DeNovo", "endotype_denovo", el_dn_centers)]:
        erows = []
        for eid in range(1, opt_k + 1):
            sub = el_el[el_el[col_e] == eid]
            row = {
                "cohort": mode, "endotype": eid, "label": name_map[eid],
                "n": len(sub), "pct": 100.0 * len(sub) / len(el_el),
                "female_pct": 100.0 * sub["female"].mean(),
                "age_mean": sub["age"].mean(),
            }
            for idx_a, ax in enumerate(AXES):
                row[f"z_{ax}"] = float(Z_el.loc[sub.index, ax].mean()) if cmat is None else float(cmat[eid - 1, idx_a])
                row[f"raw_mean_{ax}"] = float(sub[ax].mean())
                row[f"raw_med_{ax}"] = float(sub[ax].median())
            erows.append(row)
        fn = "fig2_centroids_elsa_proj.csv" if cmat is None else "fig2_centroids_elsa_denovo.csv"
        pd.DataFrame(erows).to_csv(OUT_DIR / "figure_data" / fn, index=False, encoding="utf-8-sig")

    # ELSA W2 -> W4 Transitions
    paired_el = el_el[el_el["paired_w2_w4"] == 1].copy()
    Z_el_w2 = apply_sex_z(paired_el, el_z_params, suffix="_w2", clip=4.0)
    paired_el["endotype_w2"] = assign_nearest(Z_el_w2, ordered_centers) + 1
    el_el["endotype_w2"] = paired_el["endotype_w2"].reindex(el_el.index)
    el_el["trans_group"] = make_trans_group(el_el["endotype_w2"], el_el["endotype"])
    el_el.loc[el_el["endotype_w2"].isna(), "trans_group"] = np.nan

    el_tmat = pd.crosstab(paired_el["endotype_w2"], paired_el["endotype"], margins=False)
    el_tlong = []
    for w2_e in range(1, opt_k + 1):
        rt = int(el_tmat.loc[w2_e].sum()) if w2_e in el_tmat.index else 0
        for w4_e in range(1, opt_k + 1):
            cnt = int(el_tmat.loc[w2_e, w4_e]) if (w2_e in el_tmat.index and w4_e in el_tmat.columns) else 0
            el_tlong.append({
                "cohort": "ELSA_W2_to_W4",
                "from_endotype": w2_e, "from_label": name_map[w2_e],
                "to_endotype": w4_e, "to_label": name_map[w4_e],
                "n": cnt, "pct_of_from": (100.0 * cnt / rt) if rt else 0.0,
                "pct_of_total": 100.0 * cnt / len(paired_el),
            })
    el_tdf = pd.DataFrame(el_tlong)
    el_tdf.to_csv(OUT_DIR / "tables" / "table_transitions_elsa.csv", index=False, encoding="utf-8-sig")
    el_tdf.to_csv(OUT_DIR / "figure_data" / "fig4_sankey_elsa.csv", index=False, encoding="utf-8-sig")

    # Save clustered datasets (full eligible + merge back non-eligible with cluster=-1 for sensitivity S3)
    charls_out = charls.copy()
    for col in ["endotype", "endotype_label", "endotype_km1", "endotype_kp1", "endotype_nosex", "endotype_w1", "trans_group"]:
        charls_out[col] = ch_el[col].reindex(charls_out.index)
    # Also assign non-fasting eligible sensitivity participants using locked sex-z and centroids
    ch_nofast = charls[charls["age"].between(45, 110) & charls["base_cancer"].eq(0) & charls["has_followup"].eq(1)].copy()
    Z_nf = apply_sex_z(ch_nofast, ch_z_params, suffix="_nofast", clip=4.0)
    ok_nf = Z_nf[AXES].notna().all(axis=1)
    charls_out["endotype_nofast"] = np.nan
    charls_out.loc[ok_nf.index[ok_nf], "endotype_nofast"] = (assign_nearest(Z_nf.loc[ok_nf], ordered_centers) + 1).values
    charls_out.to_csv(OUT_DIR / "derived" / "charls_clustered.csv", index=True, encoding="utf-8-sig")

    elsa_out = elsa.copy()
    for col in ["endotype", "endotype_label", "endotype_denovo", "endotype_charls_params", "endotype_w2", "trans_group"]:
        elsa_out[col] = el_el[col].reindex(elsa_out.index)
    # Also assign nodrink sensitivity
    el_nd = elsa[elsa["eligible_w4_nodrink"] == 1].copy()
    Z_nd = apply_sex_z(el_nd, el_z_params, clip=4.0)
    elsa_out["endotype_nodrink"] = (assign_nearest(Z_nd, ordered_centers) + 1).reindex(elsa_out.index)
    elsa_out.to_csv(OUT_DIR / "derived" / "elsa_clustered.csv", index=True, encoding="utf-8-sig")

    print(cent_df[["endotype","label","n","pct","female_pct","z_hgi","z_rcii","z_epwv","z_whtr","z_hgs_bmi"]].round(3).to_string(index=False))
    print(rep_df.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
