from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.patches import Rectangle

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DERIVED_DIR, FIG_DATA_DIR as DATA_DIR, FIGURES_DIR as FIG_DIR, TABLES_DIR as TAB_DIR
import plot_style as ps

def main():
    ps.init_style()

    roc_df = pd.read_csv(os.path.join(DATA_DIR, "fig6_roc.csv"))
    cal_df = pd.read_csv(os.path.join(DATA_DIR, "fig6_calibration.csv"))
    dca_df = pd.read_csv(os.path.join(DATA_DIR, "fig6_dca.csv"))
    s6_df = pd.read_csv(os.path.join(DATA_DIR, "fig6_sensitivity.csv"))
    s5_tbl = pd.read_csv(os.path.join(TAB_DIR, "tableS5_prediction.csv"))

    H = 152.0
    cv = ps.Canvas(ps.FIG_W_MM, H)

    models_show = [
        ("M0_Base", "M0 Base", "--", 0.80),
        ("M1_Base_TyG", "M1 +TyG", "-", 0.80),
        ("M2_Base_eGDR", "M2 +eGDR", "-", 0.80),
        ("M4_Base_Endotypes", "M4 +Endotypes", "-", 1.05),
        ("M5_Base_All5Axes", "M5 +All 5 Axes", "-", 1.10),
    ]

    # TOP SHARED MODEL LEGEND (y = 1.0 .. 5.6 mm)
    ax_leg = cv.raw_ax(2.0, 1.0, 179.0, 4.8)
    leg_xs = [4.0, 32.0, 62.0, 96.0, 138.0]
    for (mk, mlbl, mls, mlw), lx in zip(models_show, leg_xs):
        col = ps.MODEL_COLORS[mk]
        ax_leg.plot([lx, lx + 4.5], [2.4, 2.4], color=col, lw=mlw, ls=mls)
        ax_leg.text(lx + 5.4, 2.4, mlbl, ha="left", va="center", size=ps.TEXT_PT, weight="bold" if mk in ("M4_Base_Endotypes", "M5_Base_All5Axes") else "normal")

    # ROW 1: Panels A (ROC), B (Calibration), C (DCA) — CHARLS 6-yr CMM Model 2
    y_lbl1 = 6.6
    y_r1 = 12.6
    h_r1 = 38.0

    # --- Panel A: ROC Curves ---
    cv.panel_label(1.5, y_lbl1, "A")
    axA = cv.ax(13.0, y_r1, 43.0, h_r1)
    axA.plot([0, 1], [0, 1], ls=":", lw=0.55, color="#888888", zorder=0)
    r_sub = roc_df[(roc_df["cohort"] == "CHARLS") & (roc_df["endpoint"] == "cmm") & (roc_df["tier"] == "Model2")]
    auc_lines = []
    for mk, mlbl, mls, mlw in models_show:
        s = r_sub[r_sub["model"] == mk].sort_values("fpr")
        col = ps.MODEL_COLORS[mk]
        axA.plot(s["fpr"], s["tpr"], color=col, ls=mls, lw=mlw, zorder=3 if "M4" in mk or "M5" in mk else 2)
        auc_val = float(s["auc"].iloc[0])
        auc_lines.append((mlbl, auc_val, col))

    # Clean bottom-right AUC summary text stack
    for li, (mlbl, aval, col) in enumerate(auc_lines):
        yy = 0.27 - li * 0.056
        axA.text(0.97, yy, f"{mlbl}: {aval:.3f}", transform=axA.transAxes, ha="right", va="bottom", size=ps.TEXT_PT, color=col, weight="bold" if "M4" in mlbl or "M5" in mlbl else "normal")

    axA.set_xlim(0, 1)
    axA.set_ylim(0, 1)
    axA.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axA.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axA.set_xlabel("1 \u2212 Specificity (FPR)", size=ps.TEXT_PT, labelpad=1.8)
    axA.set_ylabel("Sensitivity (TPR)", size=ps.TEXT_PT, labelpad=2.0)
    axA.set_title("5-fold CV 6-yr CMM ROC", size=ps.TEXT_PT, weight="bold", pad=3.0)
    ps.style_l_axis(axA)

    # --- Panel B: Decile Calibration ---
    cv.panel_label(60.5, y_lbl1, "B")
    axB = cv.ax(71.5, y_r1, 43.0, h_r1)
    axB.plot([0, 0.25], [0, 0.25], ls="--", lw=0.55, color="#888888", zorder=0)
    c_sub = cal_df[(cal_df["cohort"] == "CHARLS") & (cal_df["endpoint"] == "cmm") & (cal_df["tier"] == "Model2")]
    s5_ch_m2 = s5_tbl[(s5_tbl["cohort"] == "CHARLS") & (s5_tbl["endpoint"] == "cmm") & (s5_tbl["tier"] == "Model2")]

    cal_models = [("M0_Base", "M0"), ("M4_Base_Endotypes", "M4"), ("M5_Base_All5Axes", "M5")]
    for ci, (mk, short_n) in enumerate(cal_models):
        s = c_sub[c_sub["model"] == mk].sort_values("decile")
        col = ps.MODEL_COLORS[mk]
        axB.plot(s["mean_pred_risk"], s["obs_ipcw_risk"], color=col, lw=0.9, marker="o", ms=2.8, mfc=col, mec="white", mew=0.35, zorder=2 + ci)
        brier = float(s5_ch_m2.loc[s5_ch_m2["model"] == mk, "brier"].iloc[0])
        axB.text(0.05, 0.94 - ci * 0.075, f"{short_n} Brier: {brier:.3f}", transform=axB.transAxes, ha="left", va="top", size=ps.TEXT_PT, color=col)

    axB.set_xlim(0, 0.25)
    axB.set_ylim(0, 0.25)
    ticks_b = [0, 0.08, 0.16, 0.24]
    axB.set_xticks(ticks_b)
    axB.set_yticks(ticks_b)
    axB.set_xlabel("Predicted 6-yr risk", size=ps.TEXT_PT, labelpad=1.8)
    axB.set_ylabel("Observed IPCW risk", size=ps.TEXT_PT, labelpad=2.0)
    axB.set_title("Decile calibration", size=ps.TEXT_PT, weight="bold", pad=3.0)
    ps.style_l_axis(axB)

    # --- Panel C: Decision Curve Analysis (DCA) ---
    cv.panel_label(119.0, y_lbl1, "C")
    axC = cv.ax(131.0, y_r1, 50.0, h_r1)
    d_sub = dca_df[(dca_df["cohort"] == "CHARLS") & (dca_df["endpoint"] == "cmm") & (dca_df["tier"] == "Model2")]
    d0 = d_sub[d_sub["model"] == "M0_Base"].sort_values("threshold")
    axC.plot(d0["threshold"], d0["treat_none"], color="#222222", lw=0.55, ls="-", zorder=1)
    axC.plot(d0["threshold"], d0["treat_all"], color="#999999", lw=0.6, ls=":", zorder=1)

    for mk, mlbl, mls, mlw in models_show:
        s = d_sub[d_sub["model"] == mk].sort_values("threshold")
        col = ps.MODEL_COLORS[mk]
        axC.plot(s["threshold"], s["net_benefit"], color=col, ls=mls, lw=mlw, zorder=3 if "M4" in mk or "M5" in mk else 2)

    axC.text(0.080, 0.005, "Treat all", ha="right", va="bottom", size=ps.TEXT_PT, color="#666666")
    axC.text(0.29, -0.0018, "Treat none", ha="right", va="top", size=ps.TEXT_PT, color="#222222")
    axC.set_xlim(0.01, 0.30)
    axC.set_ylim(-0.011, 0.10)
    axC.set_xticks([0.05, 0.15, 0.25])
    yt_c = [0.00, 0.03, 0.06, 0.09]
    axC.set_yticks(yt_c)
    axC.set_xlabel("Risk threshold probability", size=ps.TEXT_PT, labelpad=1.8)
    axC.set_ylabel("Net benefit", size=ps.TEXT_PT, labelpad=2.0)
    axC.set_title("Decision curve analysis", size=ps.TEXT_PT, weight="bold", pad=3.0)
    ps.style_l_axis(axC)

    # ROW 2: Panel D (Delta C-index) & Panel E (Sensitivity Forest + E-value)
    y_lbl2 = 57.5
    y_r2 = 67.5
    h_r2 = 75.5  # bottom = 143.0 mm (leaves 9.0 mm for x-axis ticks + label)

    # --- Panel D: Cross-cohort Delta C-index ---
    cv.panel_label(1.5, y_lbl2, "D")
    cv.text(24.0 + 19.5, 59.3, "Incremental \u0394C-index vs Base", size=ps.TEXT_PT, weight="bold", ha="center", va="center")
    # Compact 4-model legend cleanly above axD
    axD_leg = cv.raw_ax(12.0, 61.8, 52.0, 4.2)
    dc_models = [
        ("M1_Base_TyG", "+TyG"),
        ("M2_Base_eGDR", "+eGDR"),
        ("M4_Base_Endotypes", "+Endo"),
        ("M5_Base_All5Axes", "+5 Axes"),
    ]
    for di, (mk, dlbl) in enumerate(dc_models):
        lx = di * 13.0
        col = ps.MODEL_COLORS[mk]
        axD_leg.plot([lx], [2.1], marker="o", ms=3.4, mfc=col, mec="white", mew=0.35)
        axD_leg.text(lx + 1.8, 2.1, dlbl, ha="left", va="center", size=ps.TEXT_PT)

    axD = cv.ax(24.0, y_r2, 39.5, h_r2)
    dc_groups = [
        ("CHARLS", "cmm", "Model2", "CHARLS\nCMM (M2)"),
        ("CHARLS", "cmm", "Model3", "CHARLS\nCMM (M3)"),
        ("ELSA", "cmm", "Model3", "ELSA\nCMM (M3)"),
        ("CHARLS", "cvd", "Model3", "CHARLS\nCVD (M3)"),
        ("ELSA", "cvd", "Model3", "ELSA\nCVD (M3)"),
    ]
    for gi in range(len(dc_groups)):
        if gi % 2 == 1:
            axD.axhspan(gi - 0.45, gi + 0.45, color=ps.BG_ZEBRA, lw=0, zorder=0)
    axD.axvline(0.0, color="#777777", ls="--", lw=0.55, zorder=1)

    dodge_dc = [-0.24, -0.08, 0.08, 0.24]
    for gi, (coh, ep, tr, glbl) in enumerate(dc_groups):
        for di, (mk, _) in enumerate(dc_models):
            row = s5_tbl[(s5_tbl["cohort"] == coh) & (s5_tbl["endpoint"] == ep) & (s5_tbl["tier"] == tr) & (s5_tbl["model"] == mk)].iloc[0]
            col = ps.MODEL_COLORS[mk]
            yy = gi + dodge_dc[di]
            axD.plot([float(row["delta_c_ci_low"]), float(row["delta_c_ci_high"])], [yy, yy], color=col, lw=0.75, zorder=2)
            axD.plot([float(row["delta_c_vs_m0"])], [yy], marker="o", ms=3.2, mfc=col, mec="white", mew=0.35, zorder=3)

    axD.set_xlim(-0.025, 0.155)
    axD.set_ylim(-0.5, len(dc_groups) - 0.5)
    axD.invert_yaxis()
    axD.set_yticks(range(len(dc_groups)))
    axD.set_yticklabels([g[3] for g in dc_groups], size=ps.TEXT_PT, linespacing=1.05)
    xts_dc = [0.00, 0.05, 0.10, 0.15]
    axD.set_xticks(xts_dc)
    axD.set_xticklabels([f"{v:.2f}" for v in xts_dc])
    axD.set_xlabel("\u0394Harrell's C-index (95% CI)", size=ps.TEXT_PT, labelpad=1.8)
    ps.style_l_axis(axD)

    # --- Panel E: Sensitivity Stability Forest Plot (15 CMM scenarios) ---
    cv.panel_label(66.5, y_lbl2, "E")
    axE_hdr = cv.raw_ax(67.5, y_lbl2, 114.0, 9.8)
    axE_hdr.text(4.5, 2.0, "CMM sensitivity stability (highest-risk endotype vs E1, Model 3)", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axE_hdr.text(0.5, 7.0, "Scenario", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axE_hdr.text(72.0, 7.0, "HR (95% CI)", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axE_hdr.text(113.5, 7.0, "E-val (CI)", ha="right", va="center", size=ps.TEXT_PT, weight="bold")
    axE_hdr.plot([0.2, 113.5], [9.5, 9.5], color="#333333", lw=0.6)

    s6_cmm = s6_df[s6_df["endpoint"] == "cmm"].copy()
    # Select highest cluster number per (cohort, scenario)
    s6_top = s6_cmm.sort_values("endotype").groupby(["cohort", "scenario"], as_index=False).last()

    scen_order = [
        ("CHARLS", "S0_Primary_Model3", "CH: Primary (E4)"),
        ("CHARLS", "S1_Lag_2yr", "CH: 2-yr lag excl."),
        ("CHARLS", "S2_Excl_CRP_gt10", "CH: CRP \u226410 mg/L"),
        ("CHARLS", "S3_Incl_NonFasting", "CH: Incl. non-fasting"),
        ("CHARLS", "S4_MICE_m10", "CH: MICE (m = 10)"),
        ("CHARLS", "S5_AltK_3", "CH: K = 3 (E3 vs E1)"),
        ("CHARLS", "S5_AltK_5", "CH: K = 5 (E5 vs E1)"),
        ("ELSA", "S0_Primary_Model3", "EL: Primary (E4)"),
        ("ELSA", "S1_Lag_2yr", "EL: 2-yr lag excl."),
        ("ELSA", "S2_Excl_CRP_gt10", "EL: CRP \u226410 mg/L"),
        ("ELSA", "S4_MICE_m10", "EL: MICE (m = 10)"),
        ("ELSA", "S6a_ELSA_CHARLS_ZParams", "EL: CHARLS Z-params"),
        ("ELSA", "S6b_ELSA_DeNovo_KMeans", "EL: De novo K-means"),
        ("ELSA", "S6c_ELSA_NoDrinkDrop", "EL: No drink drop"),
        ("ELSA", "S6d_ELSA_Excl_W10Disputed", "EL: Excl. W10 disputed"),
    ]
    n_sc = len(scen_order)
    row_h_sc = h_r2 / n_sc

    axE_bg = cv.raw_ax(67.5, y_r2, 114.0, h_r2)
    axE_f = cv.ax(102.5, y_r2, 34.5, h_r2, facecolor="none")
    axE_f.set_xscale("log")
    axE_f.set_xlim(1.0, 9.0)
    axE_f.set_ylim(-0.5, n_sc - 0.5)
    axE_f.invert_yaxis()
    axE_f.axvline(1.0, color="#777777", ls="--", lw=0.55, zorder=1)
    xts_e = [1, 2, 4, 8]
    axE_f.set_xticks(xts_e)
    axE_f.set_xticklabels([str(v) for v in xts_e])
    axE_f.set_yticks([])
    axE_f.spines["top"].set_visible(False)
    axE_f.spines["right"].set_visible(False)
    axE_f.spines["left"].set_visible(False)
    axE_f.spines["bottom"].set_linewidth(0.6)
    axE_f.tick_params(direction="out", length=2.2, width=0.6, pad=1.8, labelsize=ps.TEXT_PT)
    axE_f.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axE_f.set_xlabel("Hazard ratio (95% CI, log scale)", size=ps.TEXT_PT, labelpad=1.8)

    # Zebra band for CHARLS vs ELSA block separation + divider line
    axE_bg.add_patch(Rectangle((0.2, 7 * row_h_sc), 113.3, 8 * row_h_sc, facecolor=ps.BG_ZEBRA, edgecolor="none", zorder=0))
    axE_bg.plot([0.2, 113.5], [7 * row_h_sc, 7 * row_h_sc], color="#BBBBBB", lw=0.5, ls="--", zorder=1)

    for ri, (coh, sc_k, sc_lbl) in enumerate(scen_order):
        ym = (ri + 0.5) * row_h_sc
        row = s6_top[(s6_top["cohort"] == coh) & (s6_top["scenario"] == sc_k)].iloc[0]
        is_prim = sc_k == "S0_Primary_Model3"
        col = ps.ENDOTYPE_COLORS[4] if coh == "CHARLS" else "#9E3520"
        axE_bg.text(0.5, ym, sc_lbl, ha="left", va="center", size=ps.TEXT_PT, weight="bold" if is_prim else "normal", color=ps.COHORT_COLORS[coh] if is_prim else "#111111")

        hr, lo, hi = float(row["hr"]), float(row["ci_low"]), float(row["ci_high"])
        axE_f.plot([lo, hi], [ri, ri], color=col, lw=0.75, zorder=2)
        axE_f.plot([hr], [ri], marker="s", ms=3.2, mfc=col, mec="white", mew=0.35, zorder=3)

        axE_bg.text(72.0, ym, ps.fmt_hr(hr, lo, hi), ha="left", va="center", size=ps.TEXT_PT, weight="bold" if is_prim else "normal")
        ev_pt, ev_ci = float(row["evalue_point"]), float(row["evalue_ci"])
        axE_bg.text(113.5, ym, f"{ev_pt:.2f} ({ev_ci:.2f})", ha="right", va="center", size=ps.TEXT_PT, color="#333333")

    axE_bg.plot([0.2, 113.5], [h_r2, h_r2], color="#333333", lw=0.5)
    ps.save_fig(cv, FIG_DIR, "Figure6")

if __name__ == "__main__":
    main()
