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

def _within_sex_z(df: pd.DataFrame, col: str, sex_col: str) -> np.ndarray:
    s = df[col].astype(float)
    z = df.groupby(sex_col)[col].transform(lambda g: (g - g.mean()) / g.std(ddof=1))
    return z.dropna().values

def main():
    ps.init_style()

    rcs_cmm = pd.read_csv(os.path.join(DATA_DIR, "fig5_rcs_cmm.csv"))
    rcs_cvd = pd.read_csv(os.path.join(DATA_DIR, "fig5_rcs_cvd.csv"))
    sub_df = pd.read_csv(os.path.join(DATA_DIR, "fig5_subgroups.csv"))
    sub_cmm = sub_df[sub_df["endpoint"] == "cmm"].copy()
    w6_df = pd.read_csv(os.path.join(DATA_DIR, "fig5_w6_extended.csv"))
    w6_m3 = w6_df[w6_df["tier"] == "Model3"].copy()

    # Load CHARLS clustered dataset for background Z-score density histograms
    cl_df = pd.read_csv(os.path.join(DERIVED_DIR, "charls_clustered.csv"))
    sex_col = next(c for c in ["female", "sex", "gender", "ragender"] if c in cl_df.columns)

    H = 166.0
    cv = ps.Canvas(ps.FIG_W_MM, H)

    # ROW 1: Panel A — 5-Panel Synchronized Horizontal 4-Knot RCS Strip
    cv.panel_label(1.5, 1.8, "A")
    cv.text(12.5, 3.6, "Dose-response splines for individual sex-standardized axes (Model 3)", size=ps.TEXT_PT, weight="bold")
    # Shared frameless legend top-right
    axA_leg = cv.raw_ax(121.0, 1.2, 60.5, 4.5)
    axA_leg.plot([0.0, 5.0], [2.2, 2.2], color=ps.ENDOTYPE_COLORS[4], lw=1.0)
    axA_leg.fill_between([0.0, 5.0], [1.1, 1.1], [3.3, 3.3], color=ps.ENDOTYPE_COLORS[4], alpha=0.16, lw=0)
    axA_leg.text(6.0, 2.2, "CMM (95% CI)", ha="left", va="center", size=ps.TEXT_PT)
    axA_leg.plot([34.0, 39.0], [2.2, 2.2], color=ps.COHORT_COLORS["CHARLS"], lw=0.8, ls="--")
    axA_leg.text(40.0, 2.2, "CVD (ref)", ha="left", va="center", size=ps.TEXT_PT)

    y_rcs = 8.2
    h_rcs = 27.5
    w_rcs = 29.8
    g_rcs = 4.2
    x0_rcs = 13.5

    for idx, axis_key in enumerate(ps.AXIS_ORDER):
        xx = x0_rcs + idx * (w_rcs + g_rcs)
        ax = cv.ax(xx, y_rcs, w_rcs, h_rcs)

        s_cmm = rcs_cmm[rcs_cmm["axis"] == axis_key].sort_values("z_value")
        s_cvd = rcs_cvd[rcs_cvd["axis"] == axis_key].sort_values("z_value")

        z_min, z_max = float(s_cmm["z_value"].min()), float(s_cmm["z_value"].max())
        z_vals = _within_sex_z(cl_df, axis_key, sex_col)
        z_vals = z_vals[(z_vals >= z_min) & (z_vals <= z_max)]

        # Subtle background population histogram along bottom 28% using twin axis
        ax_hist = ax.twinx()
        ax_hist.hist(z_vals, bins=24, density=True, color="#E4E7ED", edgecolor="white", linewidth=0.25, zorder=0)
        h_max = ax_hist.get_ylim()[1]
        ax_hist.set_ylim(0, h_max * 3.4)
        ax_hist.axis("off")
        ax.set_zorder(ax_hist.get_zorder() + 1)
        ax.patch.set_visible(False)

        ax.axhline(1.0, color="#888888", ls="--", lw=0.5, zorder=1)
        ax.plot(s_cvd["z_value"], s_cvd["hr"], color=ps.COHORT_COLORS["CHARLS"], lw=0.75, ls="--", alpha=0.85, zorder=2)
        ax.fill_between(s_cmm["z_value"], s_cmm["ci_low"], s_cmm["ci_high"], color=ps.ENDOTYPE_COLORS[4], alpha=0.14, lw=0, zorder=2)
        ax.plot(s_cmm["z_value"], s_cmm["hr"], color=ps.ENDOTYPE_COLORS[4], lw=0.95, zorder=3)

        z_ref = float(s_cmm["z_ref"].iloc[0])
        ax.plot([z_ref], [1.0], marker="o", ms=3.0, mfc="white", mec=ps.ENDOTYPE_COLORS[4], mew=0.65, zorder=4)

        p_ov = float(s_cmm["p_overall"].iloc[0])
        p_nl = float(s_cmm["p_nonlinear"].iloc[0])
        stat_str = f"CMM {ps.fmt_p(p_ov)}\nNonlin {ps.fmt_p(p_nl, with_p=False)}"
        ax.text(0.05, 0.95, stat_str, transform=ax.transAxes, ha="left", va="top", size=ps.TEXT_PT, linespacing=1.12, zorder=5)

        ax.set_xlim(-2.4, 2.7)
        ax.set_ylim(0.25, 2.08)
        xts = [-2, 0, 2]
        ax.set_xticks(xts)
        ax.set_xticklabels([ps.fmt_minus(str(v)) for v in xts])
        ax.set_yticks([0.5, 1.0, 1.5, 2.0])
        if idx == 0:
            ax.set_ylabel("Adjusted HR (95% CI)", size=ps.TEXT_PT, labelpad=2.0)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(f"{ps.AXIS_LABELS[axis_key]} (Z)", size=ps.TEXT_PT, labelpad=1.8)
        ps.style_l_axis(ax)

    # ROW 2: Panel B — CHARLS CMM Subgroup Consistency Forest Plot (11 Strata)
    cv.panel_label(1.5, 43.2, "B")
    axB_hdr = cv.raw_ax(1.5, 43.2, 180.0, 9.5)
    axB_hdr.text(7.5, 2.2, "CHARLS CMM subgroup stratification (Model 3, ref: E1)", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    for idx, ei in enumerate((2, 3, 4)):
        lx = 112.0 + idx * 14.5
        col = ps.ENDOTYPE_COLORS[ei]
        axB_hdr.plot([lx, lx + 2.6], [2.2, 2.2], color=col, lw=0.8)
        axB_hdr.plot([lx + 1.3], [2.2], marker="o", ms=3.2, mfc=col, mec="white", mew=0.35)
        axB_hdr.text(lx + 3.4, 2.2, f"E{ei}", ha="left", va="center", size=ps.TEXT_PT, weight="bold", color=col)

    axB_hdr.text(1.0, 6.8, "Subgroup / Stratum", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axB_hdr.text(122.0, 6.8, "E4 Events/N", ha="right", va="center", size=ps.TEXT_PT, weight="bold")
    axB_hdr.text(126.5, 6.8, "E4 vs E1 HR (95% CI)", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axB_hdr.text(178.5, 6.8, r"$\it{P}$" + " int", ha="right", va="center", size=ps.TEXT_PT, weight="bold")
    axB_hdr.plot([0.5, 179.0], [9.2, 9.2], color="#333333", lw=0.6)

    sub_rows = [
        ("Age", "<60 years", "Age: <60 yr"),
        ("Age", ">=60 years", "Age: \u226560 yr"),
        ("Sex", "Male", "Sex: Male"),
        ("Sex", "Female", "Sex: Female"),
        ("Residence", "Urban", "Residence: Urban"),
        ("Residence", "Rural", "Residence: Rural"),
        ("Baseline_CMD", "0 CMD", "Baseline: 0 CMD"),
        ("Baseline_CMD", "1 CMD", "Baseline: 1 CMD"),
        ("CKM_Proxy", "Stage 0-1", "CKM: Stage 0\u20131"),
        ("CKM_Proxy", "Stage 2", "CKM: Stage 2"),
        ("CKM_Proxy", "Stage 3", "CKM: Stage 3"),
    ]
    n_sb = len(sub_rows)
    y_sb = 52.8
    h_sb = 52.5
    row_h_sb = h_sb / n_sb

    axB_bg = cv.raw_ax(1.5, y_sb, 180.0, h_sb)
    axB_f = cv.ax(45.5, y_sb, 52.0, h_sb, facecolor="none")
    axB_f.set_xscale("log")
    axB_f.set_xlim(0.42, 7.5)
    axB_f.set_ylim(-0.5, n_sb - 0.5)
    axB_f.invert_yaxis()
    axB_f.axvline(1.0, color="#777777", ls="--", lw=0.55, zorder=1)
    xts_b = [0.5, 1, 2, 4, 7]
    axB_f.set_xticks(xts_b)
    axB_f.set_xticklabels([str(v) for v in xts_b])
    axB_f.set_yticks([])
    axB_f.spines["top"].set_visible(False)
    axB_f.spines["right"].set_visible(False)
    axB_f.spines["left"].set_visible(False)
    axB_f.spines["bottom"].set_linewidth(0.6)
    axB_f.tick_params(direction="out", length=2.2, width=0.6, pad=1.8, labelsize=ps.TEXT_PT)
    axB_f.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axB_f.set_xlabel("Adjusted HR vs E1 (95% CI, log scale)", size=ps.TEXT_PT, labelpad=1.6)

    # Zebra tint by subgroup family
    fam_idx = {"Age": 0, "Sex": 1, "Residence": 2, "Baseline_CMD": 3, "CKM_Proxy": 4}
    fam_spans = {"Age": (0, 2), "Sex": (2, 4), "Residence": (4, 6), "Baseline_CMD": (6, 8), "CKM_Proxy": (8, 11)}
    for fvar, (r0, r1) in fam_spans.items():
        if fam_idx[fvar] % 2 == 1:
            axB_bg.add_patch(Rectangle((0.5, r0 * row_h_sb), 178.5, (r1 - r0) * row_h_sb, facecolor=ps.BG_ZEBRA, edgecolor="none", zorder=0))
        p_int = float(sub_cmm[sub_cmm["subgroup_var"] == fvar]["p_interaction_any_high"].iloc[0])
        ym_fam = 0.5 * (r0 + r1) * row_h_sb
        axB_bg.text(
            178.5,
            ym_fam,
            ps.fmt_p(p_int, with_p=False),
            ha="right",
            va="center",
            size=ps.TEXT_PT,
            weight="bold" if p_int < 0.05 else "normal",
            color="#C84B31" if p_int < 0.05 else "#222222",
        )

    dodge = {2: -0.22, 3: 0.0, 4: 0.22}
    for ri, (svar, slvl, slbl) in enumerate(sub_rows):
        ym = (ri + 0.5) * row_h_sb
        axB_bg.text(1.0, ym, slbl, ha="left", va="center", size=ps.TEXT_PT, color="#111111")
        sub_row = sub_cmm[(sub_cmm["subgroup_var"] == svar) & (sub_cmm["subgroup_level"] == slvl)]
        r4 = sub_row[sub_row["endotype"] == 4].iloc[0]
        axB_bg.text(122.0, ym, f"{int(r4['events'])}/{int(r4['n']):,}", ha="right", va="center", size=ps.TEXT_PT, color="#222222")

        if svar == "CKM_Proxy" and slvl == "Stage 3":
            # Quasi-complete separation (E1: 0/3 events) — report honestly as NE
            axB_f.text(1.8, ri, "NE (0/3 E1 events)", ha="center", va="center", size=ps.TEXT_PT, color="#555555", bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=4)
            axB_bg.text(126.5, ym, "NE (0/3 ref events)", ha="left", va="center", size=ps.TEXT_PT, color="#555555")
            continue

        axB_bg.text(126.5, ym, ps.fmt_hr(float(r4["hr"]), float(r4["ci_low"]), float(r4["ci_high"])), ha="left", va="center", size=ps.TEXT_PT, color="#111111")
        for ei in (2, 3, 4):
            re_ = sub_row[sub_row["endotype"] == ei].iloc[0]
            col = ps.ENDOTYPE_COLORS[ei]
            yy = ri + dodge[ei]
            axB_f.plot([float(re_["ci_low"]), float(re_["ci_high"])], [yy, yy], color=col, lw=0.65, zorder=2)
            axB_f.plot([float(re_["hr"])], [yy], marker="o", ms=2.9, mfc=col, mec="white", mew=0.35, zorder=3)

    axB_bg.plot([0.5, 179.0], [h_sb, h_sb], color="#333333", lw=0.5)

    # ROW 3: Panel C — Wave 6 Exclusive Endpoints (Modified Poisson RR)
    cv.panel_label(1.5, 113.5, "C")
    axC_hdr = cv.raw_ax(1.5, 113.5, 180.0, 9.2)
    axC_hdr.text(7.5, 2.2, "CHARLS Wave 6 new neurovascular & specialist inpatient endpoints (Model 3, ref: E1)", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axC_hdr.text(1.0, 6.7, "Wave 6 Endpoint", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axC_hdr.text(115.0, 6.7, "E3 RR (95% CI)", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axC_hdr.text(142.5, 6.7, "E4 RR (95% CI)", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axC_hdr.text(178.5, 6.7, r"E4 $\it{P}$", ha="right", va="center", size=ps.TEXT_PT, weight="bold")
    axC_hdr.plot([0.5, 179.0], [9.0, 9.0], color="#333333", lw=0.6)

    w6_rows = [
        ("w6_dementia_atrophy", "New dementia / atrophy"),
        ("w6_dementia", "Incident dementia"),
        ("w6_atrophy", "Brain atrophy"),
        ("w6_cm_hosp", "CM inpatient admission"),
        ("w6_card_hosp", "Cardiology / cardiac surg."),
        ("w6_heart_attack", "Recurrent heart attack"),
    ]
    n_w6 = len(w6_rows)
    y_w6 = 123.0
    h_w6 = 34.5
    row_h_w6 = h_w6 / n_w6

    axC_bg = cv.raw_ax(1.5, y_w6, 180.0, h_w6)
    axC_f = cv.ax(45.5, y_w6, 65.0, h_w6, facecolor="none")
    axC_f.set_xscale("log")
    axC_f.set_xlim(0.32, 4.6)
    axC_f.set_ylim(-0.5, n_w6 - 0.5)
    axC_f.invert_yaxis()
    axC_f.axvline(1.0, color="#777777", ls="--", lw=0.55, zorder=1)
    xts_w6 = [0.5, 1, 2, 4]
    axC_f.set_xticks(xts_w6)
    axC_f.set_xticklabels([str(v) for v in xts_w6])
    axC_f.set_yticks([])
    axC_f.spines["top"].set_visible(False)
    axC_f.spines["right"].set_visible(False)
    axC_f.spines["left"].set_visible(False)
    axC_f.spines["bottom"].set_linewidth(0.6)
    axC_f.tick_params(direction="out", length=2.2, width=0.6, pad=1.8, labelsize=ps.TEXT_PT)
    axC_f.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axC_f.set_xlabel("Modified Poisson risk ratio vs E1 (95% CI, log scale)", size=ps.TEXT_PT, labelpad=1.6)

    for ri, (ep_k, ep_lbl) in enumerate(w6_rows):
        ym = (ri + 0.5) * row_h_w6
        if ri % 2 == 1:
            axC_bg.add_patch(Rectangle((0.5, ri * row_h_w6), 178.5, row_h_w6, facecolor=ps.BG_ZEBRA, edgecolor="none", zorder=0))
        axC_bg.text(1.0, ym, ep_lbl, ha="left", va="center", size=ps.TEXT_PT, color="#111111")

        sub_ep = w6_m3[w6_m3["endpoint"] == ep_k]
        for ei in (2, 3, 4):
            re_ = sub_ep[sub_ep["endotype"] == ei].iloc[0]
            col = ps.ENDOTYPE_COLORS[ei]
            yy = ri + dodge[ei]
            axC_f.plot([float(re_["ci_low"]), float(re_["ci_high"])], [yy, yy], color=col, lw=0.65, zorder=2)
            axC_f.plot([float(re_["rr"])], [yy], marker="s", ms=2.9, mfc=col, mec="white", mew=0.35, zorder=3)

        r3 = sub_ep[sub_ep["endotype"] == 3].iloc[0]
        r4 = sub_ep[sub_ep["endotype"] == 4].iloc[0]
        axC_bg.text(115.0, ym, ps.fmt_hr(float(r3["rr"]), float(r3["ci_low"]), float(r3["ci_high"])), ha="left", va="center", size=ps.TEXT_PT, color="#222222")
        axC_bg.text(142.5, ym, ps.fmt_hr(float(r4["rr"]), float(r4["ci_low"]), float(r4["ci_high"])), ha="left", va="center", size=ps.TEXT_PT, color="#111111")
        p4 = float(r4["p_value"])
        axC_bg.text(178.5, ym, ps.fmt_p(p4, with_p=False), ha="right", va="center", size=ps.TEXT_PT, weight="bold" if p4 < 0.05 else "normal", color="#C84B31" if p4 < 0.05 else "#444444")

    axC_bg.plot([0.5, 179.0], [h_w6, h_w6], color="#333333", lw=0.5)
    ps.save_fig(cv, FIG_DIR, "Figure5")

if __name__ == "__main__":
    main()
