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

def _at_risk_at_ticks(sub_df: pd.DataFrame, tick_times):
    res = []
    sub_s = sub_df.sort_values("time_yr")
    for t in tick_times:
        cand = sub_s[sub_s["time_yr"] <= t + 1e-6]
        if len(cand) > 0:
            res.append(int(cand["n_at_risk"].iloc[-1]))
        else:
            res.append(int(sub_s["n_at_risk"].iloc[0]))
    return res

def _plot_cif_panel(ax, ax_tbl, df_all, endpoint, title, xmax, ymax, xticks, yticks, show_ylabel=True, show_row_headers=False):
    sub_ep = df_all[df_all["endpoint"] == endpoint]
    gray_p = float(sub_ep["gray_type_min_p"].dropna().iloc[0])

    for ei in (1, 2, 3, 4):
        col = ps.ENDOTYPE_COLORS[ei]
        s_cd = sub_ep[(sub_ep["endotype"] == ei) & (sub_ep["event_type"] == "competing_death")].sort_values("time_yr")
        if len(s_cd) > 0:
            ax.step(s_cd["time_yr"], s_cd["cif"] * 100.0, where="post", color=col, lw=0.50, ls="--", alpha=0.65, zorder=2)

        s_ev = sub_ep[(sub_ep["endotype"] == ei) & (sub_ep["event_type"] == "target_event")].sort_values("time_yr")
        if len(s_ev) > 0:
            t = s_ev["time_yr"].values
            y = s_ev["cif"].values * 100.0
            lo = s_ev["ci_low"].values * 100.0
            hi = s_ev["ci_high"].values * 100.0
            ax.fill_between(t, lo, hi, step="post", color=col, alpha=0.11, lw=0, zorder=1)
            ax.step(t, y, where="post", color=col, lw=0.95, zorder=3)

        counts = _at_risk_at_ticks(s_ev, xticks)
        ry = 3.2 + (ei - 1) * 2.55
        if show_row_headers:
            # Place E1..E4 left of the leftmost 0-yr count (t=0 count spans ~[-3.8, +3.8] mm)
            ax_tbl.text(-6.0, ry, f"E{ei}", ha="right", va="center", size=ps.TEXT_PT, weight="bold", color=col, clip_on=False)
        for xt, cnt in zip(xticks, counts):
            x_frac = xt / float(xmax)
            ax_tbl.text(x_frac * ax_tbl.get_xlim()[1], ry, f"{cnt:,}", ha="center", va="center", size=ps.TEXT_PT, color="#222222", clip_on=False)

    if show_row_headers:
        ax_tbl.text(-6.0, 0.5, "At risk", ha="right", va="center", size=ps.TEXT_PT, weight="bold", color="#222222", clip_on=False)

    ax.text(0.04, 0.94, f"Gray-type {ps.fmt_p(gray_p)}", transform=ax.transAxes, ha="left", va="top", size=ps.TEXT_PT)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax * 100.0)
    ax.set_xticks(xticks)
    ax.set_yticks([v * 100.0 for v in yticks])
    ax.set_xlabel("Follow-up (years)", size=ps.TEXT_PT, labelpad=1.6)
    if show_ylabel:
        ax.set_ylabel("Cumulative incidence (%)", size=ps.TEXT_PT, labelpad=2.0)
    ax.set_title(title, size=ps.TEXT_PT, weight="bold", pad=3.0)
    ps.style_l_axis(ax)

def main():
    ps.init_style()

    cif_ch = pd.read_csv(os.path.join(DATA_DIR, "fig3_cif_charls.csv"))
    cif_el = pd.read_csv(os.path.join(DATA_DIR, "fig3_cif_elsa.csv"))
    f3 = pd.read_csv(os.path.join(DATA_DIR, "fig3_forest_endotypes.csv"))
    f3_m3 = f3[f3["tier"] == "Model3"].copy()

    H = 160.0
    cv = ps.Canvas(ps.FIG_W_MM, H)

    # TOP SHARED LEGEND (y = 1.0 .. 5.8 mm) — fits cleanly within [2, 180] mm
    ax_leg = cv.raw_ax(2.0, 1.0, 179.0, 4.8)
    x_cur = 2.0
    for ei in (1, 2, 3, 4):
        col = ps.ENDOTYPE_COLORS[ei]
        ax_leg.plot([x_cur, x_cur + 3.8], [2.4, 2.4], color=col, lw=1.2)
        ax_leg.text(x_cur + 4.6, 2.4, ps.ENDOTYPE_LABELS[ei], ha="left", va="center", size=ps.TEXT_PT, color="#111111")
        x_cur += 27.5
    ax_leg.plot([x_cur + 1.0, x_cur + 5.2], [2.4, 2.4], color="#333333", lw=1.0, ls="-")
    ax_leg.text(x_cur + 6.0, 2.4, "Event (solid)", ha="left", va="center", size=ps.TEXT_PT, color="#333333")
    ax_leg.plot([x_cur + 28.0, x_cur + 32.2], [2.4, 2.4], color="#333333", lw=0.6, ls="--")
    ax_leg.text(x_cur + 33.0, 2.4, "Death (dashed)", ha="left", va="center", size=ps.TEXT_PT, color="#333333")

    # ROW 1: Panels A, B, C (CIF + Aligned Risk Tables)
    y_lbl1 = 6.8
    y_ax1 = 12.6
    h_ax1 = 33.0
    w_ax1 = 46.5
    y_tbl = 52.8
    h_tbl = 13.0

    xs_r1 = [15.0, 73.5, 132.0]
    configs = [
        ("A", cif_ch, "cmm", "CHARLS: Incident CMM", 7.0, 0.28, [0, 2, 4, 6], [0, 0.08, 0.16, 0.24], True, True),
        ("B", cif_ch, "cvd", "CHARLS: Incident CVD", 7.0, 0.36, [0, 2, 4, 6], [0, 0.10, 0.20, 0.30], False, False),
        ("C", cif_el, "cmm", "ELSA: Incident CMM", 14.0, 0.25, [0, 4, 8, 12], [0, 0.08, 0.16, 0.24], False, False),
    ]
    for idx, (ltr, df_c, ep, ttl, xm, ym, xts, yts, syd, srh) in enumerate(configs):
        xx = xs_r1[idx]
        cv.panel_label(1.5 if idx == 0 else xx - 8.5, y_lbl1, ltr)
        ax_c = cv.ax(xx, y_ax1, w_ax1, h_ax1)
        ax_t = cv.raw_ax(xx, y_tbl, w_ax1, h_tbl)
        _plot_cif_panel(ax_c, ax_t, df_c, ep, ttl, xm, ym, xts, yts, show_ylabel=syd, show_row_headers=srh)

    # ROW 2: Panel D — Dual-Cohort, Dual-Model (CS-HR & FG-SHR) Forest Plot
    cv.panel_label(1.5, 68.5, "D")
    axD_hdr = cv.raw_ax(1.5, 68.5, 180.0, 12.5)
    axD_hdr.text(10.0, 2.5, "Model 3 Adjusted Hazard Ratios (ref: E1)", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axD_hdr.plot([122.0], [2.5], marker="s", ms=3.4, mfc="#333333", mec="#333333", mew=0)
    axD_hdr.text(124.2, 2.5, "CS-HR (95% CI)", ha="left", va="center", size=ps.TEXT_PT)
    axD_hdr.plot([153.0], [2.5], marker="o", ms=3.4, mfc="white", mec="#333333", mew=0.75)
    axD_hdr.text(155.2, 2.5, "FG-SHR (95% CI)", ha="left", va="center", size=ps.TEXT_PT)

    # Cohort headings centered directly over CHARLS & ELSA forest+table blocks
    axD_hdr.text(46.0, 7.2, "CHARLS (Discovery)", ha="center", va="center", size=ps.TEXT_PT, weight="bold", color=ps.COHORT_COLORS["CHARLS"])
    axD_hdr.text(122.0, 7.2, "ELSA (Validation)", ha="center", va="center", size=ps.TEXT_PT, weight="bold", color=ps.COHORT_COLORS["ELSA"])

    # Column subheaders strictly aligned with coordinates below
    # Full x inside axD_hdr / axD_bg is 0..180 mm (page x = 1.5..181.5 mm)
    axD_hdr.text(1.0, 10.8, "Outcome / Group", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axD_hdr.text(65.0, 10.8, "CS-HR (95% CI)", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axD_hdr.text(98.0, 10.8, "FG-SHR", ha="right", va="center", size=ps.TEXT_PT, weight="bold")
    axD_hdr.text(139.0, 10.8, "CS-HR (95% CI)", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axD_hdr.text(178.5, 10.8, "FG-SHR", ha="right", va="center", size=ps.TEXT_PT, weight="bold")
    axD_hdr.plot([0.5, 179.0], [12.3, 12.3], color="#333333", lw=0.6)

    endpoints_order = [
        ("cmm", "CMM"),
        ("cvd", "CVD"),
        ("heart", "Heart dis."),
        ("stroke", "Stroke"),
        ("diabetes", "Diabetes"),
        ("death", "All death"),
    ]

    y_tbl_d = 81.0
    h_tbl_d = 71.0

    axD_bg = cv.raw_ax(1.5, y_tbl_d, 180.0, h_tbl_d)
    n_rows = 18

    axF_ch = cv.ax(30.0, y_tbl_d, 34.0, h_tbl_d, facecolor="none")
    axF_el = cv.ax(104.0, y_tbl_d, 34.0, h_tbl_d, facecolor="none")

    for axF, xm_lo, xm_hi, xts in [
        (axF_ch, 0.65, 6.2, [0.7, 1, 2, 4]),
        (axF_el, 0.25, 22.0, [0.5, 1, 2, 5, 15]),
    ]:
        axF.set_xscale("log")
        axF.set_xlim(xm_lo, xm_hi)
        axF.set_ylim(-0.5, n_rows - 0.5)
        axF.invert_yaxis()
        axF.axvline(1.0, color="#777777", ls="--", lw=0.55, zorder=1)
        axF.set_xticks(xts)
        axF.set_xticklabels([str(x) for x in xts])
        axF.set_yticks([])
        axF.spines["top"].set_visible(False)
        axF.spines["right"].set_visible(False)
        axF.spines["left"].set_visible(False)
        axF.spines["bottom"].set_linewidth(0.6)
        axF.tick_params(direction="out", length=2.2, width=0.6, pad=1.8, labelsize=ps.TEXT_PT)
        axF.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        axF.set_xlabel("Hazard ratio (95% CI, log scale)", size=ps.TEXT_PT, labelpad=1.8)

    row_h_mm = h_tbl_d / n_rows
    for ep_idx, (ep_key, ep_label) in enumerate(endpoints_order):
        r_start = ep_idx * 3
        y_top_blk = r_start * row_h_mm
        if ep_idx % 2 == 1:
            axD_bg.add_patch(Rectangle((0.5, y_top_blk), 178.5, 3 * row_h_mm, facecolor=ps.BG_ZEBRA, edgecolor="none", zorder=0))

        axD_bg.text(1.0, y_top_blk + 1.5 * row_h_mm, ep_label, ha="left", va="center", size=ps.TEXT_PT, weight="bold", color="#111111")

        for off_idx, ei in enumerate((2, 3, 4)):
            row_i = r_start + off_idx
            ym_row = (row_i + 0.5) * row_h_mm
            col = ps.ENDOTYPE_COLORS[ei]
            axD_bg.text(18.5, ym_row, f"E{ei}", ha="left", va="center", size=ps.TEXT_PT, weight="bold", color=col)

            rc = f3_m3[(f3_m3["cohort"] == "CHARLS") & (f3_m3["endpoint"] == ep_key) & (f3_m3["endotype"] == ei)].iloc[0]
            re_ = f3_m3[(f3_m3["cohort"] == "ELSA") & (f3_m3["endpoint"] == ep_key) & (f3_m3["endotype"] == ei)].iloc[0]

            for axF, rr, tx_cs, tx_fg in [
                (axF_ch, rc, 65.0, 98.0),
                (axF_el, re_, 139.0, 178.5),
            ]:
                cs_hr, cs_lo, cs_hi = float(rr["cs_hr"]), float(rr["cs_ci_low"]), float(rr["cs_ci_high"])
                fg_shr, fg_lo, fg_hi = rr["fg_shr"], rr["fg_ci_low"], rr["fg_ci_high"]

                dy_cs = -0.17 if pd.notna(fg_shr) else 0.0
                axF.plot([cs_lo, cs_hi], [row_i + dy_cs, row_i + dy_cs], color=col, lw=0.65, zorder=2)
                axF.plot([cs_hr], [row_i + dy_cs], marker="s", ms=3.0, mfc=col, mec="white", mew=0.3, zorder=3)
                axD_bg.text(tx_cs, ym_row, ps.fmt_hr(cs_hr, cs_lo, cs_hi), ha="left", va="center", size=ps.TEXT_PT, color="#111111")

                if pd.notna(fg_shr):
                    axF.plot([float(fg_lo), float(fg_hi)], [row_i + 0.17, row_i + 0.17], color=col, lw=0.55, alpha=0.80, zorder=2)
                    axF.plot([float(fg_shr)], [row_i + 0.17], marker="o", ms=2.9, mfc="white", mec=col, mew=0.65, zorder=3)
                    axD_bg.text(tx_fg, ym_row, f"{float(fg_shr):.2f}", ha="right", va="center", size=ps.TEXT_PT, color="#444444")
                else:
                    axD_bg.text(tx_fg, ym_row, "\u2014", ha="right", va="center", size=ps.TEXT_PT, color="#777777")

    axD_bg.plot([0.5, 179.0], [h_tbl_d, h_tbl_d], color="#333333", lw=0.5)
    ps.save_fig(cv, FIG_DIR, "Figure3")

if __name__ == "__main__":
    main()
