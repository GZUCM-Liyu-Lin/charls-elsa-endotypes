from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(_Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(_Path(__file__).resolve().parent))

from config import DERIVED_DIR, FIG_DATA_DIR as DATA_DIR, FIGURES_DIR as FIG_DIR, TABLES_DIR as TAB_DIR
import plot_style as ps

def _draw_alluvial(ax, df_sk, x0, x1, y_top, h_tot, bar_w=3.2, gap_mm=1.6, title=""):
    """Draw 4-endotype cubic-Bezier alluvial diagram in raw_ax (y downward)."""
    ax.text(0.5 * (x0 + x1) + bar_w / 2.0, y_top - 3.2, title, ha="center", va="center", size=ps.TEXT_PT, weight="bold", color="#111111")

    N_tot = float(df_sk["n"].sum())
    avail_h = h_tot - 3 * gap_mm

    from_tots = df_sk.groupby("from_endotype")["n"].sum().reindex([1, 2, 3, 4], fill_value=0)
    to_tots = df_sk.groupby("to_endotype")["n"].sum().reindex([1, 2, 3, 4], fill_value=0)

    # Compute y-intervals [y_start, y_end] for each endotype on left and right bars
    def _layout_bars(tots):
        intervals = {}
        yy = y_top
        for ei in (1, 2, 3, 4):
            bh = avail_h * (float(tots[ei]) / N_tot) if N_tot > 0 else 0.0
            intervals[ei] = (yy, yy + bh)
            yy += bh + gap_mm
        return intervals

    left_pos = _layout_bars(from_tots)
    right_pos = _layout_bars(to_tots)

    # Track running cursor inside each bar for ribbon stacking
    left_cur = {ei: left_pos[ei][0] for ei in (1, 2, 3, 4)}
    right_cur = {ei: right_pos[ei][0] for ei in (1, 2, 3, 4)}

    # Draw ribbons
    xm = 0.5 * (x0 + bar_w + x1)
    for fi in (1, 2, 3, 4):
        col = ps.ENDOTYPE_COLORS[fi]
        for ti in (1, 2, 3, 4):
            sub = df_sk[(df_sk["from_endotype"] == fi) & (df_sk["to_endotype"] == ti)]
            n_flow = float(sub["n"].iloc[0]) if len(sub) > 0 else 0.0
            if n_flow <= 0:
                continue
            fh = avail_h * (n_flow / N_tot)
            y0a, y0b = left_cur[fi], left_cur[fi] + fh
            y1a, y1b = right_cur[ti], right_cur[ti] + fh
            left_cur[fi] = y0b
            right_cur[ti] = y1b

            verts = [
                (x0 + bar_w, y0a),
                (xm, y0a),
                (xm, y1a),
                (x1, y1a),
                (x1, y1b),
                (xm, y1b),
                (xm, y0b),
                (x0 + bar_w, y0b),
                (x0 + bar_w, y0a),
            ]
            codes = [
                Path.MOVETO,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.LINETO,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CLOSEPOLY,
            ]
            patch = PathPatch(Path(verts, codes), facecolor=col, edgecolor="none", alpha=0.38, zorder=1)
            ax.add_patch(patch)

    # Draw node bars + clean outside percentages
    for ei in (1, 2, 3, 4):
        col = ps.ENDOTYPE_COLORS[ei]
        ly0, ly1 = left_pos[ei]
        if ly1 - ly0 > 0.15:
            ax.add_patch(Rectangle((x0, ly0), bar_w, max(ly1 - ly0, 0.35), facecolor=col, edgecolor="white", lw=0.4, zorder=3))
        pct_l = float(from_tots[ei]) / N_tot * 100.0
        if pct_l >= 2.5:
            ax.text(x0 - 0.8, 0.5 * (ly0 + ly1), f"E{ei} {pct_l:.0f}%", ha="right", va="center", size=ps.TEXT_PT, color="#111111", zorder=4)

        ry0, ry1 = right_pos[ei]
        if ry1 - ry0 > 0.15:
            ax.add_patch(Rectangle((x1, ry0), bar_w, max(ry1 - ry0, 0.35), facecolor=col, edgecolor="white", lw=0.4, zorder=3))
        pct_r = float(to_tots[ei]) / N_tot * 100.0
        if pct_r >= 2.5:
            ax.text(x1 + bar_w + 0.8, 0.5 * (ry0 + ry1), f"E{ei} {pct_r:.0f}%", ha="left", va="center", size=ps.TEXT_PT, color="#111111", zorder=4)

def main():
    ps.init_style()

    ms_df = pd.read_csv(os.path.join(DATA_DIR, "fig4_multistate_hrs.csv"))
    ms_m3 = ms_df[ms_df["tier"] == "Model3"].copy()
    sk_ch = pd.read_csv(os.path.join(DATA_DIR, "fig4_sankey_charls.csv"))
    sk_el = pd.read_csv(os.path.join(DATA_DIR, "fig4_sankey_elsa.csv"))
    tr_df = pd.read_csv(os.path.join(DATA_DIR, "fig4_transition_hrs.csv"))
    tr_m3 = tr_df[tr_df["tier"] == "Model3"].copy()

    H = 152.0
    cv = ps.Canvas(ps.FIG_W_MM, H)

    # ROW 1: Panel A (Multi-state topology + HRs) & Panel B (Alluvial Sankey)
    cv.panel_label(1.5, 1.8, "A")

    # Top Multi-State Schematic inside Panel A
    axA_top = cv.raw_ax(2.0, 2.0, 93.0, 23.0)
    axA_top.text(47.5, 1.8, "Multi-state progression hazards (Model 3)", ha="center", va="center", size=ps.TEXT_PT, weight="bold")

    # 4 crisp state nodes
    ps.draw_rbox(axA_top, 3.0, 4.6, 22.0, 7.2, "CMD-Free", fc="#EDF3F8", ec=ps.COHORT_COLORS["CHARLS"], lw=0.6, weight="bold", rounding=0.7)
    ps.draw_rbox(axA_top, 35.5, 4.6, 22.0, 7.2, "First CMD", fc="#FDF6EB", ec=ps.ENDOTYPE_COLORS[3], lw=0.6, weight="bold", rounding=0.7)
    ps.draw_rbox(axA_top, 68.0, 4.6, 22.0, 7.2, "CMM", fc="#FBEFEE", ec=ps.ENDOTYPE_COLORS[4], lw=0.6, weight="bold", rounding=0.7)
    ps.draw_rbox(axA_top, 31.5, 16.0, 30.0, 6.2, "All-cause death", fc="#F1F2F4", ec="#555555", lw=0.6, weight="bold", rounding=0.7)

    # Transitions T1-T5
    ps.draw_arrow_mm(axA_top, 25.2, 8.2, 35.2, 8.2, color="#333333", lw=0.65, ms=4.8)
    axA_top.text(30.2, 6.4, "T1", ha="center", va="center", size=ps.TEXT_PT, weight="bold", color="#222222")
    ps.draw_arrow_mm(axA_top, 57.7, 8.2, 67.7, 8.2, color="#333333", lw=0.65, ms=4.8)
    axA_top.text(62.7, 6.4, "T3", ha="center", va="center", size=ps.TEXT_PT, weight="bold", color="#222222")

    ps.draw_arrow_mm(axA_top, 14.0, 12.0, 31.2, 18.8, color="#555555", lw=0.6, ms=4.5)
    axA_top.text(20.0, 16.8, "T2", ha="center", va="center", size=ps.TEXT_PT, weight="bold", color="#444444")
    ps.draw_arrow_mm(axA_top, 46.5, 12.0, 46.5, 15.8, color="#555555", lw=0.6, ms=4.5)
    axA_top.text(49.5, 13.9, "T4", ha="left", va="center", size=ps.TEXT_PT, weight="bold", color="#444444")
    ps.draw_arrow_mm(axA_top, 79.0, 12.0, 61.8, 18.8, color="#555555", lw=0.6, ms=4.5)
    axA_top.text(73.0, 16.8, "T5", ha="center", va="center", size=ps.TEXT_PT, weight="bold", color="#444444")

    # Multi-state legend cleanly above bottom plots
    axA_leg = cv.raw_ax(23.5, 25.4, 71.0, 4.4)
    for idx, ei in enumerate((2, 3, 4)):
        lx = idx * 23.5
        col = ps.ENDOTYPE_COLORS[ei]
        axA_leg.plot([lx, lx + 2.6], [2.2, 2.2], color=col, lw=0.9)
        axA_leg.plot([lx + 1.3], [2.2], marker="o", ms=3.4, mfc=col, mec="white", mew=0.4)
        axA_leg.text(lx + 3.2, 2.2, ps.ENDOTYPE_LABELS[ei], ha="left", va="center", size=ps.TEXT_PT)

    # Bottom dual-cohort dot-whisker HR plots
    y_ms = 32.8
    h_ms = 34.5
    trans_list = [
        ("T1_Free_to_FirstCMD", "T1: Free \u2192 1st CMD"),
        ("T2_Free_to_Death", "T2: Free \u2192 Death"),
        ("T3_FirstCMD_to_CMM_clockreset", "T3: 1st CMD \u2192 CMM"),
        ("T4_FirstCMD_to_Death_clockreset", "T4: 1st CMD \u2192 Death"),
        ("T5_PrevalentCMM_to_Death", "T5: CMM \u2192 Death"),
    ]
    axA_bg = cv.raw_ax(1.5, y_ms, 93.5, h_ms)
    axA_ch = cv.ax(26.5, y_ms, 31.5, h_ms, facecolor="none")
    axA_el = cv.ax(61.5, y_ms, 31.5, h_ms, facecolor="none")

    for axM, coh_title, coh_col in [
        (axA_ch, "CHARLS", ps.COHORT_COLORS["CHARLS"]),
        (axA_el, "ELSA", ps.COHORT_COLORS["ELSA"]),
    ]:
        axM.set_xscale("log")
        axM.set_xlim(0.10, 9.0)
        axM.set_ylim(-0.5, 4.5)
        axM.invert_yaxis()
        axM.axvline(1.0, color="#777777", ls="--", lw=0.55, zorder=1)
        xts = [0.2, 0.5, 1, 2, 5]
        axM.set_xticks(xts)
        axM.set_xticklabels([str(v) for v in xts])
        axM.set_yticks([])
        axM.spines["top"].set_visible(False)
        axM.spines["right"].set_visible(False)
        axM.spines["left"].set_visible(False)
        axM.spines["bottom"].set_linewidth(0.6)
        axM.tick_params(direction="out", length=2.0, width=0.6, pad=1.6, labelsize=ps.TEXT_PT)
        axM.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        axM.set_title(coh_title, size=ps.TEXT_PT, weight="bold", color=coh_col, pad=2.0)
        axM.set_xlabel("CS-HR vs E1 (95% CI)", size=ps.TEXT_PT, labelpad=1.6)

    row_h_ms = h_ms / 5.0
    dodge = {2: -0.22, 3: 0.0, 4: 0.22}
    for ri, (t_key, t_lbl) in enumerate(trans_list):
        if ri % 2 == 1:
            axA_bg.add_patch(Rectangle((0.5, ri * row_h_ms), 92.5, row_h_ms, facecolor=ps.BG_ZEBRA, edgecolor="none", zorder=0))
        axA_bg.text(0.8, (ri + 0.5) * row_h_ms, t_lbl, ha="left", va="center", size=ps.TEXT_PT, color="#111111")

        for axM, coh in [(axA_ch, "CHARLS"), (axA_el, "ELSA")]:
            sub_t = ms_m3[(ms_m3["cohort"] == coh) & (ms_m3["transition"] == t_key)]
            if len(sub_t) == 0:
                bg_row = ps.BG_ZEBRA if ri % 2 == 1 else "white"
                axM.text(
                    1.0,
                    ri,
                    "NE (sparse)",
                    ha="center",
                    va="center",
                    size=ps.TEXT_PT,
                    color="#555555",
                    bbox=dict(facecolor=bg_row, edgecolor="none", pad=0.7),
                    zorder=4,
                )
                continue
            for ei in (2, 3, 4):
                r_e = sub_t[sub_t["endotype"] == ei]
                if len(r_e) == 0 or pd.isna(r_e["hr"].iloc[0]):
                    continue
                row = r_e.iloc[0]
                col = ps.ENDOTYPE_COLORS[ei]
                yy = ri + dodge[ei]
                axM.plot([float(row["ci_low"]), float(row["ci_high"])], [yy, yy], color=col, lw=0.65, zorder=2)
                axM.plot([float(row["hr"])], [yy], marker="o", ms=3.0, mfc=col, mec="white", mew=0.35, zorder=3)

    # Panel B: Paired 4-Year Endotype Alluvial Diagrams
    cv.panel_label(96.5, 1.8, "B")
    axB = cv.raw_ax(97.0, 2.0, 84.5, 72.5)
    axB.text(42.2, 1.8, "Four-year endotype transitions", ha="center", va="center", size=ps.TEXT_PT, weight="bold")

    _draw_alluvial(axB, sk_ch, x0=11.0, x1=28.5, y_top=10.5, h_tot=52.5, bar_w=3.0, gap_mm=1.6, title="CHARLS W1\u2192W3 (n = 4,179)")
    _draw_alluvial(axB, sk_el, x0=52.5, x1=70.0, y_top=10.5, h_tot=52.5, bar_w=3.0, gap_mm=1.6, title="ELSA W2\u2192W4 (n = 935)")

    # Column labels below bars + sparse ELSA note
    axB.text(12.5, 65.5, "W1", ha="center", va="center", size=ps.TEXT_PT, color="#444444")
    axB.text(30.0, 65.5, "W3", ha="center", va="center", size=ps.TEXT_PT, color="#444444")
    axB.text(54.0, 65.5, "W2", ha="center", va="center", size=ps.TEXT_PT, color="#444444")
    axB.text(71.5, 65.5, "W4", ha="center", va="center", size=ps.TEXT_PT, color="#444444")
    axB.text(62.8, 69.8, "ELSA W2: E3 n = 1; E4 n = 2", ha="center", va="center", size=ps.TEXT_PT, color="#555555")

    # ROW 2: Panel C — 4-Year Transition Category Risks (Full 183 mm Width)
    cv.panel_label(1.5, 77.2, "C")
    axC_hdr = cv.raw_ax(1.5, 77.2, 180.0, 9.5)
    axC_hdr.text(7.5, 2.2, "Four-year endotype transition categories & incident CMM / CVD (Model 3, ref: Stable Low)", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axC_hdr.plot([138.0], [2.2], marker="s", ms=3.3, mfc="#333333", mec="#333333", mew=0)
    axC_hdr.text(140.0, 2.2, "CS-HR", ha="left", va="center", size=ps.TEXT_PT)
    axC_hdr.plot([158.0], [2.2], marker="o", ms=3.3, mfc="white", mec="#333333", mew=0.7)
    axC_hdr.text(160.0, 2.2, "FG-SHR", ha="left", va="center", size=ps.TEXT_PT)

    axC_hdr.text(1.0, 6.8, "Cohort / Outcome", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axC_hdr.text(30.5, 6.8, "Transition group", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axC_hdr.text(68.5, 6.8, "Events/N", ha="right", va="center", size=ps.TEXT_PT, weight="bold")
    axC_hdr.text(124.0, 6.8, "CS-HR (95% CI)", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axC_hdr.text(153.5, 6.8, "FG-SHR (95% CI)", ha="left", va="center", size=ps.TEXT_PT, weight="bold")
    axC_hdr.plot([0.5, 179.0], [9.2, 9.2], color="#333333", lw=0.6)

    y_tbl_c = 86.8
    h_tbl_c = 55.5
    n_rows_c = 12
    row_h_c = h_tbl_c / n_rows_c

    axC_bg = cv.raw_ax(1.5, y_tbl_c, 180.0, h_tbl_c)
    axC_f = cv.ax(72.5, y_tbl_c, 48.0, h_tbl_c, facecolor="none")
    axC_f.set_xscale("log")
    axC_f.set_xlim(0.20, 4.6)
    axC_f.set_ylim(-0.5, n_rows_c - 0.5)
    axC_f.invert_yaxis()
    axC_f.axvline(1.0, color="#777777", ls="--", lw=0.55, zorder=1)
    xts_c = [0.25, 0.5, 1, 2, 4]
    axC_f.set_xticks(xts_c)
    axC_f.set_xticklabels([str(v) for v in xts_c])
    axC_f.set_yticks([])
    axC_f.spines["top"].set_visible(False)
    axC_f.spines["right"].set_visible(False)
    axC_f.spines["left"].set_visible(False)
    axC_f.spines["bottom"].set_linewidth(0.6)
    axC_f.tick_params(direction="out", length=2.2, width=0.6, pad=1.8, labelsize=ps.TEXT_PT)
    axC_f.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axC_f.set_xlabel("Adjusted hazard ratio vs Stable Low-Risk (95% CI, log scale)", size=ps.TEXT_PT, labelpad=1.8)

    blocks = [
        ("CHARLS", "cmm", "CHARLS \u2014 CMM"),
        ("CHARLS", "cvd", "CHARLS \u2014 CVD"),
        ("ELSA", "cmm", "ELSA \u2014 CMM"),
        ("ELSA", "cvd", "ELSA \u2014 CVD"),
    ]
    trans_rows = [
        ("2_Favorable_Transition", "Favorable transition", ps.ENDOTYPE_COLORS[2]),
        ("3_Adverse_Transition", "Adverse transition", ps.ENDOTYPE_COLORS[3]),
        ("4_Stable_High", "Stable high-risk", ps.ENDOTYPE_COLORS[4]),
    ]

    for blk_i, (coh, ep, blk_lbl) in enumerate(blocks):
        r0 = blk_i * 3
        if blk_i % 2 == 1:
            axC_bg.add_patch(Rectangle((0.5, r0 * row_h_c), 178.5, 3 * row_h_c, facecolor=ps.BG_ZEBRA, edgecolor="none", zorder=0))
        axC_bg.text(1.0, (r0 + 1.5) * row_h_c, blk_lbl, ha="left", va="center", size=ps.TEXT_PT, weight="bold", color=ps.COHORT_COLORS[coh])

        for off_i, (tg_key, tg_lbl, tcol) in enumerate(trans_rows):
            ri = r0 + off_i
            ym = (ri + 0.5) * row_h_c
            sub = tr_m3[(tr_m3["cohort"] == coh) & (tr_m3["endpoint"] == ep) & (tr_m3["trans_group"] == tg_key)].iloc[0]

            axC_bg.text(30.5, ym, tg_lbl, ha="left", va="center", size=ps.TEXT_PT, color=tcol, weight="bold" if tg_key == "4_Stable_High" else "normal")
            axC_bg.text(68.5, ym, f"{int(sub['events'])}/{int(sub['n']):,}", ha="right", va="center", size=ps.TEXT_PT, color="#222222")

            if pd.isna(sub["cs_hr"]):
                # sparse zero-event masking: NE (0 events)
                bg_c = ps.BG_ZEBRA if blk_i % 2 == 1 else "white"
                axC_f.text(1.0, ri, "NE (0 events)", ha="center", va="center", size=ps.TEXT_PT, color="#555555", bbox=dict(facecolor=bg_c, edgecolor="none", pad=0.7), zorder=4)
                axC_bg.text(124.0, ym, "NE (0 events)", ha="left", va="center", size=ps.TEXT_PT, color="#555555")
                axC_bg.text(153.5, ym, "NE (0 events)", ha="left", va="center", size=ps.TEXT_PT, color="#555555")
            else:
                cs_hr, cs_lo, cs_hi = float(sub["cs_hr"]), float(sub["cs_ci_low"]), float(sub["cs_ci_high"])
                fg_hr, fg_lo, fg_hi = float(sub["fg_shr"]), float(sub["fg_ci_low"]), float(sub["fg_ci_high"])
                axC_f.plot([cs_lo, cs_hi], [ri - 0.16, ri - 0.16], color=tcol, lw=0.65, zorder=2)
                axC_f.plot([cs_hr], [ri - 0.16], marker="s", ms=3.1, mfc=tcol, mec="white", mew=0.35, zorder=3)
                axC_f.plot([fg_lo, fg_hi], [ri + 0.16, ri + 0.16], color=tcol, lw=0.55, alpha=0.80, zorder=2)
                axC_f.plot([fg_hr], [ri + 0.16], marker="o", ms=3.0, mfc="white", mec=tcol, mew=0.65, zorder=3)

                axC_bg.text(124.0, ym, ps.fmt_hr(cs_hr, cs_lo, cs_hi), ha="left", va="center", size=ps.TEXT_PT, color="#111111")
                axC_bg.text(153.5, ym, ps.fmt_hr(fg_hr, fg_lo, fg_hi), ha="left", va="center", size=ps.TEXT_PT, color="#444444")

    axC_bg.plot([0.5, 179.0], [h_tbl_c, h_tbl_c], color="#333333", lw=0.5)
    ps.save_fig(cv, FIG_DIR, "Figure4")

if __name__ == "__main__":
    main()
