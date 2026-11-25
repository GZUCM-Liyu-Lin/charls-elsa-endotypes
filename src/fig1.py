from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from scipy.stats import pearsonr

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DERIVED_DIR, FIG_DATA_DIR as DATA_DIR, FIGURES_DIR as FIG_DIR, TABLES_DIR as TAB_DIR
import plot_style as ps

def _build_corr_matrix(df_cohort: pd.DataFrame):
    n = len(ps.AXIS_ORDER)
    rmat = np.zeros((n, n), dtype=float)
    pmat = np.ones((n, n), dtype=float)
    lookup = {
        (row["axis_1"], row["axis_2"]): (float(row["spearman_r"]), float(row["p_value"]))
        for _, row in df_cohort.iterrows()
    }
    for i, a1 in enumerate(ps.AXIS_ORDER):
        for j, a2 in enumerate(ps.AXIS_ORDER):
            rmat[i, j], pmat[i, j] = lookup[(a1, a2)]
    return rmat, pmat

def _draw_corr_heatmap(ax, rmat, pmat, title, show_ylabels=True):
    n = len(ps.AXIS_ORDER)
    masked = np.ma.masked_where(np.eye(n, dtype=bool), rmat)
    cmap = ps.DIVERGING_CMAP.copy()
    cmap.set_bad(color=ps.BG_NEUTRAL)

    im = ax.imshow(
        masked,
        cmap=cmap,
        vmin=-0.6,
        vmax=0.6,
        aspect="equal",
        interpolation="nearest",
        zorder=1,
    )

    for k in np.arange(-0.5, n, 1.0):
        ax.axhline(k, color="white", lw=0.7, zorder=2)
        ax.axvline(k, color="white", lw=0.7, zorder=2)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(
                    j,
                    i,
                    "\u2014",
                    ha="center",
                    va="center",
                    size=ps.TEXT_PT,
                    color=ps.GRAY_MID,
                    zorder=3,
                )
            else:
                r = rmat[i, j]
                p = pmat[i, j]
                txt_col = "white" if abs(r) >= 0.34 else "#111111"
                label = f"{ps.fmt_minus(f'{r:.2f}')}\n{ps.fmt_star(p)}"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    size=ps.TEXT_PT,
                    color=txt_col,
                    linespacing=1.02,
                    zorder=3,
                )

    labels = [ps.AXIS_LABELS[a] for a in ps.AXIS_ORDER]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=42, ha="right", size=ps.TEXT_PT)
    ax.set_yticks(range(n))
    if show_ylabels:
        ax.set_yticklabels(labels, size=ps.TEXT_PT)
    else:
        ax.set_yticklabels([])

    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(0.5)
        ax.spines[sp].set_color("#333333")

    ax.tick_params(
        direction="out",
        length=2.0,
        width=0.5,
        pad=1.8,
        labelsize=ps.TEXT_PT,
        left=show_ylabels,
        bottom=True,
    )
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_title(title, size=ps.TEXT_PT, weight="bold", pad=3.0)
    return im

def _draw_stage_badge(ax, x, y, num_str):
    circ = Circle((x, y), radius=1.9, facecolor="#2B3A4A", edgecolor="none", zorder=4)
    ax.add_patch(circ)
    ax.text(x, y + 0.08, num_str, ha="center", va="center", size=ps.TEXT_PT, weight="bold", color="white", zorder=5)

def main():
    ps.init_style()

    flow_c = pd.read_csv(os.path.join(DATA_DIR, "fig1_flow_charls.csv"))
    flow_e = pd.read_csv(os.path.join(DATA_DIR, "fig1_flow_elsa.csv"))
    corr_df = pd.read_csv(os.path.join(DATA_DIR, "fig1_axis_corr.csv"))

    # Canvas: 183 x 120 mm
    H = 120.0
    cv = ps.Canvas(ps.FIG_W_MM, H)

    # ROW 1: Panel A (3-Stage Visual Hierarchy) & Panel B (STROBE Flowchart)
    cv.panel_label(1.5, 1.8, "A")
    axA = cv.raw_ax(2.0, 1.5, 92.0, 63.0)
    axA.text(48.0, 2.0, "Study design & multidimensional endotyping", ha="center", va="center", size=ps.TEXT_PT, weight="bold")

    # Left stage spine & numbered badges (1, 2, 3)
    spine_x = 2.6
    axA.plot([spine_x, spine_x], [15.8, 57.6], color="#D0D5DC", lw=0.7, zorder=1)
    _draw_stage_badge(axA, spine_x, 15.8, "1")
    _draw_stage_badge(axA, spine_x, 39.2, "2")
    _draw_stage_badge(axA, spine_x, 57.6, "3")

    # Stage 1: Dual-cohort longitudinal milestone rails inside 1 crisp frame
    s1_x, s1_y, s1_w, s1_h = 6.2, 4.8, 85.0, 22.0
    ps.draw_rbox(axA, s1_x, s1_y, s1_w, s1_h, "", fc="#FAFBFC", ec="#D6DCE4", lw=0.5, rounding=0.9, zorder=1)

    x_prior, x_land, x_fu = 33.5, 58.0, 82.0
    axA.text(x_prior, 7.1, "Prior wave", ha="center", va="center", size=ps.TEXT_PT, color="#444444", zorder=3)
    axA.text(x_land, 7.1, "Landmark (5 axes)", ha="center", va="center", size=ps.TEXT_PT, weight="bold", color="#111111", zorder=3)
    axA.text(x_fu, 7.1, "Follow-up", ha="center", va="center", size=ps.TEXT_PT, color="#444444", zorder=3)

    # Vertical dashed landmark anchor across both tracks
    axA.plot([x_land, x_land], [9.2, 26.0], color="#D6DCE4", lw=0.6, ls="--", zorder=2)

    tracks = [
        {
            "y": 12.2,
            "name": "CHARLS",
            "sub": "(China)",
            "color": ps.COHORT_COLORS["CHARLS"],
            "dur1": "4 yr",
            "dur2": "6.0 yr",
            "w1": "W1 (2011)",
            "w2": "W3 (2015; 9,843)",
            "w3": "W4\u2013W6",
        },
        {
            "y": 21.0,
            "name": "ELSA",
            "sub": "(England)",
            "color": ps.COHORT_COLORS["ELSA"],
            "dur1": "4 yr",
            "dur2": "10.2 yr",
            "w1": "W2 (04/05)",
            "w2": "W4 (08/09; 2,694)",
            "w3": "W5\u2013W10",
        },
    ]

    for tr in tracks:
        yy = tr["y"]
        col = tr["color"]
        axA.text(8.0, yy - 1.1, tr["name"], ha="left", va="center", size=ps.TEXT_PT, weight="bold", color=col, zorder=3)
        axA.text(8.0, yy + 1.8, tr["sub"], ha="left", va="center", size=ps.TEXT_PT, color=col, zorder=3)

        # Horizontal timeline arrow
        ps.draw_arrow_mm(axA, x_prior, yy, x_fu + 4.8, yy, color=col, lw=1.1, ms=5.5, zorder=3)

        # Prior wave open circle & Landmark filled circle
        c_prior = Circle((x_prior, yy), radius=1.25, facecolor="white", edgecolor=col, linewidth=1.0, zorder=4)
        c_land = Circle((x_land, yy), radius=1.75, facecolor=col, edgecolor="white", linewidth=0.75, zorder=5)
        axA.add_patch(c_prior)
        axA.add_patch(c_land)

        # Duration tags cleanly resting on the rail with #FAFBFC pill halo
        for mx, dur in ((0.5 * (x_prior + x_land), tr["dur1"]), (71.5, tr["dur2"])):
            axA.text(
                mx,
                yy,
                dur,
                ha="center",
                va="center",
                size=ps.TEXT_PT,
                color=col,
                bbox=dict(facecolor="#FAFBFC", edgecolor="none", pad=0.6),
                zorder=6,
            )

        # Milestone labels beneath rail
        axA.text(x_prior, yy + 3.5, tr["w1"], ha="center", va="center", size=ps.TEXT_PT, color="#333333", zorder=3)
        axA.text(x_land, yy + 3.5, tr["w2"], ha="center", va="center", size=ps.TEXT_PT, weight="bold", color="#111111", zorder=3)
        axA.text(x_fu, yy + 3.5, tr["w3"], ha="center", va="center", size=ps.TEXT_PT, color="#333333", zorder=3)

    # Connector Stage 1 -> Stage 2
    ps.draw_arrow_mm(axA, 48.7, 27.0, 48.7, 29.5, color="#333333", lw=0.6, ms=4.8)

    # Stage 2: 5 Top-Accented Physiological Axes -> Bracket -> 4 Named Endotypes
    ax_w = 16.2
    ax_gap = 1.0
    axis_accent = ["#3B6EA5", "#2A9D8F", "#E09128", "#C84B31", "#5C6B73"]
    card_centers = []
    for idx, key in enumerate(ps.AXIS_ORDER):
        bx = s1_x + idx * (ax_w + ax_gap)
        card_centers.append(bx + ax_w / 2.0)
        ps.draw_rbox(axA, bx, 29.7, ax_w, 7.1, "", fc="#FFFFFF", ec="#CFD5DE", lw=0.5, rounding=0.5, zorder=2)
        # Solid top accent bar (1.1 mm tall)
        axA.add_patch(Rectangle((bx + 0.15, 29.7), ax_w - 0.3, 1.1, facecolor=axis_accent[idx], edgecolor="none", zorder=3))
        axA.text(bx + ax_w / 2.0, 32.4, ps.AXIS_LABELS[key], ha="center", va="center", size=ps.TEXT_PT, weight="bold", color="#111111", zorder=4)
        axA.text(bx + ax_w / 2.0, 35.1, ps.AXIS_SUBLABELS[key], ha="center", va="center", size=ps.TEXT_PT, color="#444444", zorder=4)

    # Converging bracket + K-means label
    y_br = 37.8
    for cx_card in card_centers:
        axA.plot([cx_card, cx_card], [36.9, y_br], color="#8C95A2", lw=0.5, zorder=2)
    axA.plot([card_centers[0], card_centers[-1]], [y_br, y_br], color="#8C95A2", lw=0.55, zorder=2)
    ps.draw_arrow_mm(axA, 48.7, y_br, 48.7, 42.1, color="#333333", lw=0.6, ms=4.8, zorder=2)
    axA.text(
        48.7,
        39.5,
        "Within-sex Z K-means (K = 4)",
        ha="center",
        va="center",
        size=ps.TEXT_PT,
        weight="bold",
        color="#222222",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.7),
        zorder=4,
    )

    # 4 Named Endotype Badges with Left Color Accent Strip
    ew = 20.4
    egap = (s1_w - 4 * ew) / 3.0
    endo_bg = {1: "#EDF3F8", 2: "#EAF5F3", 3: "#FDF6EB", 4: "#FBEFEE"}
    endo_sub = {1: "Resilient", 2: "Inflam-Adip", 3: "Vasc-Sarc", 4: "Multi-High"}
    for ei in (1, 2, 3, 4):
        ex = s1_x + (ei - 1) * (ew + egap)
        col = ps.ENDOTYPE_COLORS[ei]
        ps.draw_rbox(axA, ex, 42.3, ew, 6.7, "", fc=endo_bg[ei], ec=col, lw=0.55, rounding=0.6, zorder=2)
        axA.add_patch(Rectangle((ex, 42.3), 2.3, 6.7, facecolor=col, edgecolor="none", zorder=3))
        tx = ex + 2.3 + (ew - 2.3) / 2.0
        axA.text(tx, 44.3, f"E{ei}", ha="center", va="center", size=ps.TEXT_PT, weight="bold", color=col, zorder=4)
        axA.text(tx, 47.2, endo_sub[ei], ha="center", va="center", size=ps.TEXT_PT, color="#222222", zorder=4)

    # Stage 3: 3-Branch Forked Connector -> 3 Outcome & Validation Modules
    ow = 27.4
    ogap = (s1_w - 3 * ow) / 2.0
    out_centers = [s1_x + idx * (ow + ogap) + ow / 2.0 for idx in range(3)]
    y_fork = 50.8
    axA.plot([48.7, 48.7], [49.1, y_fork], color="#555555", lw=0.55, zorder=2)
    axA.plot([out_centers[0], out_centers[-1]], [y_fork, y_fork], color="#555555", lw=0.55, zorder=2)
    for ocx in out_centers:
        ps.draw_arrow_mm(axA, ocx, y_fork, ocx, 52.8, color="#444444", lw=0.55, ms=4.5, zorder=2)

    outcomes_meta = [
        ("Competing-risk risks", "CMM, CVD, DM & death"),
        ("Dynamic progression", "Multi-state & 4-yr shifts"),
        ("W6 & 5-fold CV", "Dementia, hosp & \u0394C"),
    ]
    for idx, (line1, line2) in enumerate(outcomes_meta):
        ox = s1_x + idx * (ow + ogap)
        ps.draw_rbox(axA, ox, 53.0, ow, 9.2, "", fc="#F6F7F9", ec="#7C8694", lw=0.5, rounding=0.7, zorder=2)
        axA.text(ox + ow / 2.0, 55.8, line1, ha="center", va="center", size=ps.TEXT_PT, weight="bold", color="#111111", zorder=3)
        axA.text(ox + ow / 2.0, 59.3, line2, ha="center", va="center", size=ps.TEXT_PT, color="#333333", zorder=3)

    # Panel B: Side-by-Side STROBE Participant Flow (values pulled from CSVs)
    cv.panel_label(95.5, 1.8, "B")
    axB = cv.raw_ax(96.0, 1.5, 85.5, 63.0)
    axB.text(21.2, 2.0, "CHARLS (Discovery)", ha="center", va="center", size=ps.TEXT_PT, weight="bold", color=ps.COHORT_COLORS["CHARLS"])
    axB.text(64.3, 2.0, "ELSA (Validation)", ha="center", va="center", size=ps.TEXT_PT, weight="bold", color=ps.COHORT_COLORS["ELSA"])

    c_n = flow_c["n_remaining"].astype(int).tolist()
    c_ex = flow_c["n_excluded"].astype(int).tolist()
    e_n = flow_e["n_remaining"].astype(int).tolist()
    e_ex = flow_e["n_excluded"].astype(int).tolist()

    steps_charls = [
        (f"Aged \u226545 yr (n = {c_n[0]:,})", None, False),
        (f"Fasting HGI & RCII (n = {c_n[1]:,})", f"\u2212{c_ex[1]:,} no blood", False),
        (f"Complete 5 axes (n = {c_n[2]:,})", f"\u2212{c_ex[2]:,} missing axes", False),
        (f"No cancer & \u22651 FU (n = {c_n[4]:,})", f"\u2212{c_ex[3]+c_ex[4]:,} cancer/no FU", False),
        (f"Landmark cohort (n = {c_n[5]:,})", f"\u2212{c_ex[5]:,} missing covs", True),
    ]
    steps_elsa = [
        (f"Aged \u226545 yr (n = {e_n[0]:,})", None, False),
        (f"Fasting HGI & RCII (n = {e_n[1]:,})", f"\u2212{e_ex[1]:,} no blood", False),
        (f"Complete 5 axes (n = {e_n[2]:,})", f"\u2212{e_ex[2]:,} missing axes", False),
        (f"No cancer & \u22651 FU (n = {e_n[4]:,})", f"\u2212{e_ex[3]+e_ex[4]:,} cancer/no FU", False),
        (f"Landmark cohort (n = {e_n[5]:,})", f"\u2212{e_ex[5]:,} missing covs", True),
    ]

    for col_idx, (steps, bx, col_color, hi_fc) in enumerate(
        [
            (steps_charls, 0.8, ps.COHORT_COLORS["CHARLS"], "#DCE7F2"),
            (steps_elsa, 43.9, ps.COHORT_COLORS["ELSA"], "#D9EEEA"),
        ]
    ):
        bw = 40.8
        bh = 5.6
        pitch = 9.1
        y0 = 4.8
        cx = bx + bw / 2.0
        for si, (label, excl, is_hi) in enumerate(steps):
            yy = y0 + si * pitch
            if si > 0 and excl:
                arr_x = bx + 9.5
                ps.draw_arrow_mm(axB, arr_x, yy - 3.4, arr_x, yy - 0.15, color=col_color, lw=0.55)
                axB.text(arr_x + 1.5, yy - 1.8, excl, ha="left", va="center", size=ps.TEXT_PT, color="#404040")
            ps.draw_rbox(
                axB,
                bx,
                yy,
                bw,
                bh,
                label,
                fc=hi_fc if is_hi else "#FFFFFF",
                ec=col_color if is_hi else "#777777",
                lw=0.75 if is_hi else 0.5,
                weight="bold" if is_hi else "normal",
                rounding=0.7,
            )

        yy_sub = 49.6
        ps.draw_arrow_mm(axB, cx, 47.0, cx, yy_sub - 0.2, color=col_color, lw=0.55)
        n_arr = c_n if col_idx == 0 else e_n
        sub_txt = (
            f"CMM-free: {n_arr[6]:,} | CVD-free: {n_arr[7]:,}\n"
            f"Multi-state: {n_arr[8]:,} | Trans: {n_arr[9]:,}"
        )
        ps.draw_rbox(
            axB,
            bx,
            yy_sub,
            bw,
            12.6,
            sub_txt,
            fc="#F6F7F9",
            ec=col_color,
            lw=0.5,
            rounding=0.7,
            linespacing=1.22,
        )

    # ROW 2: Panels C, D (5x5 Heatmaps) & Panel E (Cross-Cohort Concordance)
    # All share identical top (73.5 mm) and plot height (34.5 mm)
    y_row2_lbl = 67.5
    y_ax2 = 73.5
    h_ax2 = 34.5

    cv.panel_label(1.5, y_row2_lbl, "C")
    axC = cv.ax(14.5, y_ax2, h_ax2, h_ax2)
    r_c, p_c = _build_corr_matrix(corr_df[corr_df["cohort"] == "CHARLS_W3"])
    imC = _draw_corr_heatmap(axC, r_c, p_c, "CHARLS W3 (n = 9,843)", show_ylabels=True)

    cv.panel_label(51.5, y_row2_lbl, "D")
    axD = cv.ax(53.5, y_ax2, h_ax2, h_ax2)
    r_e, p_e = _build_corr_matrix(corr_df[corr_df["cohort"] == "ELSA_W4"])
    _draw_corr_heatmap(axD, r_e, p_e, "ELSA W4 (n = 2,694)", show_ylabels=False)

    cax = cv.ax(90.5, y_ax2, 2.3, h_ax2)
    cb = cv.fig.colorbar(imC, cax=cax, orientation="vertical", ticks=[-0.6, -0.3, 0.0, 0.3, 0.6])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(direction="out", length=1.8, width=0.5, pad=1.8, labelsize=ps.TEXT_PT)
    cb.ax.set_yticklabels([ps.fmt_minus(f"{v:.1f}") if v != 0 else "0.0" for v in [-0.6, -0.3, 0.0, 0.3, 0.6]])
    cb.set_label("Spearman r", size=ps.TEXT_PT, labelpad=2.5)

    # Panel E: Cross-Cohort Axis Correlation Concordance (10 upper-triangle pairs)
    cv.panel_label(106.5, y_row2_lbl, "E")
    axE = cv.ax(119.5, y_ax2, 61.5, h_ax2)

    pairs_c, pairs_e, pair_names = [], [], []
    n_ax = len(ps.AXIS_ORDER)
    for i in range(n_ax):
        for j in range(i + 1, n_ax):
            pairs_c.append(r_c[i, j])
            pairs_e.append(r_e[i, j])
            pair_names.append(f"{ps.AXIS_LABELS[ps.AXIS_ORDER[i]]}\u2013{ps.AXIS_LABELS[ps.AXIS_ORDER[j]]}")
    pairs_c = np.asarray(pairs_c)
    pairs_e = np.asarray(pairs_e)
    r_conc, p_conc = pearsonr(pairs_c, pairs_e)

    axE.axhline(0, color="#CCCCCC", lw=0.45, zorder=0)
    axE.axvline(0, color="#CCCCCC", lw=0.45, zorder=0)
    axE.plot([-0.6, 0.6], [-0.6, 0.6], ls="--", lw=0.55, color="#777777", zorder=1)

    slope, intercept = np.polyfit(pairs_c, pairs_e, 1)
    xs_fit = np.array([-0.55, 0.42])
    axE.plot(xs_fit, slope * xs_fit + intercept, lw=0.85, color=ps.ENDOTYPE_COLORS[4], alpha=0.85, zorder=2)

    axE.scatter(
        pairs_c,
        pairs_e,
        s=22,
        color=ps.COHORT_COLORS["CHARLS"],
        edgecolors="white",
        linewidths=0.55,
        zorder=4,
    )

    callout_cfg = {
        "RCII\u2013WHtR": (0.16, 0.50, "right", "center"),
        "ePWV\u2013WHtR": (-0.10, 0.34, "right", "center"),
        "WHtR\u2013HGS/BMI": (-0.56, -0.04, "left", "center"),
        "RCII\u2013HGS/BMI": (-0.14, -0.46, "center", "top"),
    }
    for xc, ye, pname in zip(pairs_c, pairs_e, pair_names):
        if pname in callout_cfg:
            tx, ty, ha, va = callout_cfg[pname]
            axE.annotate(
                pname,
                xy=(xc, ye),
                xytext=(tx, ty),
                ha=ha,
                va=va,
                size=ps.TEXT_PT,
                color="#222222",
                arrowprops=dict(arrowstyle="-", lw=0.45, color="#777777", shrinkA=1.2, shrinkB=2.4),
                zorder=5,
            )

    axE.text(
        0.96,
        0.06,
        f"Concordance $\\it{{r}}$ = {r_conc:.2f}\n{ps.fmt_p(p_conc)}",
        transform=axE.transAxes,
        ha="right",
        va="bottom",
        size=ps.TEXT_PT,
        linespacing=1.18,
    )

    ticks = [-0.6, -0.3, 0.0, 0.3, 0.6]
    axE.set_xlim(-0.62, 0.62)
    axE.set_ylim(-0.62, 0.62)
    axE.set_xticks(ticks)
    axE.set_yticks(ticks)
    axE.set_xticklabels([ps.fmt_minus(f"{t:.1f}") if t != 0 else "0.0" for t in ticks])
    axE.set_yticklabels([ps.fmt_minus(f"{t:.1f}") if t != 0 else "0.0" for t in ticks])
    axE.set_xlabel("Spearman $\\it{r}$ (CHARLS W3)", size=ps.TEXT_PT, labelpad=2.0)
    axE.set_ylabel("Spearman $\\it{r}$ (ELSA W4)", size=ps.TEXT_PT, labelpad=2.0)
    axE.set_title("Cross-cohort axis correlation concordance", size=ps.TEXT_PT, weight="bold", pad=3.0)
    ps.style_l_axis(axE)
    ps.save_fig(cv, FIG_DIR, "Figure1")

if __name__ == "__main__":
    main()
