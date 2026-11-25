from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Rectangle

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DERIVED_DIR, FIG_DATA_DIR as DATA_DIR, FIGURES_DIR as FIG_DIR, TABLES_DIR as TAB_DIR
import plot_style as ps

def _cov_ellipse(ax, x, y, color, n_std=1.51, lw=0.85):
    """Draw 68% bivariate covariance ellipse."""
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2.0 * n_std * np.sqrt(np.maximum(vals, 1e-9))
    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=w,
        height=h,
        angle=theta,
        facecolor="none",
        edgecolor=color,
        linewidth=lw,
        linestyle="--",
        zorder=4,
    )
    ax.add_patch(ell)

def main():
    ps.init_style()

    k_df = pd.read_csv(os.path.join(DATA_DIR, "fig2_kmeans_k_metrics.csv")).sort_values("k")
    pca_df = pd.read_csv(os.path.join(DATA_DIR, "fig2_pca_coords.csv"))
    cen_ch = pd.read_csv(os.path.join(DATA_DIR, "fig2_centroids_charls.csv")).sort_values("endotype")
    cen_ep = pd.read_csv(os.path.join(DATA_DIR, "fig2_centroids_elsa_proj.csv")).sort_values("endotype")
    cen_ed = pd.read_csv(os.path.join(DATA_DIR, "fig2_centroids_elsa_denovo.csv")).sort_values("endotype")
    rep_qc = pd.read_csv(os.path.join(TAB_DIR, "table_elsa_replication_qc.csv")).sort_values("endotype")

    H = 144.0
    cv = ps.Canvas(ps.FIG_W_MM, H)

    # ROW 1: Panels A, B, C (y_lbl = 1.8, row top = 8.0, total h = 44.0 mm)
    y_lbl1 = 1.8
    y_r1 = 8.0
    h_sub = 19.0
    gap_v = 6.0  # A2/C2 top = 33.0, bottom = 52.0 (aligned with B)

    # --- Panel A ---
    cv.panel_label(1.5, y_lbl1, "A")
    axA1 = cv.ax(13.5, y_r1, 44.0, h_sub)
    ks = k_df["k"].values
    ari_m = k_df["boot_ari_median"].values
    ari_lo = k_df["boot_ari_q25"].values
    ari_hi = k_df["boot_ari_q75"].values
    sil = k_df["silhouette"].values

    axA1.axvline(4, color=ps.ENDOTYPE_COLORS[4], ls="--", lw=0.65, zorder=1)
    axA1.fill_between(ks, ari_lo, ari_hi, color=ps.COHORT_COLORS["CHARLS"], alpha=0.16, lw=0, zorder=2)
    axA1.plot(ks, ari_m, color=ps.COHORT_COLORS["CHARLS"], lw=0.95, marker="o", ms=3.2, mfc=ps.COHORT_COLORS["CHARLS"], mec="white", mew=0.4, zorder=3)
    axA1.plot([4], [ari_m[ks == 4][0]], marker="o", ms=4.2, mfc=ps.ENDOTYPE_COLORS[4], mec="white", mew=0.5, zorder=4)
    axA1.text(
        4.18,
        0.96,
        f"K = 4\nARI = {ari_m[ks == 4][0]:.2f}",
        ha="left",
        va="top",
        size=ps.TEXT_PT,
        color=ps.ENDOTYPE_COLORS[4],
        weight="bold",
        linespacing=1.1,
    )
    axA1.set_xlim(1.7, 6.3)
    axA1.set_ylim(0.72, 1.02)
    axA1.set_xticks(ks)
    axA1.set_xticklabels([])
    axA1.set_yticks([0.75, 0.85, 0.95])
    axA1.set_ylabel("Bootstrap ARI", size=ps.TEXT_PT, labelpad=2.0)
    axA1.set_title("Cluster stability & validity", size=ps.TEXT_PT, weight="bold", pad=3.0)
    ps.style_l_axis(axA1)

    axA2 = cv.ax(13.5, y_r1 + h_sub + gap_v, 44.0, h_sub)
    axA2.axvline(4, color=ps.ENDOTYPE_COLORS[4], ls="--", lw=0.65, zorder=1)
    axA2.plot(ks, sil, color="#2A9D8F", lw=0.95, marker="s", ms=3.0, mfc="#2A9D8F", mec="white", mew=0.4, zorder=3)
    axA2.plot([4], [sil[ks == 4][0]], marker="s", ms=3.8, mfc=ps.ENDOTYPE_COLORS[4], mec="white", mew=0.5, zorder=4)
    axA2.set_xlim(1.7, 6.3)
    axA2.set_ylim(0.145, 0.235)
    axA2.set_xticks(ks)
    axA2.set_yticks([0.16, 0.19, 0.22])
    axA2.set_xlabel("Number of clusters (K)", size=ps.TEXT_PT, labelpad=2.0)
    axA2.set_ylabel("Silhouette", size=ps.TEXT_PT, labelpad=2.0)
    ps.style_l_axis(axA2)

    # --- Panel B: PCA 2D Projection ---
    cv.panel_label(61.5, y_lbl1, "B")
    axB = cv.ax(72.0, y_r1, 47.0, 2 * h_sub + gap_v)
    v1 = pca_df["pc1_var_exp"].iloc[0] * 100.0
    v2 = pca_df["pc2_var_exp"].iloc[0] * 100.0

    for ei in (1, 2, 3, 4):
        sub = pca_df[pca_df["endotype"] == ei]
        col = ps.ENDOTYPE_COLORS[ei]
        axB.scatter(sub["pc1"], sub["pc2"], s=1.8, c=col, alpha=0.35, lw=0, rasterized=True, zorder=2)
        _cov_ellipse(axB, sub["pc1"].values, sub["pc2"].values, col, n_std=1.51, lw=0.85)
        cx, cy = sub["pc1"].mean(), sub["pc2"].mean()
        axB.scatter([cx], [cy], s=26, marker="D", c=col, edgecolors="white", linewidths=0.65, zorder=5)

    badge_offsets = {1: (0.0, -0.68), 2: (0.25, 0.72), 3: (-0.55, -0.78), 4: (-0.45, 0.75)}
    for ei in (1, 2, 3, 4):
        sub = pca_df[pca_df["endotype"] == ei]
        cx, cy = sub["pc1"].mean(), sub["pc2"].mean()
        dx, dy = badge_offsets[ei]
        col = ps.ENDOTYPE_COLORS[ei]
        axB.text(
            cx + dx,
            cy + dy,
            f"E{ei}",
            ha="center",
            va="center",
            size=ps.TEXT_PT,
            weight="bold",
            color=col,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=col, linewidth=0.55, alpha=0.92),
            zorder=6,
        )

    axB.set_xlim(-4.7, 4.7)
    axB.set_ylim(-3.9, 3.9)
    xticks_b = [-4, -2, 0, 2, 4]
    yticks_b = [-3, 0, 3]
    axB.set_xticks(xticks_b)
    axB.set_yticks(yticks_b)
    axB.set_xticklabels([ps.fmt_minus(str(v)) for v in xticks_b])
    axB.set_yticklabels([ps.fmt_minus(str(v)) for v in yticks_b])
    axB.set_xlabel(f"PC1 ({v1:.1f}% variance)", size=ps.TEXT_PT, labelpad=2.0)
    axB.set_ylabel(f"PC2 ({v2:.1f}% variance)", size=ps.TEXT_PT, labelpad=2.0)
    axB.set_title("CHARLS PCA projection (n = 9,843)", size=ps.TEXT_PT, weight="bold", pad=3.0)
    ps.style_l_axis(axB)

    # --- Panel C: Prevalence (%) & Sex Balance (%) ---
    cv.panel_label(122.5, y_lbl1, "C")
    axC1 = cv.ax(137.0, y_r1, 44.0, h_sub)
    cohort_rows = [
        ("De Novo", cen_ed),
        ("ELSA Proj", cen_ep),
        ("CHARLS", cen_ch),
    ]
    for ri, (cname, cdf) in enumerate(cohort_rows):
        left = 0.0
        for ei in (1, 2, 3, 4):
            pct = float(cdf.loc[cdf["endotype"] == ei, "pct"].iloc[0])
            axC1.barh(ri, pct, left=left, height=0.56, color=ps.ENDOTYPE_COLORS[ei], edgecolor="white", lw=0.6, zorder=2)
            axC1.text(left + pct / 2.0, ri, f"{pct:.0f}%", ha="center", va="center", size=ps.TEXT_PT, weight="bold", color="white", zorder=3)
            left += pct
    axC1.set_xlim(0, 100)
    axC1.set_ylim(-0.55, 2.55)
    axC1.set_yticks([0, 1, 2])
    axC1.set_yticklabels([r[0] for r in cohort_rows], size=ps.TEXT_PT)
    axC1.set_xticks([0, 25, 50, 75, 100])
    # Title carries (%) so zero collision with C2 legend below!
    axC1.set_title("Cross-cohort prevalence (%)", size=ps.TEXT_PT, weight="bold", pad=3.0)
    ps.style_l_axis(axC1)

    axC2 = cv.ax(137.0, y_r1 + h_sub + gap_v, 44.0, h_sub)
    axC2.axhline(50.0, color="#999999", ls="--", lw=0.55, zorder=1)
    ex_pos = np.array([1, 2, 3, 4], dtype=float)
    f_ch = cen_ch["female_pct"].values
    f_ep = cen_ep["female_pct"].values
    f_ed = cen_ed["female_pct"].values
    axC2.plot(ex_pos, f_ch, color=ps.COHORT_COLORS["CHARLS"], lw=0.85, marker="o", ms=3.4, mfc=ps.COHORT_COLORS["CHARLS"], mec="white", mew=0.4, label="CHARLS", zorder=3)
    axC2.plot(ex_pos, f_ep, color=ps.COHORT_COLORS["ELSA"], lw=0.85, ls="--", marker="D", ms=3.0, mfc="white", mec=ps.COHORT_COLORS["ELSA"], mew=0.7, label="ELSA Proj", zorder=3)
    axC2.plot(ex_pos, f_ed, color="#666666", lw=0.75, ls=":", marker="^", ms=3.0, mfc="white", mec="#666666", mew=0.65, label="De Novo", zorder=2)

    axC2.set_xlim(0.55, 4.45)
    axC2.set_ylim(43, 66)
    axC2.set_xticks(ex_pos)
    axC2.set_xticklabels(["E1", "E2", "E3", "E4"])
    axC2.set_yticks([45, 50, 55, 60])
    axC2.set_xlabel("Sex-balanced endotypes", size=ps.TEXT_PT, labelpad=2.0)
    axC2.set_ylabel("Female (%)", size=ps.TEXT_PT, labelpad=2.0)
    leg_c2 = axC2.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.15), ncol=3, handlelength=1.0, columnspacing=0.5, handletextpad=0.25, borderaxespad=0.0)
    for t in leg_c2.get_texts():
        t.set_size(ps.TEXT_PT)
    ps.style_l_axis(axC2)

    # ROW 2: Panel D (5-Axis Z-Score Profile) & Panel E (Cross-Cohort Strip)
    # y_lbl2 = 59.0, y_r2 = 66.5, h_r2 = 62.5 mm -> bottom = 129.0 mm
    y_lbl2 = 59.0
    y_r2 = 66.5
    h_r2 = 62.5

    cv.panel_label(1.5, y_lbl2, "D")
    # Compact frameless legend cleanly inside [15.0 .. 94.0] mm
    axD_leg = cv.raw_ax(15.0, 59.5, 81.5, 5.2)
    leg_xs = [0.0, 18.5, 39.5, 59.5]
    for idx, ei in enumerate((1, 2, 3, 4)):
        lx = leg_xs[idx]
        col = ps.ENDOTYPE_COLORS[ei]
        axD_leg.plot([lx, lx + 2.8], [2.6, 2.6], color=col, lw=1.1)
        axD_leg.plot([lx + 1.4], [2.6], marker="o", ms=3.6, mfc=col, mec="white", mew=0.45)
        axD_leg.text(lx + 3.5, 2.6, ps.ENDOTYPE_LABELS[ei], ha="left", va="center", size=ps.TEXT_PT, color="#111111")

    axD = cv.ax(15.5, y_r2, 81.0, h_r2)
    z_cols = [f"z_{a}" for a in ps.AXIS_ORDER]
    n_axes = len(ps.AXIS_ORDER)
    y_bases = np.arange(n_axes)[::-1]
    offsets = {1: 0.30, 2: 0.10, 3: -0.10, 4: -0.30}

    for yi in y_bases:
        if yi % 2 == 1:
            axD.axhspan(yi - 0.46, yi + 0.46, color="#F5F6F8", lw=0, zorder=0)
    axD.axvline(0.0, color="#666666", ls="--", lw=0.6, zorder=1)

    for ax_idx, (a_key, zc) in enumerate(zip(ps.AXIS_ORDER, z_cols)):
        yb = y_bases[ax_idx]
        for ei in (1, 2, 3, 4):
            val = float(cen_ch.loc[cen_ch["endotype"] == ei, zc].iloc[0])
            yy = yb + offsets[ei]
            col = ps.ENDOTYPE_COLORS[ei]
            axD.plot([0.0, val], [yy, yy], color=col, lw=1.15, zorder=2)
            axD.plot([val], [yy], marker="o", ms=4.0, mfc=col, mec="white", mew=0.45, zorder=3)
            txt = ps.fmt_minus(f"{val:+.2f}") if val < 0 else f"+{val:.2f}"
            if val >= 0:
                axD.text(val + 0.045, yy, txt, ha="left", va="center", size=ps.TEXT_PT, color=col, zorder=4)
            else:
                axD.text(val - 0.045, yy, txt, ha="right", va="center", size=ps.TEXT_PT, color=col, zorder=4)

    axD.set_xlim(-1.35, 1.52)
    axD.set_ylim(-0.50, 4.50)
    axD.set_yticks(y_bases)
    axD.set_yticklabels([f"{ps.AXIS_LABELS[a]}\n({ps.AXIS_SUBLABELS[a]})" for a in ps.AXIS_ORDER], size=ps.TEXT_PT, linespacing=1.05)
    xt_d = [-1.0, -0.5, 0.0, 0.5, 1.0]
    axD.set_xticks(xt_d)
    axD.set_xticklabels([ps.fmt_minus(f"{v:.1f}") if v != 0 else "0.0" for v in xt_d])
    axD.set_xlabel("Within-sex standardized centroid (Z-score)", size=ps.TEXT_PT, labelpad=2.0)
    ps.style_l_axis(axD)

    # --- Panel E: 3-Tier Cross-Cohort Replication Strip Heatmap ---
    cv.panel_label(100.5, y_lbl2, "E")
    hm_x = 112.0
    hm_w = 43.0
    ari_rep = float(rep_qc["overall_proj_vs_denovo_ari"].iloc[0])
    cv.text(hm_x + hm_w / 2.0, 60.8, f"ELSA replication (ARI = {ari_rep:.2f})", size=ps.TEXT_PT, weight="bold", ha="center", va="center")

    axE = cv.ax(hm_x, y_r2, hm_w, h_r2)
    mat = np.full((15, 5), np.nan)
    row_map = []
    for ei in (1, 2, 3, 4):
        base_r = (ei - 1) * 4
        for offset, (tag, cdf) in enumerate([("CH", cen_ch), ("EP", cen_ep), ("ED", cen_ed)]):
            vals = [float(cdf.loc[cdf["endotype"] == ei, zc].iloc[0]) for zc in z_cols]
            mat[base_r + offset, :] = vals
            row_map.append((base_r + offset, ei, tag))

    cmap_e = ps.DIVERGING_CMAP.copy()
    cmap_e.set_bad(color="white")
    imE = axE.imshow(mat, cmap=cmap_e, vmin=-1.2, vmax=1.2, aspect="auto", interpolation="nearest", zorder=1)

    for r in range(15):
        if r not in (3, 7, 11):
            axE.axhline(r - 0.5, color="white", lw=0.45, zorder=2)
            axE.axhline(r + 0.5, color="white", lw=0.45, zorder=2)
    for c in np.arange(-0.5, 5, 1.0):
        axE.axvline(c, color="white", lw=0.65, zorder=2)

    for ei in (1, 2, 3, 4):
        y0 = (ei - 1) * 4 - 0.5
        axE.add_patch(Rectangle((-0.5, y0), 5.0, 3.0, fill=False, ec="#333333", lw=0.5, zorder=3))

    axE.set_yticks([r for r, _, _ in row_map])
    axE.set_yticklabels([tag for _, _, tag in row_map], size=ps.TEXT_PT)
    axE.set_xticks(range(5))
    axE.set_xticklabels([ps.AXIS_LABELS[a] for a in ps.AXIS_ORDER], rotation=42, ha="right", size=ps.TEXT_PT)
    axE.tick_params(direction="out", length=1.6, width=0.5, pad=1.5, labelsize=ps.TEXT_PT, left=True, bottom=True)
    for sp in ("top", "right", "left", "bottom"):
        axE.spines[sp].set_visible(False)

    axE_left = cv.raw_ax(100.5, y_r2, 5.5, h_r2)
    axE_right = cv.raw_ax(hm_x + hm_w + 1.2, y_r2, 13.5, h_r2)
    cv.text(hm_x + hm_w + 1.6, 65.0, "Cos sim", size=ps.TEXT_PT, weight="bold", ha="left", va="baseline")

    for ei in (1, 2, 3, 4):
        r0 = (ei - 1) * 4
        y_top_mm = r0 / 15.0 * h_r2
        blk_h_mm = 3.0 / 15.0 * h_r2
        yc_mm = y_top_mm + blk_h_mm / 2.0
        col = ps.ENDOTYPE_COLORS[ei]
        axE_left.add_patch(Rectangle((4.0, y_top_mm + 0.3), 1.2, blk_h_mm - 0.6, facecolor=col, edgecolor="none"))
        axE_left.text(3.3, yc_mm, f"E{ei}", ha="right", va="center", size=ps.TEXT_PT, weight="bold", color=col)

        cos_v = float(rep_qc.loc[rep_qc["endotype"] == ei, "centroid_cosine_sim"].iloc[0])
        r_v = float(rep_qc.loc[rep_qc["endotype"] == ei, "centroid_pearson_r"].iloc[0])
        axE_right.text(0.4, yc_mm - 1.6, f"{cos_v:.3f}", ha="left", va="center", size=ps.TEXT_PT, weight="bold", color="#111111")
        axE_right.text(0.4, yc_mm + 1.8, f"(r={r_v:.2f})", ha="left", va="center", size=ps.TEXT_PT, color="#555555")

    caxE = cv.ax(171.8, y_r2, 2.2, h_r2)
    cbE = cv.fig.colorbar(imE, cax=caxE, orientation="vertical", ticks=[-1.2, -0.6, 0.0, 0.6, 1.2])
    cbE.outline.set_linewidth(0.4)
    cbE.ax.tick_params(direction="out", length=1.8, width=0.5, pad=1.6, labelsize=ps.TEXT_PT)
    cbE.ax.set_yticklabels([ps.fmt_minus(f"{v:.1f}") if v != 0 else "0.0" for v in [-1.2, -0.6, 0.0, 0.6, 1.2]])
    cbE.set_label("Mean Z-score", size=ps.TEXT_PT, labelpad=2.0)

    cv.text(101.0, 142.0, "CH: CHARLS W3 | EP: ELSA Proj | ED: ELSA De Novo", size=ps.TEXT_PT, color="#444444")
    ps.save_fig(cv, FIG_DIR, "Figure2")

if __name__ == "__main__":
    main()
