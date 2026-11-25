from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

MM = 1.0 / 25.4
PT_PER_MM = 72.0 / 25.4
FIG_W_MM = 183.0
TEXT_PT = 7.0
PANEL_PT = 7.5

AXIS_ORDER = ["hgi", "rcii", "epwv", "whtr", "hgs_bmi"]
AXIS_LABELS = {"hgi": "HGI", "rcii": "RCII", "epwv": "ePWV", "whtr": "WHtR", "hgs_bmi": "HGS/BMI"}
AXIS_SUBLABELS = {"hgi": "Glycation", "rcii": "Inflam\u2013lipid", "epwv": "Vascular", "whtr": "Adiposity", "hgs_bmi": "Muscle"}

ENDOTYPE_COLORS = {1: "#3B6EA5", 2: "#2A9D8F", 3: "#E09128", 4: "#C84B31"}
ENDOTYPE_LABELS = {1: "E1: Resilient", 2: "E2: Inflam-Adip", 3: "E3: Vasc-Sarc", 4: "E4: Multi-High"}
COHORT_COLORS = {"CHARLS": "#3B6EA5", "ELSA": "#2A9D8F"}
MODEL_COLORS = {
    "M0_Base": "#4D4D4D",
    "M1_Base_TyG": "#6C8EAD",
    "M2_Base_eGDR": "#52A699",
    "M3_Base_ePWV": "#8C8C8C",
    "M4_Base_Endotypes": "#E09128",
    "M5_Base_All5Axes": "#C84B31",
}
DIVERGING_CMAP = LinearSegmentedColormap.from_list("diverging", ["#2E5E8C", "#FFFFFF", "#B83B2E"])

GRAY_MID = "#595959"
BG_ZEBRA = "#F5F6F8"
BG_NEUTRAL = "#F1F2F4"


class Canvas:
    def __init__(self, w_mm=FIG_W_MM, h_mm=150.0):
        self.w = w_mm
        self.h = h_mm
        self.fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))

    def _rect(self, x, y, w, h):
        return [x / self.w, 1.0 - (y + h) / self.h, w / self.w, h / self.h]

    def ax(self, x, y, w, h, **kw):
        return self.fig.add_axes(self._rect(x, y, w, h), **kw)

    def raw_ax(self, x, y, w, h):
        a = self.fig.add_axes(self._rect(x, y, w, h))
        a.set_xlim(0, w)
        a.set_ylim(h, 0)
        a.axis("off")
        return a

    def text(self, x, y, s, size=None, ha="left", va="baseline", **kw):
        return self.fig.text(x / self.w, 1.0 - y / self.h, s, size=size or TEXT_PT, ha=ha, va=va, **kw)

    def panel_label(self, x, y, letter):
        return self.fig.text(
            x / self.w, 1.0 - y / self.h, letter,
            size=PANEL_PT, weight="bold", family="sans-serif", ha="left", va="top",
        )


def init_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": TEXT_PT,
        "axes.titlesize": TEXT_PT,
        "axes.labelsize": TEXT_PT,
        "xtick.labelsize": TEXT_PT,
        "ytick.labelsize": TEXT_PT,
        "legend.fontsize": TEXT_PT,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
    })


def fmt_minus(text: str) -> str:
    import re
    return re.sub(r"(^|[\s(\[,=])-(\d|\.)", lambda m: f"{m.group(1)}\u2212{m.group(2)}", str(text))


def fmt_p(p, with_p=True) -> str:
    import numpy as np
    if p is None or not np.isfinite(p):
        return r"$\it{P}$ = NA" if with_p else "NA"
    if p < 0.001:
        return r"$\it{P}$ < 0.001" if with_p else "< 0.001"
    return rf"$\it{{P}}$ = {p:.3f}" if with_p else f"{p:.3f}"


def fmt_star(p) -> str:
    import numpy as np
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def fmt_hr(est: float, lo: float, hi: float) -> str:
    return f"{est:.2f} ({lo:.2f}\u2013{hi:.2f})"


def style_l_axis(ax, hide_y=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(not hide_y)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color("black")
    ax.tick_params(direction="out", length=2.2, width=0.6, color="black", labelcolor="black", pad=2.0, labelsize=TEXT_PT)
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.grid(False)


def draw_rbox(ax, x, y, w, h, text, fc="#FFFFFF", ec="#333333", lw=0.5, tc="#111111", weight="normal", rounding=0.8, ha="center", va="center", linespacing=1.16, zorder=3):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(patch)
    if text:
        tx = x + w / 2.0 if ha == "center" else (x + 1.2 if ha == "left" else x + w - 1.2)
        ax.text(tx, y + h / 2.0, text, ha=ha, va=va, size=TEXT_PT, color=tc, weight=weight, linespacing=linespacing, zorder=zorder + 1)
    return patch


def draw_arrow_mm(ax, x0, y0, x1, y1, color="#333333", lw=0.55, ms=5.0, zorder=2):
    arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=0.45,head_width=0.28", mutation_scale=ms, linewidth=lw, color=color, shrinkA=0.0, shrinkB=0.0, zorder=zorder)
    ax.add_patch(arr)
    return arr


def save_fig(cv: Canvas, out_dir: Path, stem: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}_600dpi.png"
    tif_path = out_dir / f"{stem}_600dpi.tif"
    cv.fig.savefig(pdf_path, dpi=600, facecolor="white")
    cv.fig.savefig(png_path, dpi=600, facecolor="white")
    with Image.open(png_path) as im:
        im.convert("RGB").save(tif_path, format="TIFF", compression="tiff_lzw", dpi=(600, 600))
    plt.close(cv.fig)
    return pdf_path, png_path, tif_path
