import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHARLS_DIR = Path(os.environ.get("CHARLS_DIR", ROOT / "data" / "charls"))
ELSA_DIR = Path(os.environ.get("ELSA_DIR", ROOT / "data" / "elsa"))
OUT_DIR = Path(os.environ.get("OUT_DIR", ROOT / "output"))

DERIVED_DIR = OUT_DIR / "derived"
TABLES_DIR = OUT_DIR / "tables"
FIG_DATA_DIR = OUT_DIR / "figure_data"
FIGURES_DIR = Path(os.environ.get("FIGURES_OUT_DIR", OUT_DIR / "figures"))


def ensure_dirs():
    for d in (DERIVED_DIR, TABLES_DIR, FIG_DATA_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
