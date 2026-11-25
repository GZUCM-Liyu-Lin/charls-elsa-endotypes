# charls-elsa-endotypes

Code accompanying the cross-national prospective study of five-axis cardiometabolic-vascular-sarcopenic endotypes, four-year phenotype transitions, and incident cardiometabolic multimorbidity in CHARLS (2015-2023) and ELSA (2008-2023).

## Structure

- `config.py`: Input and output directory paths via environment variables.
- `run_all.py`: Sequential runner for statistical modeling and figure generation.
- `src/01_build_charls.py`: Harmonized CHARLS Wave 1-6 data extraction and axis construction.
- `src/02_build_elsa.py`: External validation ELSA Wave 2-10 extraction and harmonization.
- `src/03_clustering.py`: Within-sex standardized K-means endotyping, bootstrap stability, and ELSA replication.
- `src/04_survival_multistate.py`: Competing-risk Cox/Fine-Gray, multi-state Markov, transition, subgroup, and Wave 6 models.
- `src/05_prediction_sensitivity.py`: Five-fold cross-validated discrimination, calibration, DCA, and sensitivity analyses.
- `src/fig1.py` - `src/fig6.py`, `src/plot_style.py`: Vector and 600 dpi TIFF figure scripts.

## Setup

```bash
pip install -r requirements.txt
```

Set raw data locations (CHARLS archives/CSVs under `CHARLS_DIR`; ELSA `.dta` files under `ELSA_DIR`):

```bash
export CHARLS_DIR=/path/to/charls
export ELSA_DIR=/path/to/elsa
export OUT_DIR=./output
python run_all.py --figures
```

## Data Access

Raw datasets are not distributed in this repository. CHARLS data are available upon registration at https://charls.pku.edu.cn/, and ELSA data (SN 5050) are available via the UK Data Service (https://ukdataservice.ac.uk/) and Gateway to Global Aging Data (https://g2aging.org/).
