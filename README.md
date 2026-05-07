# Ovarian Cancer Treatment and Insurance Analysis

An interactive analysis of treatment patterns and survival outcomes for ovarian cancer patients, with a focus on how insurance characteristics relate to the use of PARP inhibitors and bevacizumab.

**Live notebook:** https://adam-katz.github.io/ovarian-cancer-analysis/

---

## Research Question

How do insurance characteristics (payer type, plan design) relate to treatment patterns for ovarian cancer patients receiving PARP inhibitors and/or bevacizumab?

---

## Data

De-identified claims and registry data from TriNetX, covering 23,636 patients with ovarian cancer diagnoses. Six source files:

| File | Rows | Description |
|------|------|-------------|
| `diagnosis.parquet` | 27.7M | ICD-10 diagnosis codes per encounter |
| `medication_ingredient.parquet` | 26.1M | Medication ingredient records |
| `member_enrollment.parquet` | 70K | Insurance enrollment periods |
| `patient.parquet` | 23.6K | Demographics and death dates |
| `tumor.parquet` | 163.8K | Tumor staging and site |
| `tumor_properties.parquet` | 171K | Receptor status, grade, other properties |

Raw data files are not committed to this repository. The notebook runs against a pre-filtered subset (`data-filtered/`) that contains only the rows and columns needed for the analysis.

---

## Analysis Sections

1. **Descriptive Statistics** — cohort characteristics: age, race, ethnicity, marital status, payer type
2. **Treatment Patterns** — PARP inhibitor and bevacizumab use; treated vs. untreated rates; time from diagnosis to treatment
3. **Insurance and Treatment** — treatment rates broken down by payer type and plan design
4. **Statistical Analysis** — chi-square tests, logistic regression for treatment likelihood by insurance type
5. **Survival Analysis** — Kaplan-Meier curves at 2-year and 5-year horizons by treatment status, insurance type, and age group; log-rank tests; Cox proportional hazards models
6. **Tumor Sub-Analysis** — interactive KM plot with toggles for stage, histology, treatment status, age bin, and payer type
7. **Mediation Analysis** — whether insurance type affects survival directly or through treatment as a mediator
8. **Enrollment / Death Signal Consistency** — data quality checks: patients with post-death enrollment, patients with long gaps in enrollment but no recorded death

---

## Running Locally

Requires Python 3.11+.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install marimo

# Full analysis (requires raw data in data-2026-04-22/)
marimo run analysis_2026-04-22.py

# Filtered version (requires data-filtered/)
marimo run analysis_filtered.py
```

---

## Deployment

The live notebook runs entirely in the browser via [Pyodide](https://pyodide.org) (Python compiled to WebAssembly) — no server required after the initial page load.

**Build and deploy:**

```bash
# Export the notebook as a self-contained WASM app
marimo export html-wasm analysis_filtered.py -o ovarian_cancer_analysis/index.html --mode run

# Push to main — GitHub Actions deploys automatically
git push
```

The GitHub Actions workflow (`.github/workflows/deploy.yml`) uploads the `ovarian_cancer_analysis/` directory to GitHub Pages on every push to `main`.

At runtime the browser:
1. Boots Pyodide 0.27.7 in a Web Worker
2. Installs `plotly`, `lifelines`, `statsmodels` via micropip
3. Fetches the filtered parquet files over HTTP from the same GitHub Pages origin
4. Runs the full analysis reactively in-browser

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.3.3 | Data manipulation |
| numpy | 2.3.5 | Numerical operations |
| plotly | 6.5.0 | Interactive charts |
| scipy | 1.16.3 | Statistical tests |
| statsmodels | 0.14.6 | Regression models |
| lifelines | 0.30.1 | Kaplan-Meier and Cox survival analysis |
| pyarrow | 22.0.0 | Parquet file I/O |
| marimo | — | Reactive notebook framework |
