# Release Notes (2026-04)

## Scope
This release adds end-to-end reporting (statistics + figures), strengthens data loading reliability, and hardens CI.

## Data Source
- UCI Student Performance dataset (ID 320): `student-mat.csv`
- Citation/source retained in project metadata and reports.

## Reporting Added
- New `analysis/` package with:
  - `report.py` (build report artifacts)
  - `plotting.py` (visuals)
  - `stats_utils.py` (metrics/statistics)
  - `json_util.py` (strict JSON serialization)
  - module entrypoint: `python -m analysis`
- Generated outputs:
  - `reports/summary.json`
  - `reports/REPORT.md`
  - `reports/figures/actual_vs_predicted.png`
  - `reports/figures/residuals.png`

## Latest Report Snapshot
- MAE: `1.1749`
- RMSE: `1.9737`
- R^2: `0.8100`
- Residual mean ± std: `0.0592 ± 1.9855`

## Reliability and CI
- Added ZIP-based UCI fetch fallback via `app/uci_fetch.py` (official static dataset archive).
- Ensured local/offline stability with vendored `data/student-mat.csv`.
- Added/updated CI to run:
  - dependency install
  - `pytest`
  - `python -m analysis` smoke step
- Upgraded actions to Node24-compatible versions:
  - `actions/checkout@v6`
  - `actions/setup-python@v6`

## Latest CI Status
- Latest successful run: https://github.com/milos-plavsic/agentic-ml-pipeline/actions/runs/24447651576

## Dependency Notes
- Core stack includes `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `fastapi`, `pytest`, `httpx`.
