# 02 - Agentic ML Pipeline (Auto-EDA to Model Card)

[![CI](https://github.com/milos-plavsic/agentic-ml-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/milos-plavsic/agentic-ml-pipeline/actions/workflows/ci.yml)
[![Python3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

An AI-driven machine learning workflow that inspects tabular data, selects modeling strategy, trains candidate models, explains outcomes, and exports a reproducible model card.

## Real-world data (education sector)

The default training path uses **`data/student-mat.csv`** from the UCI *Student Performance* dataset (secondary school mathematics, Portugal): [UCI ML Repository — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance). Citation: P. Cortez and A. Silva, *Using Data Mining to Predict Secondary School Student Performance*, FUBUTEC 2008.

The starter trains a **Random Forest regressor** to predict **final math grade `G3`** from questionnaire and prior-grade features (one-hot encoded categoricals).

## Quickstart

```bash
make install
make run
make api
make test
```

Docker API: `make docker-api`.

## Related repos (tabular ensembles)

- [tabular-ensemble-arena](https://github.com/milos-plavsic/tabular-ensemble-arena) — **Random Forest**, **XGBoost**, and **CatBoost** on numeric tabular data with stratified CV and a leaderboard API.
- [categorical-boost-lab](https://github.com/milos-plavsic/categorical-boost-lab) — mixed numeric + categorical features: **OneHot + RF**, **Ordinal + XGBoost**, **native CatBoost**.
- [education-nn-predictor](https://github.com/milos-plavsic/education-nn-predictor) — **PyTorch MLP** on the same UCI student-math cohort (`G3` regression).

## API

- OpenAPI docs: `http://127.0.0.1:8000/docs`
- Health: `GET /health`
- Pipeline run: `POST /v1/pipeline/run` with JSON body `{"dataset_name":"uci_student_math"}` (only this dataset key is supported today)

## Architecture

```mermaid
flowchart LR
  D[Dataset] --> Prof[Profiler]
  Prof --> R[Task router]
  R --> F[Feature engineer]
  F --> T[Model trainer]
  T --> E[Evaluator]
  E --> X[Explainability]
  X --> M[Model card]
```

## Why This Project Stands Out

- Bridges LLM planning with concrete ML execution.
- Clearly demonstrates decision routing in a graph pipeline.
- Produces tangible outputs: report, artifacts, model card.

## Core Capabilities

- Dataset profiling and quality diagnostics.
- Dynamic task routing (classification vs regression vs time-series).
- Feature engineering recommendations and application.
- Automated model comparison with cross-validation.
- SHAP explainability snapshots and drift baseline capture.

## Suggested Tech Stack

- Python 3.11+
- `pandas`, `polars`, `scikit-learn`, `xgboost`, `shap`, `mlflow`
- `langgraph` for orchestration
- FastAPI for serving reports

## Architecture (Graph)

`ingest_data -> data_profiler -> task_router -> feature_engineer -> model_trainer -> evaluator -> explainability -> model_card_writer -> artifact_publisher`

## Usage Suggestions

- Build internal benchmark sets and track score changes by commit.
- Add strict data leakage checks in the evaluator node.
- Export markdown + JSON reports for downstream automation.

## Portfolio Additions

- Include side-by-side experiment comparison dashboard.
- Show "why model A won" rationale from evaluator node.
- Add Github Actions CI to run smoke training on sample data.

## Milestones

- `v0.1`: profiling + baseline model.
- `v0.2`: branching router and model comparison.
- `v0.3`: explainability and model card generation.
- `v1.0`: API deployment + experiment tracking.

## Demo Scenarios

1. Customer churn prediction from SaaS usage logs.
2. House-price regression with robust preprocessing.
3. Defect detection on structured manufacturing metrics.
