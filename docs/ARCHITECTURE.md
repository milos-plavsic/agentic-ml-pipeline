# AutoML / Benchmark Studio — Architecture

`agentic-ml-pipeline` is the integration host for tabular ML workflows.

## Components

| Layer | Responsibility |
|-------|----------------|
| LangGraph pipeline | ingest → profile → route → train → evaluate → explain → model card |
| Studio registry | plugin dispatch (`ensemble_benchmark`, `nn_regression`, `automl_pipeline`) |
| Dataset service | CSV upload, schema inference, persisted registry |
| HTTP API | `/v1/studio/*`, `/v1/datasets/*`, OpenAPI, Prometheus metrics |
| Web UI | `/ui` upload and benchmark console |

## Plugins

| Plugin | Capability |
|--------|------------|
| `ensemble_benchmark` | RF / XGBoost / CatBoost cross-validation |
| `nn_regression` | PyTorch MLP regression baseline |
| `automl_pipeline` | Full LangGraph AutoML path (`uci_student_math`) |

## Experiment tracking

MLflow is included in `requirements.txt`. Set `MLFLOW_TRACKING_URI` (for example `http://127.0.0.1:5000`) and start a local server:

```bash
docker compose -f docker-compose.mlflow.yml up -d
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Studio runs log params, metrics, and a JSON artifact. Query recent runs via `GET /v1/studio/experiments` and check configuration with `GET /v1/studio/mlflow/status`.

## API

```http
POST /v1/datasets/upload
GET  /v1/studio/plugins
POST /v1/studio/run
GET  /v1/pipeline/status
```
