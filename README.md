# Water Potability Prediction

An end-to-end ML system that predicts whether a water sample is safe to
drink from 9 water-quality measurements (pH, hardness, solids, chloramines,
sulfate, conductivity, organic carbon, trihalomethanes, turbidity).

```
data/            water_potability.csv (Kaggle "Water Potability" dataset, 3276 rows)
src/
  eda.py         exploratory analysis -> reports/*.png + eda_summary.md
  train.py       preprocessing + model selection + training -> models/model.pkl
  schemas.py     Pydantic request/response models with validation
models/
  model.pkl      trained sklearn Pipeline (imputer -> scaler -> classifier)
  metrics.json   held-out test metrics + cross-validation results
reports/         EDA plots, confusion matrix, feature importance
static/
  index.html     browser UI ("lab reader" form) served at /
tests/
  test_api.py    pytest suite (17 tests) against the FastAPI app
main.py          FastAPI application
Dockerfile, docker-compose.yml, nixpacks.toml   deployment
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # includes EDA/train/test extras

# 1. (optional) regenerate EDA plots from the raw data
python src/eda.py

# 2. (optional) retrain the model — already-trained model.pkl is committed,
#    so you only need this if you change the data or pipeline
python src/train.py

# 3. run the API
uvicorn main:app --reload
```

Open **http://localhost:8000** for the web form, or **http://localhost:8000/docs**
for interactive Swagger docs.

### Docker

```bash
docker compose up --build
```

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the web UI |
| `/health` | GET | Liveness + whether the model loaded successfully |
| `/predict` | POST | Returns a potability prediction |
| `/docs` | GET | Swagger UI |

`POST /predict` example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "ph": 7.1, "Hardness": 196.9, "Solids": 20927.8,
        "Chloramines": 7.1, "Sulfate": 333.1, "Conductivity": 421.9,
        "Organic_carbon": 14.2, "Trihalomethanes": 66.6, "Turbidity": 3.9
      }'
```

```json
{
  "potable": false,
  "label": "Not potable",
  "probability_potable": 0.25,
  "confidence": "medium"
}
```

Every field is **optional** — the trained pipeline includes a median imputer,
so a partial reading (e.g. missing Sulfate, which is ~24% missing in the
training data) still gets a prediction. At least one field must be provided,
and every provided value is range-checked (e.g. `ph` must be 0–14) so bad
input fails fast with a `422` instead of a silent bad prediction.

## What changed from the original repo

The original repo was a bare FastAPI wrapper around a pre-trained `model.pkl`
with no training code, dataset, tests, or docs. This version adds:

- **Data + reproducible training** (`src/eda.py`, `src/train.py`): the dataset
  is fetched, profiled, and the model is retrained from scratch inside a
  proper `sklearn.Pipeline` (median imputer → standard scaler → classifier),
  instead of shipping an opaque pickle with unknown provenance.
- **Model comparison**: logistic regression, random forest, and gradient
  boosting are compared with 5-fold cross-validated F1; the best is promoted
  and evaluated on a held-out test set.
- **Input validation**: physically-sensible bounds per field (via Pydantic),
  rather than accepting any float silently.
- **Graceful error handling**: the app no longer crashes if `model.pkl` is
  missing — `/health` reports `degraded` and `/predict` returns a clear `503`
  instead of an unhandled exception.
- **Structured JSON response** (label + probability + confidence bucket)
  instead of a bare string.
- **Logging** of every request (method, path, status, latency) and of model
  load failures.
- **Tests**: 17 pytest cases covering health, valid/partial/invalid input,
  and boundary values.
- **A web UI** so the API is usable without `curl` or Swagger.
- **Docker** + updated `nixpacks.toml`/`runtime.txt` for deployment.

## Model performance

Trained on an 80/20 stratified split (2620 train / 656 test rows).
Random forest was selected by 5-fold CV F1 (0.417, beating logistic
regression at 0.416 and gradient boosting at 0.356) and evaluated on the held-out set:

| Metric | Value |
|---|---|
| Accuracy | 0.674 |
| F1 (potable class) | 0.412 |
| ROC AUC | 0.661 |
| Precision / Recall (potable) | 0.69 / 0.29 |
| Precision / Recall (not potable) | 0.67 / 0.92 |

**Read this honestly, not optimistically.** This dataset is well known in the
ML community to have a weak relationship between the 9 chemical measurements
and the potability label — published work on this exact dataset (logistic
regression, SVM, random forest, XGBoost, even deep nets) consistently lands
in the 60-70% accuracy range, similar to what a single number (predict
"not potable" for everyone) with some tuning would achieve given the ~61/39
class split. The model here is recall-heavy for "not potable" (92%) but
misses most true "potable" samples (29% recall), i.e. it errs on the side of
caution — which is reasonable for a health-relevant decision, but means you
should **not** treat this as a substitute for certified water testing. See
`reports/eda_summary.md` and `reports/correlation_heatmap.png` for why: no
single feature or pair of features is strongly correlated with the label.

## Retraining on new/updated data

Drop a replacement `water_potability.csv` (same 10 columns) into `data/` and
run:

```bash
python src/train.py
```

This overwrites `models/model.pkl` and `models/metrics.json`. Restart the API
(or redeploy) to pick up the new model — it's loaded once at startup.

## Tests

```bash
pytest tests/ -v
```

17 tests covering: root/health endpoints, a full valid prediction, partial
(imputed) input, empty-body rejection, out-of-range values (per-field), and
wrong-type input.
