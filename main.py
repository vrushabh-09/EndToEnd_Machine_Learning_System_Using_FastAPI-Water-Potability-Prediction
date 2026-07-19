"""
Water Potability Prediction API.

A FastAPI service wrapping a scikit-learn pipeline (median-imputer ->
standard-scaler -> classifier) trained on the Kaggle "Water Potability"
dataset. See README.md for how to (re)train the model and run this service.
"""
import logging
import os
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.schemas import HealthResponse, PredictionResponse, WaterSample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("water_potability_api")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "models", "model.pkl"))
STATIC_DIR = os.path.join(BASE_DIR, "static")

FEATURE_ORDER = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]

ml_state = {"model": None, "model_type": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the model once, fail loudly (but don't crash the whole
    # process) if it's missing or corrupt so /health can report it.
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        ml_state["model"] = model
        ml_state["model_type"] = type(model).__name__
        logger.info("Model loaded from %s (%s)", MODEL_PATH, ml_state["model_type"])
    except Exception:
        logger.exception("Failed to load model at startup")
        ml_state["model"] = None
        ml_state["model_type"] = None
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Water Potability Prediction",
    description="Predicts whether a water sample is safe to drink from 9 water-quality measurements.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s -> %s (%.1f ms)",
        request.method, request.url.path, response.status_code, duration_ms,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again or contact support."},
    )


@app.get("/", include_in_schema=False)
def root_page():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to the Water Potability Prediction API. See /docs."}


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health():
    return HealthResponse(
        status="ok" if ml_state["model"] is not None else "degraded",
        model_loaded=ml_state["model"] is not None,
        model_type=ml_state["model_type"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(sample: WaterSample):
    if ml_state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check /health, or retrain with `python src/train.py`.",
        )

    data = sample.model_dump()
    row = pd.DataFrame([{col: data[col] for col in FEATURE_ORDER}])

    try:
        proba = ml_state["model"].predict_proba(row)[0]
        prob_potable = float(proba[1])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    potable = prob_potable >= 0.5
    margin = abs(prob_potable - 0.5)
    if margin >= 0.3:
        confidence = "high"
    elif margin >= 0.1:
        confidence = "medium"
    else:
        confidence = "low"

    return PredictionResponse(
        potable=potable,
        label="Potable" if potable else "Not potable",
        probability_potable=round(prob_potable, 4),
        confidence=confidence,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
