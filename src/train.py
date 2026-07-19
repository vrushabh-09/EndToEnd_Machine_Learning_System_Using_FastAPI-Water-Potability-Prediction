"""
Training pipeline for the Water Potability classifier.

Steps:
  1. Load data/water_potability.csv
  2. Split train/test (stratified, 80/20)
  3. Build a Pipeline: SimpleImputer(median) -> StandardScaler -> Classifier
     (the imputer lives INSIDE the pipeline so the API can accept raw,
     occasionally-missing readings without any special-casing)
  4. Compare a few candidate models with cross-validation
  5. Fit the best one on the full training set, evaluate on the held-out test set
  6. Save: models/model.pkl (the whole pipeline), models/metrics.json,
     reports/confusion_matrix.png, reports/feature_importance.png

Run:
    python src/train.py
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "water_potability.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]
TARGET = "Potability"
RANDOM_STATE = 42


def build_pipeline(clf):
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    print("Cross-validated F1 (5-fold) on training set:")
    for name, clf in candidates.items():
        pipe = build_pipeline(clf)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        cv_results[name] = {"mean_f1": float(scores.mean()), "std_f1": float(scores.std())}
        print(f"  {name:20s}  F1 = {scores.mean():.3f} (+/- {scores.std():.3f})")

    best_name = max(cv_results, key=lambda k: cv_results[k]["mean_f1"])
    print(f"\nBest model by CV F1: {best_name}")

    best_pipeline = build_pipeline(candidates[best_name])
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    test_metrics = {
        "model": best_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "cv_results": cv_results,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURES,
    }

    print("\nHeld-out test performance:")
    print(f"  Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"  F1:       {test_metrics['f1']:.3f}")
    print(f"  ROC AUC:  {test_metrics['roc_auc']:.3f}")

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, "model.pkl"))
    print(f"\nSaved pipeline to {os.path.join(MODELS_DIR, 'model.pkl')}")

    # Confusion matrix plot
    plt.figure(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Not potable", "Potable"], cmap="Blues"
    )
    plt.title(f"Confusion matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()

    # Feature importance (if supported)
    clf = best_pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values()
        plt.figure(figsize=(7, 5))
        importances.plot(kind="barh", color="#5c8ca8")
        plt.title(f"Feature importance — {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=120)
        plt.close()

    print(f"Saved plots to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
