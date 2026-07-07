"""
Model training with MLflow experiment tracking and model registry.

What this script does:
  1. Trains 3 classifiers (LR, RandomForest, XGBoost) with GridSearchCV
  2. Logs every run to MLflow: params, metrics, ROC curve, confusion matrix
  3. Picks the best model by ROC-AUC, saves it to models/best_model.joblib
  4. Registers the best model in the MLflow Model Registry

Usage:
  # Start MLflow server first (in a separate terminal)
  mlflow server --host 0.0.0.0 --port 5000

  # Then run training
  python -m src.train
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MODEL_PATH,
    RANDOM_STATE,
)
from src.data_preparation import build_preprocessor, prepare_data

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

logger = logging.getLogger(__name__)

# ── Model catalogue ───────────────────────────────────────────────────────────
# Each entry defines the sklearn estimator and the hyperparameter grid.
# Grid keys use the "model__" prefix because the estimator sits inside
# a Pipeline step named "model".

MODELS: Dict[str, Dict[str, Any]] = {
    "LogisticRegression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "param_grid": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__solver": ["lbfgs", "liblinear"],
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0
        ),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1],
        },
    },
}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


# ── Artifact helpers ──────────────────────────────────────────────────────────

def _save_roc_curve(y_true, y_proba, model_name: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name=model_name)
    ax.set_title(f"ROC Curve — {model_name}")
    path = out_dir / f"roc_{model_name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return path


def _save_confusion_matrix(y_true, y_pred, model_name: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, display_labels=["No Disease", "Disease"]
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    path = out_dir / f"cm_{model_name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return path


# ── Single-model training run ─────────────────────────────────────────────────

def train_and_log(
    model_name: str,
    config: Dict[str, Any],
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Train one classifier with GridSearchCV and log everything to MLflow.
    Returns the best pipeline and its hold-out metrics.
    """
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    preprocessor = data["preprocessor"]

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", config["estimator"]),
    ])

    search = GridSearchCV(
        pipeline,
        param_grid=config["param_grid"],
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )

    with mlflow.start_run(run_name=model_name):
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        # Strip the pipeline prefix from param keys for readability
        clean_params = {
            k.replace("model__", ""): v for k, v in search.best_params_.items()
        }
        mlflow.log_params(clean_params)
        mlflow.log_metric("cv_best_roc_auc", float(search.best_score_))
        mlflow.log_metrics(metrics)

        # Save and log plot artifacts
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            roc_path = _save_roc_curve(y_test, y_proba, model_name, tmp_path)
            cm_path = _save_confusion_matrix(y_test, y_pred, model_name, tmp_path)
            mlflow.log_artifact(str(roc_path), artifact_path="plots")
            mlflow.log_artifact(str(cm_path), artifact_path="plots")

        mlflow.sklearn.log_model(best, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        logger.info(
            "%-20s roc_auc=%.4f  run_id=%s", model_name, metrics["roc_auc"], run_id
        )

    return {"model_name": model_name, "pipeline": best, "metrics": metrics}


# ── Best-model selection and registry ────────────────────────────────────────

def register_best_model(results: List[Dict[str, Any]]) -> None:
    """Persist the best model locally and register it in the MLflow Model Registry."""
    best = max(results, key=lambda r: r["metrics"]["roc_auc"])
    logger.info(
        "Best model: %s  (roc_auc=%.4f)", best["model_name"], best["metrics"]["roc_auc"]
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], MODEL_PATH)
    logger.info("Saved to %s", MODEL_PATH)

    with mlflow.start_run(run_name=f"register__{best['model_name']}"):
        mlflow.sklearn.log_model(
            best["pipeline"],
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
        )


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s"
    )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    data = prepare_data()
    results = []

    for model_name, config in MODELS.items():
        logger.info("Training %s ...", model_name)
        results.append(train_and_log(model_name, config, data))

    register_best_model(results)

    print("\n=== Results (sorted by ROC-AUC) ===")
    for r in sorted(results, key=lambda x: x["metrics"]["roc_auc"], reverse=True):
        m = r["metrics"]
        print(
            f"  {r['model_name']:<20s}  "
            f"roc_auc={m['roc_auc']:.4f}  "
            f"f1={m['f1']:.4f}  "
            f"acc={m['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
