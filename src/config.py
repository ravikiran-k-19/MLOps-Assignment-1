"""
Central configuration — all paths, column names, and constants live here.
Import from this module instead of hardcoding values elsewhere.
"""

from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# ── Heart Disease UCI (Cleveland) dataset schema ──────────────────────────────
# Raw file has no header; columns appear in this exact order.
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# Continuous features → StandardScaler
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Discrete/binary features → OneHotEncoder
# 'ca' and 'thal' also carry missing values (encoded as '?' in the raw file)
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

TARGET_COL = "target"  # binarised from raw 'num' column (0 → 0, 1-4 → 1)

# ── Experiment settings ───────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "heart-disease-classification"
MLFLOW_MODEL_NAME = "heart-disease-classifier"

# ── Serialised model artifact ─────────────────────────────────────────────────
MODEL_PATH = MODELS_DIR / "best_model.joblib"
