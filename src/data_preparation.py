"""
Data preparation for the Heart Disease UCI (Cleveland) dataset.

Pipeline:
  load_from_ucimlrepo → binarize_target → basic_cleaning → split_features_target
  → train_test_split → build_preprocessor

Data is fetched directly from the UCI ML Repository using the ucimlrepo package.
The local CSV loader is preserved below (commented out) as a fallback.

Missing values in 'ca' and 'thal' are handled inside the ColumnTransformer
pipeline via SimpleImputer — not dropped upfront — so imputation is fitted on
train data only and correctly applied to test/production data.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    # COLUMN_NAMES,   # only needed for local CSV loader
    # DATA_DIR,       # only needed for local CSV loader
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)


# ── Loading via ucimlrepo (active) ────────────────────────────────────────────

def load_from_ucimlrepo() -> pd.DataFrame:
    """
    Fetch the Heart Disease (Cleveland) dataset from the UCI ML Repository.

    Returns a single DataFrame with 13 feature columns + 'num' target column,
    matching the same schema that the rest of the pipeline expects.

    ucimlrepo already handles:
      - Column naming
      - '?' → NaN conversion
      - Combined Cleveland + other sites (we get all rows; Cleveland is the
        standard 303-row subset; the full fetch returns ~920 rows across sites)
    """
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=45)      # Heart Disease dataset, UCI id=45
    X = dataset.data.features           # DataFrame: 13 feature columns
    y = dataset.data.targets            # DataFrame: 'num' column (0–4)

    df = X.copy()
    df["num"] = y.iloc[:, 0].values     # attach target; iloc avoids hard-coding column name
    return df


# ── Loading from local CSV (commented out — kept as fallback) ─────────────────

# def load_raw_data(path: Path) -> pd.DataFrame:
#     """Read raw Cleveland CSV (no header, '?' marks missing values)."""
#     from src.config import COLUMN_NAMES
#     df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
#     return df


# ── Cleaning ──────────────────────────────────────────────────────────────────

def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert multi-class 'num' (0-4) → binary 'target' (0=no disease, 1=disease)."""
    df = df.copy()
    df[TARGET_COL] = (df["num"] > 0).astype(int)
    return df.drop(columns=["num"])


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and rows where the target is missing."""
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    return df.reset_index(drop=True)


# ── Feature / target split ────────────────────────────────────────────────────

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COL].astype(int)
    return X, y


# ── Preprocessing pipeline ────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer:
    - Numeric  → median imputation → StandardScaler
    - Categorical → mode imputation → OneHotEncoder (ignores unseen categories)
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


# ── High-level entrypoint ─────────────────────────────────────────────────────

def prepare_data(path: Path = None) -> Dict[str, Any]:
    """
    Run the full preparation pipeline.
    Data is fetched from UCI ML Repository via ucimlrepo.

    Returns:
        {X_train, X_test, y_train, y_test, preprocessor}
    """
    df = load_from_ucimlrepo()

    # ── Local CSV fallback (commented out) ──────────────────────────────────
    # if path is None:
    #     path = DATA_DIR / "heart_disease.csv"
    # df = load_raw_data(path)

    df = binarize_target(df)
    df = basic_cleaning(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": build_preprocessor(),
    }


if __name__ == "__main__":
    data = prepare_data()
    print("Train :", data["X_train"].shape, data["y_train"].shape)
    print("Test  :", data["X_test"].shape, data["y_test"].shape)
    print("Class distribution (train):\n", data["y_train"].value_counts(normalize=True))
