"""
Unit tests for src/data_preparation.py.

Uses a small in-memory DataFrame fixture so the tests run in CI
without needing the actual data file.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL
from src.data_preparation import (
    basic_cleaning,
    binarize_target,
    build_preprocessor,
    split_features_target,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df():
    """Minimal Cleveland-style DataFrame (14 columns, no header, '?' already NaN)."""
    return pd.DataFrame({
        "age":      [63, 37, 41, 56, 57, 44],
        "sex":      [1,  1,  0,  1,  0,  1],
        "cp":       [3,  2,  1,  1,  0,  2],
        "trestbps": [145, 130, 130, 120, 120, 140],
        "chol":     [233, 250, 204, 236, 354, 235],
        "fbs":      [1,  0,  0,  0,  0,  0],
        "restecg":  [0,  1,  0,  1,  1,  0],
        "thalach":  [150, 187, 172, 178, 163, 180],
        "exang":    [0,  0,  0,  0,  1,  0],
        "oldpeak":  [2.3, 3.5, 1.4, 0.8, 0.6, 1.0],
        "slope":    [0,  0,  2,  2,  2,  1],
        "ca":       [0,  0,  0,  0,  0,  np.nan],  # missing value
        "thal":     [1,  2,  2,  2,  2,  np.nan],  # missing value
        "num":      [0,  0,  0,  1,  1,  2],
    })


# ── Tests: binarize_target ────────────────────────────────────────────────────

def test_binarize_creates_target_column(raw_df):
    result = binarize_target(raw_df)
    assert TARGET_COL in result.columns


def test_binarize_removes_num_column(raw_df):
    result = binarize_target(raw_df)
    assert "num" not in result.columns


def test_binarize_values_are_binary(raw_df):
    result = binarize_target(raw_df)
    assert set(result[TARGET_COL].unique()).issubset({0, 1})


def test_binarize_correct_mapping(raw_df):
    result = binarize_target(raw_df)
    # num=0 → 0; num>0 → 1
    expected = [0, 0, 0, 1, 1, 1]
    assert list(result[TARGET_COL]) == expected


# ── Tests: basic_cleaning ─────────────────────────────────────────────────────

def test_basic_cleaning_drops_duplicate_rows(raw_df):
    df = binarize_target(raw_df)
    df_with_dup = pd.concat([df, df.iloc[:1]], ignore_index=True)
    cleaned = basic_cleaning(df_with_dup)
    assert len(cleaned) == len(df)


def test_basic_cleaning_drops_missing_target(raw_df):
    df = binarize_target(raw_df)
    df.loc[0, TARGET_COL] = np.nan
    cleaned = basic_cleaning(df)
    assert cleaned[TARGET_COL].isna().sum() == 0


# ── Tests: split_features_target ─────────────────────────────────────────────

def test_split_correct_feature_columns(raw_df):
    df = basic_cleaning(binarize_target(raw_df))
    X, y = split_features_target(df)
    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_split_target_column_name(raw_df):
    df = basic_cleaning(binarize_target(raw_df))
    _, y = split_features_target(df)
    assert y.name == TARGET_COL


def test_split_length_matches(raw_df):
    df = basic_cleaning(binarize_target(raw_df))
    X, y = split_features_target(df)
    assert len(X) == len(y)


# ── Tests: build_preprocessor ─────────────────────────────────────────────────

def test_preprocessor_transforms_without_error(raw_df):
    df = basic_cleaning(binarize_target(raw_df))
    X, _ = split_features_target(df)
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == len(X)


def test_preprocessor_no_nan_in_output(raw_df):
    """Imputation inside the pipeline should eliminate all NaNs."""
    df = basic_cleaning(binarize_target(raw_df))
    X, _ = split_features_target(df)
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    assert not np.isnan(X_transformed).any()
