from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ── Column definitions ────────────────────────────────────────────────────────
num_cols     = ["Age", "Credit amount", "Duration"]
nominal_cols = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
ordinal_cols = ["Job"]

_DATA_PATH = Path(__file__).parent.parent / "data" / "german_credit_data.csv"


def load_and_clean(path: Path = _DATA_PATH) -> pd.DataFrame:
    """Load the German Credit dataset, drop the index column, remove rows
    with missing values, and cast Job to string for ordinal encoding."""
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.dropna()
    df["Job"] = df["Job"].astype(str)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Return a fresh ColumnTransformer that scales numerics, one-hot encodes
    nominal features, and ordinal-encodes the Job skill level."""
    num_pipeline = Pipeline([("scaler", StandardScaler())])
    nom_pipeline = Pipeline([("onehot", OneHotEncoder(drop="first"))])
    ord_pipeline = Pipeline([("ordinal", OrdinalEncoder())])

    return ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("nom", nom_pipeline, nominal_cols),
        ("ord", ord_pipeline, ordinal_cols),
    ])
