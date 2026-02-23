import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

num_cols = ["Age", "Credit amount", "Duration"]
nominal_cols = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
ordinal_cols = ["Job"]


def load_and_clean(path="../data/german_credit_data.csv"):
    """Load the German credit dataset, drop unused column, remove NaNs, cast Job to str."""
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"])
    df = df.dropna()
    df["Job"] = df["Job"].astype("str")
    return df


def build_preprocessor():
    """Return a ColumnTransformer that scales numerics, one-hot encodes nominals,
    and ordinal-encodes Job."""
    num_pipeline = Pipeline([("scaler", StandardScaler())])
    nominal_pipeline = Pipeline([("onehot", OneHotEncoder(drop="first"))])
    ordinal_pipeline = Pipeline([("ordinal", OrdinalEncoder())])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("nom", nominal_pipeline, nominal_cols),
        ("ord", ordinal_pipeline, ordinal_cols),
    ])
    return preprocessor
