"""Turn the cleaned players table into a model-ready feature matrix.

Feature set matches the stats the dashboard shows: minutes, goals,
assists, passes completed, tackles, successful dribbles, cards, plus
age, height, preferred foot, position and years left on contract.
"""
import pandas as pd

NUMERIC_FEATURES = [
    "Age",
    "Height",
    "Min_standard",
    "Gls_standard",
    "Ast_standard",
    "Cmp_passing",
    "Tkl_defense",
    "Succ_possession",
    "CrdY_misc",
    "CrdR_misc",
    "contract_years_remaining",
]
CATEGORICAL_FEATURES = ["Foot", "Pos"]
TARGET = "MarketValue"


def add_contract_years_remaining(df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    df = df.copy()
    as_of = as_of or pd.Timestamp.now()
    expires = pd.to_datetime(df.get("ContractExpires"), errors="coerce")
    years_remaining = (expires - as_of).dt.days / 365.25
    df["contract_years_remaining"] = years_remaining.clip(lower=0).fillna(0)
    return df


def build_feature_matrix(df: pd.DataFrame, feature_columns: list[str] | None = None):
    """Return (X, y, feature_columns). Pass feature_columns (from a
    previously trained model) when featurizing new data for prediction,
    so the dummy-encoded columns line up with what the model expects."""
    df = add_contract_years_remaining(df)

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0
    numeric = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"
    categorical = pd.get_dummies(df[CATEGORICAL_FEATURES].fillna("Unknown"), prefix=CATEGORICAL_FEATURES)

    X = pd.concat([numeric, categorical], axis=1)

    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)
    else:
        feature_columns = list(X.columns)

    y = df[TARGET] if TARGET in df.columns else None
    return X, y, feature_columns
